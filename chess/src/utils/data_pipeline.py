"""
Chess data processing and dataset management (pipeline).
Split from src.data to keep preprocessing separate from dataset/dataloader code.
"""

import chess
import chess.pgn
import gc
import io
import os
import hashlib
import json
import pickle
import re
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.utils.data_helpers import ACTION_SIZE, get_position_size


def _phase_desc(phase):
    return f"  {phase}"


def _print_preprocessing_dataset_table(rows):
    if not rows:
        return

    headers = ("Dataset", "Positions", "Status")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))

    def line():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    print(line())
    print("| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line())
    for row in rows:
        print("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
    print(line(), flush=True)


# ==============================================================================
# WORKER COUNT RESOLUTION
# ==============================================================================

def _resolve_workers(value, label):
    """
    Resolve worker count from config.
    Accepts int or strings like "all" to use all CPU cores.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("all", "auto", "max"):
            return max(1, os.cpu_count() or 1)
        try:
            value = int(v)
        except ValueError:
            raise ValueError(f"{label} must be an int or 'all', got: {value}")
    try:
        value = int(value)
    except Exception:
        raise ValueError(f"{label} must be an int or 'all', got: {value}")
    if value <= 0:
        return max(1, os.cpu_count() or 1)
    return value


def _resolve_phase2_chunk_size(config, total_games, phase2_workers):
    """
    Resolve how many games are sent to one Phase 2 process task.

    Windows ProcessPool overhead is high when every game is submitted as a
    separate future, so Phase 2 batches games into deterministic chunks.
    """
    data_cfg = config.get('data', {}) if config else {}
    raw_value = data_cfg.get('phase2_chunk_size', 512)

    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text in {'auto', '0', 'none'}:
            if total_games <= 0:
                return 1
            target_chunks = max(phase2_workers * 32, 1)
            return max(64, min(2048, (total_games + target_chunks - 1) // target_chunks))
        try:
            raw_value = int(text)
        except ValueError as exc:
            raise ValueError(f"phase2_chunk_size must be an int or 'auto', got: {raw_value}") from exc

    try:
        chunk_size = int(raw_value)
    except Exception as exc:
        raise ValueError(f"phase2_chunk_size must be an int or 'auto', got: {raw_value}") from exc

    if chunk_size <= 0:
        return _resolve_phase2_chunk_size({'data': {'phase2_chunk_size': 'auto'}}, total_games, phase2_workers)
    return max(1, chunk_size)


def _normalize_max_games(value):
    """
    Resolve max_games from config.

    Returns:
        int for a hard limit, or None for "use the entire PGN file".
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"max", "all", "full", "entire", "inf", "infinite", "none"}:
            return None
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(
                f"max_games must be an int or one of: max/all/full/inf, got: {value}"
            ) from exc
    try:
        value = int(value)
    except Exception as exc:
        raise ValueError(f"max_games must be an int or 'max', got: {value}") from exc
    if value <= 0:
        return None
    return value


def _get_game_selection_settings(config):
    raw_max_games = config['data'].get('max_games', 100000)
    max_games = _normalize_max_games(raw_max_games)
    full_file_mode = max_games is None
    return {
        'raw_max_games': raw_max_games,
        'max_games': max_games,
        'full_file_mode': full_file_mode,
    }


_EVENT_HEADER_RE = re.compile(br"(?m)^\[Event ")


def _scan_game_start_offsets(pgn_path, max_offsets=None, chunk_size=8 * 1024 * 1024):
    """
    Scan a PGN and return byte offsets for lines that begin with [Event ...

    This is much faster than fully parsing the file and gives exact shard
    boundaries for offset-based parallel parsing.
    """
    offsets = []
    remainder = b""
    remainder_offset = 0
    file_offset = 0

    with open(pgn_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            if remainder:
                data = remainder + chunk
                data_offset = remainder_offset
            else:
                data = chunk
                data_offset = file_offset

            last_newline = data.rfind(b"\n")
            if last_newline == -1:
                remainder = data
                remainder_offset = data_offset
                file_offset += len(chunk)
                continue

            process_chunk = data[: last_newline + 1]
            remainder = data[last_newline + 1 :]
            for match in _EVENT_HEADER_RE.finditer(process_chunk):
                offsets.append(data_offset + match.start())
                if max_offsets is not None and len(offsets) >= max_offsets:
                    return offsets
            remainder_offset = data_offset + last_newline + 1
            file_offset += len(chunk)

    if remainder:
        for match in _EVENT_HEADER_RE.finditer(remainder):
            offsets.append(remainder_offset + match.start())
            if max_offsets is not None and len(offsets) >= max_offsets:
                return offsets

    if not offsets:
        raise RuntimeError(
            f"Could not detect any PGN games in {Path(pgn_path).name} via [Event] headers."
        )

    return offsets

# ==============================================================================
# 🆕 DATASET TRACKING SYSTEM
# ==============================================================================

class DatasetTracker:
    """
    Tracks processed datasets to avoid reprocessing
    Stores metadata about each processed PGN file
    """
    
    def __init__(self, preprocessing_dir):
        """
        Args:
            preprocessing_dir: Path to data/preprocessing directory
        """
        self.preprocessing_dir = Path(preprocessing_dir)
        self.preprocessing_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking file location
        self.tracking_file = self.preprocessing_dir / "processed_datasets.json"
        
        # Load existing tracking data
        self.tracking_data = self._load_tracking()
    
    def _load_tracking(self):
        """Load tracking data from JSON file"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_tracking(self):
        """Save tracking data to JSON file"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
    
    def _compute_file_hash(self, pgn_path):
        """Compute a cheap but robust PGN fingerprint for preprocessing cache."""
        pgn_path = Path(pgn_path)
        hasher = hashlib.md5()
        try:
            stat = pgn_path.stat()
            hasher.update(str(int(stat.st_size)).encode('utf-8'))
            hasher.update(str(int(stat.st_mtime_ns)).encode('utf-8'))
        except OSError:
            pass
        
        with open(pgn_path, 'rb') as f:
            chunk = f.read(10 * 1024 * 1024)
            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _compute_config_hash(self, config):
        """
        Compute hash of processing configuration
        Only includes parameters that affect binary output
        """
        selection = _get_game_selection_settings(config)
        relevant_config = {
            'min_elo': config['data'].get('min_elo', 0),
            'max_games': selection['max_games'] if selection['max_games'] is not None else 'max',
            'max_moves_per_game': config['data'].get('max_moves_per_game', 200),
            'game_filters': config['data'].get('game_filters', {}),
            'action_encoding': 'az_classic_8x8x73_v1',
            'action_size': ACTION_SIZE,
            'wdl_mode': 'always',
        }
        
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_processed_dataset(self, pgn_path, config):
        """
        Check if dataset was already processed with same config
        
        Returns:
            Path to existing binary file if found and compatible, None otherwise
        """
        pgn_path = Path(pgn_path)
        file_hash = self._compute_file_hash(pgn_path)
        config_hash = self._compute_config_hash(config)
        
        # Check if this file+config combo exists
        key = f"{pgn_path.name}_{file_hash}_{config_hash}"
        
        if key in self.tracking_data:
            entry = self.tracking_data[key]
            binary_path = Path(entry['binary_file'])
            metadata_path = Path(entry['metadata_file'])
            
            # Verify files still exist
            if binary_path.exists() and metadata_path.exists():
                return binary_path, metadata_path
        
        return None, None
    
    def register_dataset(self, pgn_path, config, binary_file, metadata_file, total_positions):
        """
        Register a newly processed dataset
        
        Args:
            pgn_path: Path to source PGN file
            config: Processing configuration
            binary_file: Path to output binary file
            metadata_file: Path to metadata file
            total_positions: Number of positions in dataset
        """
        selection = _get_game_selection_settings(config)
        pgn_path = Path(pgn_path)
        file_hash = self._compute_file_hash(pgn_path)
        config_hash = self._compute_config_hash(config)
        
        key = f"{pgn_path.name}_{file_hash}_{config_hash}"
        
        self.tracking_data[key] = {
            'pgn_file': str(pgn_path),
            'pgn_hash': file_hash,
            'config_hash': config_hash,
            'binary_file': str(binary_file),
            'metadata_file': str(metadata_file),
            'total_positions': total_positions,
            'processed_date': datetime.now().isoformat(),
            'config': {
                'min_elo': config['data'].get('min_elo', 0),
                'max_games': selection['max_games'] if selection['max_games'] is not None else 'max',
            }
        }
        
        self._save_tracking()


# ==============================================================================
# AUTO-CLEANUP SYSTEM
# ==============================================================================

def cleanup_intermediate_files(games_data_list, temp_dir):
    """
    Clean up intermediate files after position creation
    
    Args:
        games_data_list: List of (pgn_path, games_data, temp_file) tuples
        temp_dir: Temporary directory containing intermediate files
    """
    print(f"\n{'='*70}")
    print("♻️  Cleaning up intermediate files...")
    print(f"{'='*70}")
    
    cleaned_count = 0
    freed_mb = 0
    
    # Clean up game data temp files
    for pgn_path, games_data, temp_file in games_data_list:
        if temp_file and temp_file.exists():
            size_mb = temp_file.stat().st_size / (1024**2)
            temp_file.unlink()
            cleaned_count += 1
            freed_mb += size_mb
            print(f"  🗑️  Deleted: {temp_file.name} ({size_mb:.1f} MB)")
    
    # Clean up any other temporary files in temp_dir
    if temp_dir.exists():
        for temp_file in temp_dir.glob("*.tmp"):
            if temp_file.exists():
                size_mb = temp_file.stat().st_size / (1024**2)
                temp_file.unlink()
                cleaned_count += 1
                freed_mb += size_mb
                print(f"  🗑️  Deleted: {temp_file.name} ({size_mb:.1f} MB)")
    
    gc.collect()
    
    print(f"\n✅ Cleanup complete!")
    print(f"  • Files deleted: {cleaned_count}")
    print(f"  • Space freed: {freed_mb:.1f} MB")
    print(f"{'='*70}\n")


# ==============================================================================
# SMART DATASET MERGING
# ==============================================================================

def merge_binary_datasets(binary_files, metadata_files, output_binary, output_metadata):
    """
    Merge multiple binary datasets into one
    
    Args:
        binary_files: List of binary file paths
        metadata_files: List of metadata file paths
        output_binary: Output binary file path
        output_metadata: Output metadata file path
    
    Returns:
        Combined metadata dictionary
    """
    print(f"\n{'='*70}")
    print(f"🔗 Merging {len(binary_files)} datasets...")
    print(f"{'='*70}")
    
    # Load all metadata
    all_metadata = []
    total_positions = 0
    
    for meta_file in metadata_files:
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
            all_metadata.append(meta)
            total_positions += meta['total_positions']
            print(f"  • {Path(meta['binary_file']).name}: {meta['total_positions']:,} positions")
    
    # Verify compatibility
    first_meta = all_metadata[0]
    position_size = first_meta['position_size']
    
    for meta in all_metadata[1:]:
        if meta['position_size'] != position_size:
            raise ValueError("Cannot merge incompatible datasets! Different configs detected.")
    
    print(f"\n  ✅ All datasets compatible")
    print(f"  📊 Total positions: {total_positions:,}")
    
    # Merge binary files — rewriting GameIDs to be globally unique across all source files.
    # Each source file's GameIDs start at 0, so after concatenation two different games in
    # different files can share the same GameID.  We fix this by offsetting every GameID
    # by the cumulative position count of all previous files.  Because positions within one
    # source file are ordered by game (all moves of game 0, then game 1, …) and the history
    # walk relies solely on GameID equality to detect game boundaries, the remapped IDs
    # preserve that property while eliminating cross-file collisions.
    print(f"\n  🔨 Writing merged binary file (with GameID remapping)...")
    chunk_size = 100 * 1024 * 1024  # 100 MB chunks
    
    # We need per-position rewriting for GameID remapping, so we process record-by-record.
    # GameID offset: we use the max GameID seen in previous files + 1 as the base for the
    # next file.  This way IDs never collide regardless of how many games each file has.
    game_id_offset = 0  # running offset applied to each file's GameIDs
    
    with open(output_binary, 'wb') as outfile:
        for i, (binary_file, meta) in enumerate(zip(binary_files, all_metadata), 1):
            print(f"     Merging file {i}/{len(binary_files)}: {Path(binary_file).name} "
                  f"(GameID offset: {game_id_offset})")
            
            file_positions = meta['total_positions']
            max_game_id_in_file = 0
            
            with open(binary_file, 'rb') as infile:
                for _ in range(file_positions):
                    record = bytearray(infile.read(position_size))
                    if len(record) < position_size:
                        break  # truncated file, stop
                    
                    # 🔧 v4.5 CRITICAL FIX: Board is now 38B (32B pieces + 6B metadata)
                    # Layout: [Board 38B] + [GameID 4B] + [MoveIdx 2B] + [MoveTarget 2B] + [Outcome 4B] + [MTL 12B]
                    # Read original GameID (uint32 at offset 38, not 36!)
                    original_game_id = struct.unpack('I', record[38:42])[0]
                    max_game_id_in_file = max(max_game_id_in_file, original_game_id)
                    
                    # Write remapped GameID
                    new_game_id = original_game_id + game_id_offset
                    record[38:42] = struct.pack('I', new_game_id)
                    
                    outfile.write(record)
            
            # Next file's IDs start after the highest ID we just wrote
            game_id_offset += max_game_id_in_file + 1
    
    # Create combined metadata
    combined_metadata = {
        'binary_file': str(output_binary),
        'total_positions': total_positions,
        'position_size': position_size,
        'input_planes': 16,  # 🔧 v4.5 FIX: 16 planes (12 pieces + 4 metadata)
        'source_files': [str(Path(meta['binary_file']).name) for meta in all_metadata],
        'merged_date': datetime.now().isoformat()
    }
    
    with open(output_metadata, 'wb') as f:
        pickle.dump(combined_metadata, f)
    
    final_size_mb = output_binary.stat().st_size / (1024**2)
    
    print(f"\n{'='*70}")
    print("✅ Merge Complete!")
    print(f"{'='*70}")
    print(f"  • Output: {output_binary.name}")
    print(f"  • Size: {final_size_mb:.1f} MB")
    print(f"  • Total positions: {total_positions:,}")
    print(f"  • Source files: {len(binary_files)}")
    print(f"{'='*70}\n")
    
    return combined_metadata


# ==============================================================================
# PHASE 1: PGN EXTRACTION (Multi-processing)
# ==============================================================================

def extract_game_data(game):
    """Extract serializable data from chess.pgn.Game"""
    try:
        if game is None:
            return None
        
        white_elo = game.headers.get('WhiteElo', '?')
        black_elo = game.headers.get('BlackElo', '?')
        result = game.headers.get('Result', '*')
        termination = game.headers.get('Termination', '')
        
        moves = []
        for move in game.mainline_moves():
            moves.append(move.uci())
        
        return {
            'white_elo': white_elo,
            'black_elo': black_elo,
            'result': result,
            'termination': termination,
            'moves': moves
        }
    except:
        return None


def parse_games_offset_batch_worker(args):
    """
    PHASE 1 WORKER: Parse a batch of games from a byte offset.

    Used for full-file mode to avoid re-scanning from the start of the PGN in
    every worker.
    """
    pgn_path, start_offset, num_games = args

    import chess.pgn

    games_data = []

    try:
        with open(pgn_path, "rb") as raw_f:
            raw_f.seek(start_offset)
            with io.TextIOWrapper(raw_f, encoding="utf-8", errors="ignore", newline="") as f:
                for _ in range(num_games):
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    game_data = extract_game_data(game)
                    if game_data:
                        games_data.append(game_data)

    except Exception as e:
        print(f"⚠️ Worker error: {e}")

    return games_data


def extract_games_from_pgn_multiprocess(pgn_path, max_games, phase1_workers):
    """
    PHASE 1: Extract games using multi-processing
    """
    if phase1_workers <= 1:
        return extract_games_sequential(pgn_path, max_games)

    if max_games is None:
        game_offsets = _scan_game_start_offsets(pgn_path)
        effective_max_games = len(game_offsets)
    else:
        game_offsets = _scan_game_start_offsets(pgn_path, max_offsets=max_games)
        effective_max_games = len(game_offsets)
        if effective_max_games < max_games:
            print(
                f"  • PGN ended early: detected {effective_max_games:,} games "
                f"(requested {max_games:,})"
            )

    games_per_worker = (effective_max_games + phase1_workers - 1) // phase1_workers
    tasks = []
    for i in range(phase1_workers):
        start_idx = i * games_per_worker
        if start_idx >= effective_max_games:
            break
        num_games = min(games_per_worker, effective_max_games - start_idx)
        tasks.append((pgn_path, game_offsets[start_idx], num_games))

    game_chunks = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {
            executor.submit(parse_games_offset_batch_worker, task): i
            for i, task in enumerate(tasks)
        }

        with tqdm(total=len(futures), desc=_phase_desc("Phase 1: PGN parsing")) as pbar:
            for future in as_completed(futures):
                task_idx = futures[future]
                games_chunk = future.result()
                game_chunks[task_idx] = games_chunk
                pbar.update(1)
                pbar.set_postfix({'games': sum(len(chunk or []) for chunk in game_chunks)})

    all_games = []
    for chunk in game_chunks:
        if chunk:
            all_games.extend(chunk)

    return all_games[:effective_max_games]


def extract_games_sequential(pgn_path, max_games):
    """
    PHASE 1: Sequential extraction (fallback)
    """
    games_data = []
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        with tqdm(total=max_games, desc=_phase_desc("Phase 1: PGN parsing")) as pbar:
            game_count = 0
            while max_games is None or game_count < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                game_data = extract_game_data(game)
                if game_data:
                    games_data.append(game_data)
                    game_count += 1
                    pbar.update(1)
    
    return games_data


def _dedupe_games(games_data):
    """Remove duplicate games based on moves+result signature."""
    seen = set()
    deduped = []
    dupes = 0
    
    for game in games_data:
        moves = game.get('moves') or []
        result = game.get('result', '')
        sig_src = result + "|" + " ".join(moves)
        sig = hashlib.md5(sig_src.encode('utf-8')).hexdigest()
        if sig in seen:
            dupes += 1
            continue
        seen.add(sig)
        deduped.append(game)
    
    if dupes:
        print(f"  Games dedup: -{dupes:,}")
    return deduped




# ==============================================================================
# GAME FILTERING (HEADERS + LENGTH)
# ==============================================================================

def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _contains_any(text, keywords):
    text = (text or "").lower()
    for kw in keywords:
        if kw.lower() in text:
            return True
    return False


def _fullmoves_from_moves(moves):
    plies = len(moves) if moves else 0
    return (plies + 1) // 2


def _filter_games(games_data, config):
    cfg = config.get('data', {}).get('game_filters', {})
    
    filters_enabled = bool(cfg.get('enabled', False))
    
    if not filters_enabled:
        return games_data
    
    exclude_term_keywords = [s.lower() for s in (cfg.get('exclude_terminations', []) or [])]
    
    min_fullmove = int(cfg.get('min_fullmove', 0))
    min_fullmove_draw = int(cfg.get('min_fullmove_draw', min_fullmove))
    min_fullmove_resign = int(cfg.get('min_fullmove_resign', min_fullmove))
    max_elo_gap = int(cfg.get('max_elo_gap', 0) or 0)
    
    filtered = []
    stats = {
        'total': 0,
        'kept': 0,
        'termination': 0,
        'length': 0,
        'draw_short': 0,
        'resign_short': 0,
        'elo_gap': 0,
    }
    
    for game in games_data:
        stats['total'] += 1
        moves = game.get('moves') or []
        fullmoves = _fullmoves_from_moves(moves)
        
        result = game.get('result', '*')
        termination = (game.get('termination', '') or '').lower()
        
        # Termination filter
        if filters_enabled:
            if termination and exclude_term_keywords and _contains_any(termination, exclude_term_keywords):
                stats['termination'] += 1
                continue
        
        # Elo gap filter
        if filters_enabled and max_elo_gap > 0:
            white_elo = _safe_int(game.get('white_elo'))
            black_elo = _safe_int(game.get('black_elo'))
            if white_elo is not None and black_elo is not None:
                if abs(white_elo - black_elo) > max_elo_gap:
                    stats['elo_gap'] += 1
                    continue
        
        # Length filters
        if filters_enabled:
            if result == "1/2-1/2" and min_fullmove_draw > 0 and fullmoves < min_fullmove_draw:
                stats['draw_short'] += 1
                continue
            if _contains_any(termination, ["resign", "resignation"]) and min_fullmove_resign > 0 and fullmoves < min_fullmove_resign:
                stats['resign_short'] += 1
                continue
            if min_fullmove > 0 and fullmoves < min_fullmove:
                stats['length'] += 1
                continue
        
        filtered.append(game)
        stats['kept'] += 1
    
    if filters_enabled:
        removed_parts = []
        for key, label in (
            ('termination', 'term'),
            ('elo_gap', 'elo_gap'),
            ('draw_short', 'short_draw'),
            ('resign_short', 'short_resign'),
            ('length', 'min_len'),
        ):
            if stats[key]:
                removed_parts.append(f"{label}={stats[key]:,}")
        details = "; " + ", ".join(removed_parts) if removed_parts else ""
        print(f"  Games filter: {stats['total']:,} -> {stats['kept']:,}{details}")
    
    return filtered


def extract_games_from_pgn_parallel(pgn_path, max_games, phase1_threads, config=None):
    """
    PHASE 1: Extract games from PGN file
    """
    max_games = _normalize_max_games(max_games)

    all_games = extract_games_from_pgn_multiprocess(pgn_path, max_games, phase1_threads)
    
    if not all_games:
        return []
    
    # Remove duplicate games (same moves + result)
    all_games = _dedupe_games(all_games)
    
    # Apply filters (if enabled)
    if config:
        all_games = _filter_games(all_games, config)
    
    return all_games


# ==============================================================================
# PHASE 2: POSITION EXTRACTION (Multi-processing) - 🆕 POV + NO EMBEDDED HISTORY
# ==============================================================================

def extract_positions_from_game_worker(args):
    """
    PHASE 2 WORKER: Extract positions from a single game
    
    🔧 FIXED v4.3 CHANGES:
    - NO embedded history in binary format
    - Stores GameID, MoveIdx, and MoveTarget for training
    - MoveTarget is the LABEL for the network to predict
    - WDL-only mode: clean final +/-1.0/0.0 targets from side-to-move POV
    
    🔧 FIXED BINARY FORMAT:
    [Board (32B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)]
    """
    game_data, game_id, min_elo, max_moves_per_game = args
    
    import chess
    import struct
    
    # Import helpers locally
    from src.utils.data_helpers import (
        move_to_index,
        pack_position_data,
        compute_discounted_outcome
    )
    
    try:
        # Check Elo
        white_elo = game_data['white_elo']
        black_elo = game_data['black_elo']
        
        if white_elo == '?' or black_elo == '?':
            return []
        
        white_elo_int = int(white_elo)
        black_elo_int = int(black_elo)
        
        if white_elo_int < min_elo or black_elo_int < min_elo:
            return []
        
        # Check game length (drop if too long)
        moves = game_data['moves']
        if max_moves_per_game and max_moves_per_game > 0 and len(moves) > max_moves_per_game:
            return []
        
        # Replay game
        board = chess.Board()
        result = game_data['result']
        
        # Kept for API compatibility; WDL target is not temporally discounted.
        total_moves = len(moves)
        
        positions = []
        move_idx = 0
        
        for move_uci in moves:
            try:
                move = chess.Move.from_uci(move_uci)
                
                if move not in board.legal_moves:
                    break
                
                # 🔧 CRITICAL FIX: Calculate move_target BEFORE making the move
                # This is the LABEL the network should predict (0-4671)
                move_target = move_to_index(move, board)
                
                # WDL-only mode: no temporal discounting.
                outcome = compute_discounted_outcome(
                    move_idx=move_idx,
                    total_moves=total_moves,
                    result=result,
                    current_turn=board.turn,
                )
                
                # 🔧 Pack position using helper function (includes move_target)
                # Format: [Board (32B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)]
                position_data = pack_position_data(
                    board=board,
                    game_id=game_id,
                    move_idx=move_idx,
                    move_target=move_target,  # 🔧 NEW: The label to predict
                    outcome=outcome,
                )
                
                positions.append(position_data)
                
                # Make move and increment index
                board.push(move)
                move_idx += 1
                
            except Exception as e:
                break
        
        return positions
        
    except Exception as e:
        return []


def extract_positions_from_game_chunk_worker(args):
    """
    PHASE 2 CHUNK WORKER: extract positions from a group of games.

    This keeps game_id stable while avoiding one ProcessPool future per game.
    """
    indexed_games, min_elo, max_moves_per_game = args
    chunk_positions = []
    for game_id, game_data in indexed_games:
        positions = extract_positions_from_game_worker(
            (game_data, game_id, min_elo, max_moves_per_game)
        )
        if positions:
            chunk_positions.extend(positions)
    return chunk_positions


def extract_positions_parallel(games_data, config, phase2_workers):
    """
    PHASE 2: Extract positions from games using multi-processing
    """
    min_elo = config['data'].get('min_elo', 0)
    max_moves = config['data'].get('max_moves_per_game', 200)
    if phase2_workers <= 1:
        return extract_positions_sequential(games_data, config)
    
    chunk_size = _resolve_phase2_chunk_size(config, len(games_data), phase2_workers)
    indexed_games = list(enumerate(games_data))
    tasks = [
        indexed_games[start:start + chunk_size]
        for start in range(0, len(indexed_games), chunk_size)
    ]

    # 🔧 v4.3: game_id is uint32 — no modulo needed, supports up to ~4 billion games
    
    position_chunks = [None] * len(tasks)
    positions_done = 0
    games_done = 0

    with ProcessPoolExecutor(max_workers=phase2_workers) as executor:
        futures = {
            executor.submit(
                extract_positions_from_game_chunk_worker,
                (task, min_elo, max_moves),
            ): i
            for i, task in enumerate(tasks)
        }
        
        with tqdm(total=len(futures), desc=_phase_desc("Phase 2: position extraction")) as pbar:
            for future in as_completed(futures):
                task_idx = futures[future]
                positions = future.result()
                position_chunks[task_idx] = positions
                positions_done += len(positions)
                games_done += len(tasks[task_idx])
                pbar.update(1)
                pbar.set_postfix({'games': games_done, 'positions': positions_done})

    all_positions = []
    for chunk in position_chunks:
        if chunk:
            all_positions.extend(chunk)
    
    return all_positions


def extract_positions_sequential(games_data, config):
    """
    PHASE 2: Sequential position extraction (fallback)
    """
    min_elo = config['data'].get('min_elo', 0)
    max_moves = config['data'].get('max_moves_per_game', 200)
    
    all_positions = []
    
    for game_idx, game in enumerate(tqdm(games_data, desc=_phase_desc("Phase 2: position extraction"))):
        # 🔧 v4.3: game_id is uint32 — no modulo needed
        task = (game, game_idx, min_elo, max_moves)
        positions = extract_positions_from_game_worker(task)
        all_positions.extend(positions)
    
    return all_positions


# ==============================================================================
# PHASE 3 & 4: DISK WRITING & METADATA
# ==============================================================================

def write_positions_to_disk(positions, binary_file):
    """
    PHASE 3: Write positions to binary file
    """
    with open(binary_file, 'wb') as f:
        for pos_data in tqdm(positions, desc=_phase_desc("Phase 3: writing to disk"), unit="pos"):
            f.write(pos_data)


def get_dataset_metadata(binary_file, config):
    """
    Generate metadata for binary dataset
    
    🆕 v4.2: position_size calculated WITHOUT history (history is dynamic)
    """
    
    # Calculate position size WITHOUT history
    position_size = get_position_size(history_positions=0)
    
    # Count positions
    file_size = binary_file.stat().st_size
    total_positions = file_size // position_size
    
    metadata = {
        'binary_file': str(binary_file),
        'total_positions': total_positions,
        'position_size': position_size,
        'input_planes': 16,  # 🔧 v4.5 FIX: 16 planes (12 pieces + 4 metadata)
        'created_date': datetime.now().isoformat()
    }
    
    return metadata


# ==============================================================================
# MAIN PROCESSING FUNCTION
# ==============================================================================

def process_pgn_files(pgn_files, config):
    """
    Process PGN files with smart tracking and reuse
    
    Returns:
        metadata dict for combined dataset
    """
    data_dir = Path(config['paths']['data_dir'])
    preprocessing_dir = data_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracker
    tracker = DatasetTracker(preprocessing_dir)
    
    # Check for existing datasets
    existing_binaries = []
    existing_metadatas = []
    new_pgn_files = []
    dataset_summary_rows = []
    
    for pgn_file in pgn_files:
        binary_file, metadata_file = tracker.get_processed_dataset(pgn_file, config)
        
        if binary_file and metadata_file:
            existing_binaries.append(binary_file)
            existing_metadatas.append(metadata_file)
            positions_text = "?"
            try:
                with open(metadata_file, 'rb') as f:
                    cached_metadata = pickle.load(f)
                positions_text = f"{int(cached_metadata.get('total_positions', 0)):,}"
            except Exception:
                pass
            dataset_summary_rows.append((Path(pgn_file).name, positions_text, "cache"))
        else:
            new_pgn_files.append(pgn_file)
            dataset_summary_rows.append((Path(pgn_file).name, "pending", "build"))
    
    total_pgn_files = len(pgn_files)
    loaded_count = len(existing_binaries)
    remaining_count = len(new_pgn_files)
    print(
        f"  Preprocessed datasets: cache {loaded_count:,}/{total_pgn_files:,}, "
        f"to build {remaining_count:,}/{total_pgn_files:,}",
        flush=True,
    )
    _print_preprocessing_dataset_table(dataset_summary_rows)

    # Process new files if any
    new_binaries = []
    new_metadatas = []
    
    if new_pgn_files:
        phase1_workers = _resolve_workers(config['data'].get('phase1_threads', 1), "phase1_threads")
        phase2_workers = _resolve_workers(config['data'].get('phase2_threads', 1), "phase2_threads")
        selection = _get_game_selection_settings(config)
        max_games = selection['max_games']
        
        total_new = len(new_pgn_files)
        for dataset_idx, pgn_file in enumerate(new_pgn_files, start=1):
            dataset_name = Path(pgn_file).name
            loaded_now = loaded_count + dataset_idx - 1
            print(
                f"\nPreprocessing {dataset_name} "
                f"({dataset_idx:,}/{total_new:,}; cache {loaded_now:,}/{total_pgn_files:,})",
                flush=True,
            )
            print("=" * 70, flush=True)
            
            # Phase 1: Extract games
            games_data = extract_games_from_pgn_parallel(
                pgn_file,
                max_games,
                phase1_workers,
                config=config,
            )
            
            if not games_data:
                print("  ⚠️ No games found!")
                continue
            
            # Phase 2: Extract positions
            positions = extract_positions_parallel(
                games_data,
                config,
                phase2_workers,
            )
            
            if not positions:
                print("  ⚠️ No positions extracted!")
                continue
            
            # Phase 3: Write to disk
            binary_file = preprocessing_dir / f"{Path(pgn_file).stem}_positions.bin"
            write_positions_to_disk(positions, binary_file)
            
            # Phase 4: Create metadata
            metadata = get_dataset_metadata(binary_file, config)
            metadata_file = preprocessing_dir / f"{Path(pgn_file).stem}_meta.pkl"
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Register dataset
            tracker.register_dataset(pgn_file, config, binary_file, metadata_file, metadata['total_positions'])
            
            new_binaries.append(binary_file)
            new_metadatas.append(metadata_file)
            
            print(
                f"  Done: {Path(pgn_file).name} "
                f"({metadata['total_positions']:,} positions, "
                f"{binary_file.stat().st_size / (1024**2):.1f} MB)",
                flush=True,
            )
    
    # Merge all datasets (existing + new)
    all_binaries = existing_binaries + new_binaries
    all_metadatas = existing_metadatas + new_metadatas
    
    if not all_binaries:
        raise ValueError("No datasets to process!")
    
    if len(all_binaries) == 1:
        # Single dataset, no merge needed
        print(f"\n{'='*70}")
        print(f"✅ Using single dataset (no merge needed)")
        print(f"{'='*70}\n")
        
        with open(all_metadatas[0], 'rb') as f:
            return pickle.load(f)
    else:
        # Merge multiple datasets
        final_binary = preprocessing_dir / "combined_dataset.bin"
        final_metadata = preprocessing_dir / "combined_dataset_meta.pkl"
        
        combined_meta = merge_binary_datasets(
            all_binaries,
            all_metadatas,
            final_binary,
            final_metadata
        )
        
        return combined_meta
