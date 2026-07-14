"""
Chess data processing and dataset management (pipeline).
Split from src.data to keep preprocessing separate from dataset/dataloader code.
"""

import chess
import chess.pgn
import io
import os
import hashlib
import json
import pickle
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.data_helpers import ACTION_SIZE, get_position_size
from src.utils.config import normalize_data_config


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
            # v2 stores ActorElo after Outcome so old 50-byte caches cannot be reused.
            'action_encoding': 'az_classic_8x8x73_v2_actor_elo',
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
    # by the cumulative game count of all previous files.  Because positions within one
    # source file are ordered by game (all moves of game 0, then game 1, …) and the history
    # walk relies solely on GameID equality to detect game boundaries, the remapped IDs
    # preserve that property while eliminating cross-file collisions.
    print(f"\n  🔨 Writing merged binary file (with GameID remapping)...")
    chunk_bytes = 256 * 1024 * 1024
    records_per_chunk = max(1, chunk_bytes // int(position_size))
    record_dtype = np.dtype({
        'names': ['game_id'],
        'formats': ['<u4'],
        'offsets': [38],
        'itemsize': int(position_size),
    })

    # Rewrite GameIDs in large vectorized chunks. This keeps the same binary layout, but avoids
    # millions of tiny struct.unpack/pack calls while merging large PGN months.
    game_id_offset = 0
    
    with open(output_binary, 'wb') as outfile:
        for i, (binary_file, meta) in enumerate(zip(binary_files, all_metadata), 1):
            print(f"     Merging file {i}/{len(binary_files)}: {Path(binary_file).name} "
                  f"(GameID offset: {game_id_offset})")
            
            file_positions = int(meta['total_positions'])
            max_game_id_in_file = 0
            
            with open(binary_file, 'rb') as infile:
                remaining = file_positions
                while remaining > 0:
                    requested_records = min(records_per_chunk, remaining)
                    raw = infile.read(requested_records * int(position_size))
                    if not raw:
                        break

                    actual_records = len(raw) // int(position_size)
                    if actual_records <= 0:
                        break
                    if len(raw) != actual_records * int(position_size):
                        raw = raw[:actual_records * int(position_size)]

                    chunk = bytearray(raw)
                    records = np.frombuffer(chunk, dtype=record_dtype, count=actual_records)
                    game_ids = records['game_id']
                    if len(game_ids):
                        max_game_id_in_file = max(max_game_id_in_file, int(game_ids.max()))
                        if game_id_offset:
                            remapped = game_ids.astype(np.uint64, copy=False) + np.uint64(game_id_offset)
                            if int(remapped.max()) > np.iinfo(np.uint32).max:
                                raise ValueError("Merged GameID exceeds uint32 range.")
                            game_ids[:] = remapped.astype(np.uint32, copy=False)

                    outfile.write(chunk)
                    remaining -= actual_records
            
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

class _MainlineVisitor(chess.pgn.BaseVisitor):
    """Lean PGN reader: retain headers and mainline moves, skip tree allocation."""

    def __init__(self):
        self.headers = chess.pgn.Headers()
        self.moves = []

    def begin_headers(self):
        return self.headers

    def visit_header(self, tagname, tagvalue):
        self.headers[tagname] = tagvalue

    def visit_move(self, board, move):
        self.moves.append(move)

    def begin_variation(self):
        return chess.pgn.SKIP

    def handle_error(self, error):
        # Match GameBuilder's forgiving behavior: invalid tails are skipped and
        # never enter the IL binary.
        return None

    def result(self):
        return self.headers, self.moves


def _read_mainline_game(pgn_file):
    parsed = chess.pgn.read_game(pgn_file, Visitor=_MainlineVisitor)
    if parsed is None:
        return None, None
    return parsed


def _game_data_from_headers_moves(headers, move_objects):
    return {
        'white_elo': headers.get('WhiteElo', '?'),
        'black_elo': headers.get('BlackElo', '?'),
        'result': headers.get('Result', '*'),
        'termination': headers.get('Termination', ''),
        'moves': [move.uci() for move in move_objects],
    }


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


# ==============================================================================
# PHASE 2: POSITION EXTRACTION (Multi-processing) - 🆕 POV + NO EMBEDDED HISTORY
# ==============================================================================

def _game_filter_reason(game_data, filters_cfg):
    """Return ``None`` when a parsed game passes the configured PGN filters."""
    filters_cfg = filters_cfg or {}
    if not bool(filters_cfg.get('enabled', False)):
        return None

    termination = (game_data.get('termination', '') or '').lower()
    excluded = [str(value).lower() for value in (filters_cfg.get('exclude_terminations', []) or [])]
    if termination and excluded and _contains_any(termination, excluded):
        return 'termination'

    max_elo_gap = int(filters_cfg.get('max_elo_gap', 0) or 0)
    if max_elo_gap > 0:
        white_elo = _safe_int(game_data.get('white_elo'))
        black_elo = _safe_int(game_data.get('black_elo'))
        if white_elo is not None and black_elo is not None and abs(white_elo - black_elo) > max_elo_gap:
            return 'elo_gap'

    fullmoves = _fullmoves_from_moves(game_data.get('moves') or [])
    result = game_data.get('result', '*')
    min_fullmove = int(filters_cfg.get('min_fullmove', 0) or 0)
    min_draw = int(filters_cfg.get('min_fullmove_draw', min_fullmove) or 0)
    min_resign = int(filters_cfg.get('min_fullmove_resign', min_fullmove) or 0)
    if result == '1/2-1/2' and min_draw > 0 and fullmoves < min_draw:
        return 'draw_short'
    if _contains_any(termination, ['resign', 'resignation']) and min_resign > 0 and fullmoves < min_resign:
        return 'resign_short'
    if min_fullmove > 0 and fullmoves < min_fullmove:
        return 'length'
    return None


def _stream_pgn_positions_shard_worker(args):
    """Parse, filter and materialize one PGN shard directly into a temp binary part."""
    (
        pgn_path,
        start_offset,
        game_count,
        part_path,
        min_elo,
        max_moves_per_game,
        filters_cfg,
    ) = args
    import chess.pgn

    games = []
    local_game_id = 0
    positions_written = 0
    parsed_games = 0
    Path(part_path).parent.mkdir(parents=True, exist_ok=True)

    with open(part_path, 'wb') as part_file:
        with open(pgn_path, 'rb') as raw_file:
            raw_file.seek(int(start_offset))
            with io.TextIOWrapper(raw_file, encoding='utf-8', errors='ignore', newline='') as pgn_file:
                for _ in range(int(game_count)):
                    headers, move_objects = _read_mainline_game(pgn_file)
                    if headers is None:
                        break
                    game_data = _game_data_from_headers_moves(headers, move_objects)
                    parsed_games += 1
                    signature = hashlib.md5(
                        (str(game_data.get('result', '')) + '|' + ' '.join(game_data.get('moves') or [])).encode('utf-8')
                    ).digest()
                    filter_reason = _game_filter_reason(game_data, filters_cfg)
                    start = int(part_file.tell())
                    if filter_reason is None:
                        positions = extract_positions_from_game_worker(
                            (game_data, local_game_id, min_elo, max_moves_per_game, move_objects)
                        )
                        local_game_id += 1
                        if positions:
                            raw_positions = b''.join(positions)
                            part_file.write(raw_positions)
                            positions_written += len(positions)
                    end = int(part_file.tell())
                    games.append((signature, start, end, filter_reason))

    return {
        'part_path': str(part_path),
        'games': games,
        'parsed_games': int(parsed_games),
        'positions': int(positions_written),
    }


def _rewrite_game_id(raw_records, position_size, game_id):
    """Set one compact GameID in a contiguous game's binary records."""
    if not raw_records:
        return raw_records
    if len(raw_records) % int(position_size):
        raise ValueError('PGN shard contains a partial position record.')
    payload = bytearray(raw_records)
    record_dtype = np.dtype({
        'names': ['game_id'],
        'formats': ['<u4'],
        'offsets': [38],
        'itemsize': int(position_size),
    })
    records = np.frombuffer(payload, dtype=record_dtype)
    records['game_id'] = np.uint32(game_id)
    return payload


def _stream_pgn_to_binary(pgn_path, binary_file, config, workers):
    """Build one binary dataset in a single parse-and-extract multiprocessing pass."""
    data_cfg = config['data']
    max_games = _normalize_max_games(data_cfg.get('max_games', 'max'))
    offsets = _scan_game_start_offsets(pgn_path, max_offsets=max_games)
    if not offsets:
        return 0

    workers = max(1, min(int(workers), len(offsets)))
    task_count = min(len(offsets), max(workers, workers * 4))
    games_per_task = max(1, (len(offsets) + task_count - 1) // task_count)
    part_dir = Path(binary_file).parent / f'.{Path(binary_file).stem}_parts'
    temp_binary = Path(binary_file).with_suffix(Path(binary_file).suffix + '.building')
    filters_cfg = dict(data_cfg.get('game_filters', {}) or {})
    tasks = []
    for task_id, start_index in enumerate(range(0, len(offsets), games_per_task)):
        count = min(games_per_task, len(offsets) - start_index)
        tasks.append((
            str(pgn_path), int(offsets[start_index]), int(count),
            str(part_dir / f'shard_{task_id:04d}.bin'),
            int(data_cfg.get('min_elo', 0) or 0),
            int(data_cfg.get('max_moves_per_game', 0) or 0), filters_cfg,
        ))

    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    results = [None] * len(tasks)
    try:
        parsed_games = 0
        materialized_positions = 0
        if workers == 1:
            with tqdm(total=len(tasks), desc=_phase_desc('Building positions'), unit=' shard') as progress:
                for task_id, task in enumerate(tasks):
                    result = _stream_pgn_positions_shard_worker(task)
                    results[task_id] = result
                    parsed_games += int(result['parsed_games'])
                    materialized_positions += int(result['positions'])
                    progress.update(1)
                    progress.set_postfix_str(f"{parsed_games:,} games | {materialized_positions:,} positions")
        else:
            with tqdm(total=len(tasks), desc=_phase_desc('Building positions'), unit=' shard') as progress:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_stream_pgn_positions_shard_worker, task): task_id
                        for task_id, task in enumerate(tasks)
                    }
                    for future in as_completed(futures):
                        task_id = futures[future]
                        result = future.result()
                        results[task_id] = result
                        parsed_games += int(result['parsed_games'])
                        materialized_positions += int(result['positions'])
                        progress.update(1)
                        progress.set_postfix_str(f"{parsed_games:,} games | {materialized_positions:,} positions")

        seen_signatures = set()
        filter_counts = {'termination': 0, 'elo_gap': 0, 'draw_short': 0, 'resign_short': 0, 'length': 0}
        duplicate_count = 0
        next_game_id = 0
        written_positions = 0
        write_buffer = bytearray()
        flush_threshold = 8 * 1024 * 1024
        position_size = get_position_size(history_positions=0)
        with open(temp_binary, 'wb') as output_file:
            for result in results:
                with open(result['part_path'], 'rb') as part_file:
                    for signature, start, end, filter_reason in result['games']:
                        if signature in seen_signatures:
                            duplicate_count += 1
                            continue
                        seen_signatures.add(signature)
                        if filter_reason is not None:
                            filter_counts[filter_reason] += 1
                            continue
                        part_file.seek(int(start))
                        raw_records = part_file.read(int(end) - int(start))
                        if raw_records:
                            remapped = _rewrite_game_id(raw_records, position_size, next_game_id)
                            write_buffer.extend(remapped)
                            written_positions += len(remapped) // position_size
                            if len(write_buffer) >= flush_threshold:
                                output_file.write(write_buffer)
                                write_buffer.clear()
                        next_game_id += 1
            if write_buffer:
                output_file.write(write_buffer)

        if filters_cfg.get('enabled', False) or duplicate_count:
            kept_games = len(seen_signatures) - sum(filter_counts.values())
            labels = {
                'termination': 'termination',
                'elo_gap': 'elo gap',
                'draw_short': 'short draw',
                'resign_short': 'short resign',
                'length': 'short game',
            }
            removed = []
            if duplicate_count:
                removed.append(f'{duplicate_count:,} duplicates')
            removed.extend(f'{count:,} {labels[name]}' for name, count in filter_counts.items() if count)
            suffix = f"  |  removed: {', '.join(removed)}" if removed else ''
            print(f"  Games: {len(seen_signatures):,} -> {kept_games:,} kept{suffix}")
        os.replace(temp_binary, binary_file)
        return int(written_positions)
    except Exception:
        if temp_binary.exists():
            temp_binary.unlink()
        raise
    finally:
        shutil.rmtree(part_dir, ignore_errors=True)


def extract_positions_from_game_worker(args):
    """
    PHASE 2 WORKER: Extract positions from a single game
    
    🔧 FIXED v4.3 CHANGES:
    - NO embedded history in binary format
    - Stores GameID, MoveIdx, and MoveTarget for training
    - MoveTarget is the LABEL for the network to predict
    - WDL-only mode: clean final +/-1.0/0.0 targets from side-to-move POV
    
    🔧 FIXED BINARY FORMAT:
    [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [ActorElo (2B)]
    """
    game_data, game_id, min_elo, max_moves_per_game, *optional_moves = args
    
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
        moves = optional_moves[0] if optional_moves else game_data['moves']
        if max_moves_per_game and max_moves_per_game > 0 and len(moves) > max_moves_per_game:
            return []
        
        # Replay game
        board = chess.Board()
        result = game_data['result']
        
        positions = []
        move_idx = 0
        
        moves_are_objects = bool(moves) and isinstance(moves[0], chess.Move)
        for raw_move in moves:
            try:
                move = raw_move if moves_are_objects else chess.Move.from_uci(raw_move)
                # chess.pgn already validates mainline moves while parsing SAN.
                # Re-generating every legal move here costs ~15% of preprocessing
                # time and does not add safety for these parsed mainlines.
                
                # 🔧 CRITICAL FIX: Calculate move_target BEFORE making the move
                # This is the LABEL the network should predict (0-4671)
                move_target = move_to_index(move, board)
                
                # WDL-only mode: no temporal discounting.
                outcome = compute_discounted_outcome(
                    result=result,
                    current_turn=board.turn,
                )
                
                # 🔧 Pack position using helper function (includes move_target)
                # Format: [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [ActorElo (2B)]
                actor_elo = white_elo_int if board.turn == chess.WHITE else black_elo_int
                position_data = pack_position_data(
                    board=board,
                    game_id=game_id,
                    move_idx=move_idx,
                    move_target=move_target,  # 🔧 NEW: The label to predict
                    outcome=outcome,
                    actor_elo=actor_elo,
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
    config = normalize_data_config(config)
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
        preprocess_workers = _resolve_workers(
            config['data'].get('preprocess_threads', config['data'].get('phase1_threads', 'all')),
            "preprocess_threads",
        )
        
        total_new = len(new_pgn_files)
        for dataset_idx, pgn_file in enumerate(new_pgn_files, start=1):
            dataset_name = Path(pgn_file).name
            loaded_now = loaded_count + dataset_idx - 1
            print(
                f"\n[{dataset_idx:02d}/{total_new:02d}] {dataset_name}  "
                f"|  cache {loaded_now:,}/{total_pgn_files:,}",
                flush=True,
            )
            
            binary_file = preprocessing_dir / f"{Path(pgn_file).stem}_positions.bin"
            position_count = _stream_pgn_to_binary(
                pgn_file,
                binary_file,
                config,
                preprocess_workers,
            )
            if position_count <= 0:
                print("  ⚠️ No positions extracted!")
                if binary_file.exists():
                    binary_file.unlink()
                continue
            
            # The final binary is atomically ready; now persist its metadata.
            metadata = get_dataset_metadata(binary_file, config)
            metadata_file = preprocessing_dir / f"{Path(pgn_file).stem}_meta.pkl"
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Register dataset
            tracker.register_dataset(pgn_file, config, binary_file, metadata_file, metadata['total_positions'])
            
            new_binaries.append(binary_file)
            new_metadatas.append(metadata_file)
            
            print(
                f"  Done | {metadata['total_positions']:,} positions | "
                f"{binary_file.stat().st_size / (1024**2):.1f} MB",
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
