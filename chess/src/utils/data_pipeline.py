"""
Chess data processing and dataset management (pipeline).
Split from src.data to keep preprocessing separate from dataset/dataloader code.
"""

import chess
import chess.pgn
import gc
import hashlib
import json
import pickle
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.utils.data_helpers import get_position_size

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
        """Compute hash of PGN file (first 10MB for speed)"""
        hasher = hashlib.md5()
        
        with open(pgn_path, 'rb') as f:
            # Read first 10MB (enough to detect file changes)
            chunk = f.read(10 * 1024 * 1024)
            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _compute_config_hash(self, config):
        """
        Compute hash of processing configuration
        Only includes parameters that affect binary output
        🔧 v4.4 FIX: Added use_wdl_value (affects temporal discounting)
        """
        relevant_config = {
            'min_elo': config['data'].get('min_elo', 0),
            'max_games': config['data'].get('max_games', float('inf')),
            'max_moves_per_game': config['data'].get('max_moves_per_game', 200),
            'use_multitask_learning': config['model'].get('use_multitask_learning', False),
            'use_wdl_value': config['model'].get('use_wdl_value', True),  # 🔧 CRITICAL FIX
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
                print(f"  ♻️ Found preprocessed dataset: {binary_path.name}")
                print(f"     Processed on: {entry['processed_date']}")
                print(f"     Positions: {entry['total_positions']:,}")
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
                'max_games': config['data'].get('max_games', float('inf')),
                'use_mtl': config['model'].get('use_multitask_learning', False),
            }
        }
        
        self._save_tracking()
        print(f"  ✅ Dataset registered in tracking system")


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
    use_mtl = first_meta['use_mtl']
    
    for meta in all_metadata[1:]:
        if (meta['position_size'] != position_size or 
            meta['use_mtl'] != use_mtl):
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
        'use_mtl': use_mtl,
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
        
        moves = []
        for move in game.mainline_moves():
            moves.append(move.uci())
        
        return {
            'white_elo': white_elo,
            'black_elo': black_elo,
            'result': result,
            'moves': moves
        }
    except:
        return None


def parse_games_batch_worker(args):
    """
    PHASE 1 WORKER: Parse a batch of games (runs in separate process)
    """
    pgn_path, start_game, num_games = args
    
    import chess.pgn
    
    games_data = []
    
    try:
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip to start position
            for _ in range(start_game):
                game = chess.pgn.read_game(f)
                if game is None:
                    return games_data
            
            # Read our batch
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
    
    print(f"  Using {phase1_workers} processes for parallel parsing...")
    
    # Calculate games per worker
    games_per_worker = (max_games + phase1_workers - 1) // phase1_workers
    
    # Create tasks
    tasks = []
    for i in range(phase1_workers):
        start_game = i * games_per_worker
        num_games = min(games_per_worker, max_games - start_game)
        
        if num_games <= 0:
            break
        
        tasks.append((pgn_path, start_game, num_games))
    
    # Process in parallel
    all_games = []
    
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(parse_games_batch_worker, task): i for i, task in enumerate(tasks)}
        
        with tqdm(total=len(futures), desc="  Phase 1 workers") as pbar:
            for future in as_completed(futures):
                games_chunk = future.result()
                all_games.extend(games_chunk)
                pbar.update(1)
                pbar.set_postfix({'games': len(all_games)})
    
    return all_games[:max_games]


def extract_games_sequential(pgn_path, max_games):
    """
    PHASE 1: Sequential extraction (fallback)
    """
    games_data = []
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        with tqdm(total=max_games, desc="  Extracting games") as pbar:
            game_count = 0
            while game_count < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                game_data = extract_game_data(game)
                if game_data:
                    games_data.append(game_data)
                    game_count += 1
                    pbar.update(1)
    
    return games_data


def sort_games_by_elo(games_data):
    """Sort games by average Elo (descending)"""
    def get_avg_elo(game):
        try:
            white_elo = game['white_elo']
            black_elo = game['black_elo']
            
            if white_elo == '?' or black_elo == '?':
                return 0
            
            return (int(white_elo) + int(black_elo)) / 2
        except:
            return 0
    
    return sorted(games_data, key=get_avg_elo, reverse=True)


def extract_games_from_pgn_parallel(pgn_path, max_games, phase1_threads, sort_by_elo=True):
    """
    PHASE 1: Extract games from PGN file
    """
    games_to_extract = max_games * 2 if sort_by_elo else max_games
    
    if sort_by_elo:
        print(f"  📊 Extracting {games_to_extract:,} games (will sort and select top {max_games:,} by Elo)...")
    
    all_games = extract_games_from_pgn_multiprocess(pgn_path, games_to_extract, phase1_threads)
    
    if not all_games:
        return []
    
    # Sort by Elo and take top games
    if sort_by_elo:
        print(f"  🔝 Sorting {len(all_games):,} games by average Elo...")
        all_games = sort_games_by_elo(all_games)
        all_games = all_games[:max_games]
        
        # Print Elo stats
        if all_games:
            elos = []
            for game in all_games:
                try:
                    if game['white_elo'] != '?' and game['black_elo'] != '?':
                        avg_elo = (int(game['white_elo']) + int(game['black_elo'])) / 2
                        elos.append(avg_elo)
                except:
                    pass
            
            if elos:
                print(f"  ✅ Selected top {len(all_games):,} games")
                print(f"     Average Elo range: {min(elos):.0f} - {max(elos):.0f}")
                print(f"     Mean Elo: {sum(elos)/len(elos):.0f}")
    
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
    - 🆕 TEMPORAL DISCOUNTING: Outcome with sqrt(progress) discounting (ONLY if use_wdl=False)
    
    🔧 FIXED BINARY FORMAT:
    [Board (32B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [MTL (12B if enabled)]
    """
    game_data, game_id, min_elo, max_moves_per_game, use_mtl, use_wdl = args
    
    import chess
    import struct
    
    # Import helpers locally
    from src.utils.data_helpers import (
        board_to_compact,
        move_to_index,
        compute_material_balance,
        is_in_check,
        will_win,
        pack_position_data,
        compute_discounted_outcome  # 🆕 v4.3: Temporal discounting
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
        
        # Check game length
        if len(game_data['moves']) > max_moves_per_game:
            return []
        
        # Replay game
        board = chess.Board()
        result = game_data['result']
        
        # 🆕 v4.3: Calculate total moves for temporal discounting
        total_moves = len(game_data['moves'])
        
        positions = []
        move_idx = 0
        
        for move_uci in game_data['moves']:
            try:
                move = chess.Move.from_uci(move_uci)
                
                if move not in board.legal_moves:
                    break
                
                # 🔧 CRITICAL FIX: Calculate move_target BEFORE making the move
                # This is the LABEL the network should predict (0-4095)
                move_target = move_to_index(move, board)
                
                # 🆕 v4.3: TEMPORAL DISCOUNTING (conditional on use_wdl)
                # WDL: No discounting (returns ±1.0, WDL models uncertainty via probabilities)
                # MSE: Discounting enabled (early game → lower values)
                # This fixes the critical bug where discounted values (0.3) were classified as Draw!
                outcome = compute_discounted_outcome(
                    move_idx=move_idx,
                    total_moves=total_moves,
                    result=result,
                    current_turn=board.turn,
                    use_wdl=use_wdl  # 🔧 CRITICAL: Pass WDL flag
                )
                
                # MTL labels
                mtl_labels = None
                if use_mtl:
                    mtl_labels = {
                        'win': will_win(board, result),
                        'material': compute_material_balance(board),
                        'check': is_in_check(board)
                    }
                
                # 🔧 Pack position using helper function (includes move_target)
                # Format: [Board (32B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [MTL (12B if enabled)]
                position_data = pack_position_data(
                    board=board,
                    game_id=game_id,
                    move_idx=move_idx,
                    move_target=move_target,  # 🔧 NEW: The label to predict
                    outcome=outcome,  # 🆕 v4.3: Conditional discounting based on use_wdl
                    mtl_labels=mtl_labels,
                    use_wdl=use_wdl  # API consistency
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


def extract_positions_parallel(games_data, config, phase2_workers):
    """
    PHASE 2: Extract positions from games using multi-processing
    """
    min_elo = config['data'].get('min_elo', 0)
    max_moves = config['data'].get('max_moves_per_game', 200)
    use_mtl = config['model'].get('use_multitask_learning', False)
    use_wdl = config['model'].get('use_wdl_value', True)  # 🆕 Get WDL flag
    
    if phase2_workers <= 1:
        return extract_positions_sequential(games_data, config)
    
    print(f"  Using {phase2_workers} processes for parallel position extraction...")
    
    # Prepare tasks with unique game_id for each game
    # 🔧 v4.3: game_id is uint32 — no modulo needed, supports up to ~4 billion games
    tasks = [(game, game_idx, min_elo, max_moves, use_mtl, use_wdl) 
             for game_idx, game in enumerate(games_data)]
    
    # Process in parallel
    all_positions = []
    
    with ProcessPoolExecutor(max_workers=phase2_workers) as executor:
        futures = {executor.submit(extract_positions_from_game_worker, task): i 
                   for i, task in enumerate(tasks)}
        
        with tqdm(total=len(futures), desc="  Phase 2 workers") as pbar:
            for future in as_completed(futures):
                positions = future.result()
                all_positions.extend(positions)
                pbar.update(1)
                pbar.set_postfix({'positions': len(all_positions)})
    
    return all_positions


def extract_positions_sequential(games_data, config):
    """
    PHASE 2: Sequential position extraction (fallback)
    """
    min_elo = config['data'].get('min_elo', 0)
    max_moves = config['data'].get('max_moves_per_game', 200)
    use_mtl = config['model'].get('use_multitask_learning', False)
    use_wdl = config['model'].get('use_wdl_value', True)  # 🆕 Get WDL flag
    
    all_positions = []
    
    for game_idx, game in enumerate(tqdm(games_data, desc="  Extracting positions")):
        # 🔧 v4.3: game_id is uint32 — no modulo needed
        task = (game, game_idx, min_elo, max_moves, use_mtl, use_wdl)
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
    print(f"\n  💾 Writing {len(positions):,} positions to disk...")
    
    with open(binary_file, 'wb') as f:
        for pos_data in tqdm(positions, desc="  Writing", unit="pos"):
            f.write(pos_data)
    
    size_mb = binary_file.stat().st_size / (1024**2)
    print(f"  ✅ Written: {size_mb:.1f} MB")


def get_dataset_metadata(binary_file, config):
    """
    Generate metadata for binary dataset
    
    🆕 v4.2: position_size calculated WITHOUT history (history is dynamic)
    """
    use_mtl = config['model'].get('use_multitask_learning', False)
    
    # Calculate position size WITHOUT history
    position_size = get_position_size(use_mtl=use_mtl, history_positions=0)
    
    # Count positions
    file_size = binary_file.stat().st_size
    total_positions = file_size // position_size
    
    metadata = {
        'binary_file': str(binary_file),
        'total_positions': total_positions,
        'position_size': position_size,
        'use_mtl': use_mtl,
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
    
    print(f"\n{'='*70}")
    print(f"📚 Checking for preprocessed datasets...")
    print(f"{'='*70}")
    
    for pgn_file in pgn_files:
        binary_file, metadata_file = tracker.get_processed_dataset(pgn_file, config)
        
        if binary_file and metadata_file:
            existing_binaries.append(binary_file)
            existing_metadatas.append(metadata_file)
        else:
            new_pgn_files.append(pgn_file)
            print(f"  🆕 Will process: {Path(pgn_file).name}")
    
    # Process new files if any
    new_binaries = []
    new_metadatas = []
    
    if new_pgn_files:
        print(f"\n{'='*70}")
        print(f"🔨 Processing {len(new_pgn_files)} new PGN files...")
        print(f"{'='*70}")
        
        phase1_workers = config['data'].get('phase1_threads', 1)
        phase2_workers = config['data'].get('phase2_threads', 1)
        max_games = config['data'].get('max_games', 100000)
        sort_by_elo = config['data'].get('sort_by_avg_elo', True)
        
        for pgn_file in new_pgn_files:
            print(f"\n📄 Processing: {Path(pgn_file).name}")
            print(f"{'='*70}")
            
            # Phase 1: Extract games
            print("🔹 PHASE 1: PGN Parsing...")
            games_data = extract_games_from_pgn_parallel(
                pgn_file, max_games, phase1_workers, sort_by_elo
            )
            
            if not games_data:
                print("  ⚠️ No games found!")
                continue
            
            print(f"  ✅ Extracted {len(games_data):,} games")
            
            # Phase 2: Extract positions
            print(f"\n🔹 PHASE 2: Position Extraction...")
            positions = extract_positions_parallel(games_data, config, phase2_workers)
            
            if not positions:
                print("  ⚠️ No positions extracted!")
                continue
            
            print(f"  ✅ Extracted {len(positions):,} positions")
            
            # Phase 3: Write to disk
            print(f"\n🔹 PHASE 3: Writing to disk...")
            binary_file = preprocessing_dir / f"{Path(pgn_file).stem}_positions.bin"
            write_positions_to_disk(positions, binary_file)
            
            # Phase 4: Create metadata
            print(f"\n🔹 PHASE 4: Creating metadata...")
            metadata = get_dataset_metadata(binary_file, config)
            metadata_file = preprocessing_dir / f"{Path(pgn_file).stem}_meta.pkl"
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Register dataset
            tracker.register_dataset(pgn_file, config, binary_file, metadata_file, metadata['total_positions'])
            
            new_binaries.append(binary_file)
            new_metadatas.append(metadata_file)
            
            print(f"\n✅ Completed: {Path(pgn_file).name}")
            print(f"   Positions: {metadata['total_positions']:,}")
            print(f"   Size: {binary_file.stat().st_size / (1024**2):.1f} MB")
    
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

