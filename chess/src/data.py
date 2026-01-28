"""
Chess data processing and dataset management - OPTIMIZED FOR LOW RAM
🆕 v4.1: Smart dataset management with tracking and auto-cleanup
- ♻️ Auto-cleanup: Removes intermediate files after position creation
- 📝 Dataset tracking: Tracks processed datasets to avoid reprocessing
- 🔗 Smart merging: Reuses compatible preprocessed datasets
- 📁 New location: data/preprocessing instead of cache
"""

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import gc
import struct
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
from datetime import datetime


# Import helper functions from utils
from .utils.data_helpers import (
    board_to_tensor, 
    board_to_compact, 
    compact_to_tensor,
    move_to_index, 
    index_to_move,
    compute_material_balance,
    is_in_check,
    will_win
)


def extract_auxiliary_labels(board, game_result):
    """
    Extract all auxiliary task labels for Multi-Task Learning
    
    Args:
        board: chess.Board at current position
        game_result: Game result string
    
    Returns:
        dict: {
            'win': float,        # Will current player win? (0.0 or 1.0)
            'material': float,   # Material balance (-1.0 to 1.0)
            'check': float       # Is in check? (0.0 or 1.0)
        }
    """
    return {
        'win': will_win(board, game_result),
        'material': compute_material_balance(board),
        'check': is_in_check(board)
    }


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
        """
        relevant_config = {
            'min_elo': config['data'].get('min_elo', 0),
            'max_games': config['data'].get('max_games', float('inf')),
            'max_moves_per_game': config['data'].get('max_moves_per_game', 200),
            'history_positions': config['model'].get('history_positions', 0),
            'use_multitask_learning': config['model'].get('use_multitask_learning', False),
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
                'history_positions': config['model'].get('history_positions', 0),
                'use_mtl': config['model'].get('use_multitask_learning', False),
            }
        }
        
        self._save_tracking()
        print(f"  ✅ Dataset registered in tracking system")
    
    def list_processed_datasets(self):
        """List all processed datasets"""
        if not self.tracking_data:
            print("  No processed datasets found")
            return
        
        print(f"\n{'='*80}")
        print(f"📚 Processed Datasets ({len(self.tracking_data)} total)")
        print(f"{'='*80}")
        
        for i, (key, entry) in enumerate(self.tracking_data.items(), 1):
            binary_exists = Path(entry['binary_file']).exists()
            status = "✅" if binary_exists else "❌"
            
            print(f"\n{i}. {status} {entry['pgn_file']}")
            print(f"   Binary: {Path(entry['binary_file']).name}")
            print(f"   Positions: {entry['total_positions']:,}")
            print(f"   Date: {entry['processed_date'][:10]}")
            print(f"   Config: min_elo={entry['config']['min_elo']}, "
                  f"history={entry['config']['history_positions']}, "
                  f"mtl={entry['config']['use_mtl']}")
        
        print(f"\n{'='*80}\n")


# ==============================================================================
# 🆕 AUTO-CLEANUP SYSTEM
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
# 🆕 SMART DATASET MERGING
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
    history_positions = first_meta['history_positions']
    input_planes = first_meta['input_planes']
    
    for meta in all_metadata[1:]:
        if (meta['position_size'] != position_size or 
            meta['use_mtl'] != use_mtl or
            meta['history_positions'] != history_positions):
            raise ValueError("Cannot merge incompatible datasets! Different configs detected.")
    
    print(f"\n  ✅ All datasets compatible")
    print(f"  📊 Total positions: {total_positions:,}")
    
    # Merge binary files
    print(f"\n  🔨 Writing merged binary file...")
    chunk_size = 100 * 1024 * 1024  # 100 MB chunks
    
    with open(output_binary, 'wb') as outfile:
        for i, binary_file in enumerate(binary_files, 1):
            print(f"     Merging file {i}/{len(binary_files)}: {Path(binary_file).name}")
            
            with open(binary_file, 'rb') as infile:
                while True:
                    chunk = infile.read(chunk_size)
                    if not chunk:
                        break
                    outfile.write(chunk)
    
    # Create combined metadata
    combined_metadata = {
        'binary_file': str(output_binary),
        'total_positions': total_positions,
        'position_size': position_size,
        'use_mtl': use_mtl,
        'history_positions': history_positions,
        'input_planes': input_planes,
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
# PHASE 1: PGN EXTRACTION (Multi-processing for real CPU parallelism)
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
    This runs in a separate process, so it bypasses GIL!
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
    PHASE 1: Extract games using multi-processing (bypasses GIL)
    
    Each worker is a separate Python process with its own interpreter.
    This gives TRUE parallelism for CPU-bound parsing!
    """
    if phase1_workers <= 1:
        # Single process fallback
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
    
    # Process in parallel with separate processes
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
    PHASE 1: Sequential extraction (fallback for single worker)
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
    """
    Sort games by average Elo (descending)
    Returns sorted list with highest Elo games first
    """
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
    Uses multi-processing for true CPU parallelism
    
    Args:
        pgn_path: Path to PGN file
        max_games: Maximum games to extract
        phase1_threads: Number of processes
        sort_by_elo: If True, sort by Elo and take top max_games (default: True)
    """
    # Extract MORE games than needed if sorting (to ensure we get enough after filtering)
    # We'll extract 2x max_games, sort, then take top max_games
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
        
        # Take top max_games
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
# PHASE 2: POSITION EXTRACTION (Multi-processing for CPU parallelism)
# ==============================================================================

def extract_positions_from_game_worker(args):
    """
    PHASE 2 WORKER: Extract positions from a single game
    Runs in separate process for TRUE parallelism
    """
    game_data, min_elo, max_moves_per_game, use_mtl, history_positions = args
    
    import chess
    import struct
    import numpy as np
    
    # Import helpers locally (each process needs its own)
    from .utils.data_helpers import (
        board_to_compact,
        move_to_index,
        compute_material_balance,
        is_in_check,
        will_win
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
        
        positions = []
        board_history = []
        
        for move_uci in game_data['moves']:
            try:
                move = chess.Move.from_uci(move_uci)
                
                if move not in board.legal_moves:
                    break
                
                # Current board state (compact)
                current_board_compact = board_to_compact(board)
                
                # History (compact)
                history_compact_list = []
                if history_positions > 0:
                    # Get last N boards
                    hist_to_use = board_history[-history_positions:] if board_history else []
                    
                    # Pad with empty boards if needed
                    while len(hist_to_use) < history_positions:
                        empty = bytes(32)
                        hist_to_use.insert(0, empty)
                    
                    history_compact_list = hist_to_use
                
                # Move index
                move_idx = move_to_index(move)
                
                # Outcome
                if result == '1-0':
                    outcome = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == '0-1':
                    outcome = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    outcome = 0.0
                
                # MTL labels
                if use_mtl:
                    win_label = will_win(board, result)
                    material_label = compute_material_balance(board)
                    check_label = is_in_check(board)
                else:
                    win_label = material_label = check_label = 0.0
                
                # Pack position
                position_data = current_board_compact
                
                # Add history
                for hist_board in history_compact_list:
                    position_data += hist_board
                
                # Add move and outcome
                position_data += struct.pack('H', move_idx)
                position_data += struct.pack('f', outcome)
                
                # Add MTL if enabled
                if use_mtl:
                    position_data += struct.pack('f', win_label)
                    position_data += struct.pack('f', material_label)
                    position_data += struct.pack('f', check_label)
                
                positions.append(position_data)
                
                # Update history
                board_history.append(current_board_compact)
                
                # Make move
                board.push(move)
                
            except Exception as e:
                break
        
        return positions
        
    except Exception as e:
        return []


def extract_positions_parallel(games_data, config, phase2_workers):
    """
    PHASE 2: Extract positions from games using multi-processing
    
    Args:
        games_data: List of game dictionaries
        config: Configuration dictionary
        phase2_workers: Number of processes to use
    
    Returns:
        List of binary position data
    """
    min_elo = config['data'].get('min_elo', 0)
    max_moves = config['data'].get('max_moves_per_game', 200)
    use_mtl = config['model'].get('use_multitask_learning', False)
    history_pos = config['model'].get('history_positions', 0)
    
    if phase2_workers <= 1:
        # Sequential fallback
        return extract_positions_sequential(games_data, config)
    
    print(f"  Using {phase2_workers} processes for parallel position extraction...")
    
    # Prepare tasks
    tasks = [(game, min_elo, max_moves, use_mtl, history_pos) for game in games_data]
    
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
    history_pos = config['model'].get('history_positions', 0)
    
    all_positions = []
    
    for game in tqdm(games_data, desc="  Extracting positions"):
        task = (game, min_elo, max_moves, use_mtl, history_pos)
        positions = extract_positions_from_game_worker(task)
        all_positions.extend(positions)
    
    return all_positions


# ==============================================================================
# PHASE 3 & 4: DISK WRITING & CLEANUP
# ==============================================================================

def write_positions_to_disk(positions, binary_file):
    """
    PHASE 3: Write positions to binary file
    
    Args:
        positions: List of binary position data
        binary_file: Path to output file
    
    Returns:
        Number of positions written
    """
    print(f"\n  💾 Writing {len(positions):,} positions to disk...")
    
    with open(binary_file, 'wb') as f:
        for pos_data in tqdm(positions, desc="  Writing"):
            f.write(pos_data)
    
    size_mb = binary_file.stat().st_size / (1024**2)
    print(f"  ✅ Written: {size_mb:.1f} MB")
    
    return len(positions)


# ==============================================================================
# 🆕 MAIN PROCESSING FUNCTION WITH SMART MANAGEMENT
# ==============================================================================

def process_pgn_files_smart(pgn_files, config):
    """
    🆕 Smart PGN processing with:
    - Dataset tracking (avoid reprocessing)
    - Auto-cleanup (remove intermediate files)
    - Smart merging (reuse compatible datasets)
    
    Args:
        pgn_files: List of PGN file paths
        config: Configuration dictionary
    
    Returns:
        Combined metadata dictionary
    """
    # Get directory where data.py is located (chess/src/)
    # Go up one level to project root (chess/)
    project_root = Path(__file__).parent.parent
    
    # Create preprocessing directory relative to project root
    # Result: chess/data/preprocessing/
    preprocessing_dir = project_root / config['paths']['data_dir'] / 'preprocessing'
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracker
    tracker = DatasetTracker(preprocessing_dir)
    
    # List existing datasets
    tracker.list_processed_datasets()
    
    # Check which files need processing
    to_process = []
    existing_binaries = []
    existing_metadatas = []
    
    print(f"\n{'='*70}")
    print(f"🔍 Checking {len(pgn_files)} PGN files...")
    print(f"{'='*70}\n")
    
    for pgn_file in pgn_files:
        binary_path, metadata_path = tracker.get_processed_dataset(pgn_file, config)
        
        if binary_path and metadata_path:
            # Already processed!
            existing_binaries.append(binary_path)
            existing_metadatas.append(metadata_path)
            print(f"✅ SKIP: {Path(pgn_file).name} (already processed)")
        else:
            # Needs processing
            to_process.append(pgn_file)
            print(f"🆕 PROCESS: {Path(pgn_file).name}")
    
    print(f"\n{'='*70}")
    print(f"  • Already processed: {len(existing_binaries)} files")
    print(f"  • Need processing: {len(to_process)} files")
    print(f"{'='*70}\n")
    
    # Process new files
    new_binaries = []
    new_metadatas = []
    
    if to_process:
        print(f"\n{'='*70}")
        print(f"⚙️  Processing {len(to_process)} new files...")
        print(f"{'='*70}\n")
        
        for i, pgn_file in enumerate(to_process, 1):
            print(f"\n[{i}/{len(to_process)}] Processing: {Path(pgn_file).name}")
            print("="*70)
            
            # Generate output names
            pgn_name = Path(pgn_file).stem
            binary_file = preprocessing_dir / f"{pgn_name}.bin"
            metadata_file = preprocessing_dir / f"{pgn_name}_meta.pkl"
            
            # PHASE 1: Extract games
            print("\n📖 PHASE 1: Extracting games from PGN...")
            games_data = extract_games_from_pgn_parallel(
                pgn_file,
                config['data']['max_games'],
                config['data']['phase1_threads'],
                sort_by_elo=config['data'].get('sort_by_avg_elo', True)
            )
            
            if not games_data:
                print(f"⚠️  No games extracted from {Path(pgn_file).name}, skipping...")
                continue
            
            # PHASE 2: Extract positions
            print(f"\n♟️  PHASE 2: Extracting positions...")
            positions = extract_positions_parallel(
                games_data,
                config,
                config['data']['phase2_threads']
            )
            
            if not positions:
                print(f"⚠️  No positions extracted, skipping...")
                continue
            
            # PHASE 3: Write to disk
            print(f"\n💾 PHASE 3: Writing to disk...")
            total_positions = write_positions_to_disk(positions, binary_file)
            
            # Calculate position size
            use_mtl = config['model'].get('use_multitask_learning', False)
            history_pos = config['model'].get('history_positions', 0)
            input_planes = 12 * (1 + history_pos)
            
            base_size = 32 + (32 * history_pos) + 2 + 4
            position_size = base_size + (12 if use_mtl else 0)
            
            # Create metadata
            metadata = {
                'binary_file': str(binary_file),
                'total_positions': total_positions,
                'position_size': position_size,
                'use_mtl': use_mtl,
                'history_positions': history_pos,
                'input_planes': input_planes
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            # PHASE 4: Cleanup intermediate data
            print(f"\n♻️  PHASE 4: Cleanup...")
            del games_data
            del positions
            gc.collect()
            
            # Register in tracker
            tracker.register_dataset(pgn_file, config, binary_file, metadata_file, total_positions)
            
            new_binaries.append(binary_file)
            new_metadatas.append(metadata_file)
            
            print(f"\n✅ Completed: {Path(pgn_file).name}")
            print(f"   Positions: {total_positions:,}")
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


# ==============================================================================
# DATASET AND DATALOADER (unchanged)
# ==============================================================================

class BinaryChessDataset(Dataset):
    """Memory-mapped dataset with MTL and HISTORY support"""
    
    def __init__(self, binary_file, indices, position_size=38, use_mtl=None, history_positions=0):
        self.binary_file = binary_file
        self.indices = indices
        self.position_size = position_size
        self.history_positions = history_positions
        
        # Auto-detect MTL
        if use_mtl is None:
            base_size = 32 + (32 * history_positions) + 2 + 4
            self.use_mtl = (position_size == base_size + 12)
        else:
            self.use_mtl = use_mtl
        
        self._mmap = None
        self._file = None
    
    def _ensure_mmap(self):
        """Open memory-mapped file (per worker)"""
        if self._mmap is None:
            import mmap
            self._file = open(self.binary_file, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        self._ensure_mmap()
        
        position_idx = self.indices[idx]
        offset = position_idx * self.position_size
        data = self._mmap[offset:offset + self.position_size]
        
        # Unpack current board
        compact_board = data[:32]
        current_offset = 32
        
        # Unpack history
        history_size = 32 * self.history_positions
        if history_size > 0:
            history_bytes = data[current_offset:current_offset + history_size]
            current_offset += history_size
            
            # Convert history boards to tensors
            history_tensors = []
            for i in range(self.history_positions):
                hist_board = history_bytes[i*32:(i+1)*32]
                history_tensors.append(compact_to_tensor(hist_board))
        else:
            history_tensors = []
        
        # Unpack move and outcome
        move_index = struct.unpack('H', data[current_offset:current_offset+2])[0]
        current_offset += 2
        outcome = struct.unpack('f', data[current_offset:current_offset+4])[0]
        current_offset += 4
        
        # Convert current board
        board_tensor = compact_to_tensor(compact_board)
        
        # Stack history with current board
        if history_tensors:
            # Stack: [oldest_history, ..., newest_history, current]
            # Shape: (12 * (history_positions + 1), 8, 8)
            all_tensors = history_tensors + [board_tensor]
            stacked_board = np.concatenate(all_tensors, axis=0)
        else:
            stacked_board = board_tensor
        
        if self.use_mtl:
            # Unpack MTL labels
            win = struct.unpack('f', data[current_offset:current_offset+4])[0]
            current_offset += 4
            material = struct.unpack('f', data[current_offset:current_offset+4])[0]
            current_offset += 4
            check = struct.unpack('f', data[current_offset:current_offset+4])[0]
            
            return {
                'board': torch.FloatTensor(stacked_board),
                'move': torch.LongTensor([move_index])[0],
                'value': torch.FloatTensor([outcome]),
                'win': torch.FloatTensor([win]),
                'material': torch.FloatTensor([material]),
                'check': torch.FloatTensor([check])
            }
        else:
            return (
                torch.FloatTensor(stacked_board),
                torch.LongTensor([move_index])[0],
                torch.FloatTensor([outcome])
            )
    
    def __del__(self):
        if self._mmap is not None:
            self._mmap.close()
        if self._file is not None:
            self._file.close()


def create_dataloaders(metadata, config):
    """Create dataloaders with MTL and HISTORY support"""
    
    if metadata['total_positions'] == 0:
        raise ValueError("Cannot create dataloaders with 0 positions.")
    
    total_positions = metadata['total_positions']
    all_indices = list(range(total_positions))
    
    # Split
    split_idx = int(len(all_indices) * config['data']['train_split'])
    split_idx = max(1, min(split_idx, len(all_indices) - 1))
    
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    print(f"\n{'='*70}")
    print("📊 Dataset split:")
    print(f"  • Train: {len(train_indices):,} positions")
    print(f"  • Val: {len(val_indices):,} positions")
    
    # Get metadata
    position_size = metadata.get('position_size', 38)
    use_mtl = metadata.get('use_mtl', False)
    history_positions = metadata.get('history_positions', 0)
    input_planes = metadata.get('input_planes', 12)
    
    print(f"  • Position size: {position_size} bytes")
    print(f"  • MTL: {use_mtl}")
    print(f"  • History: {history_positions} positions")
    print(f"  • Input planes: {input_planes}")
    print(f"{'='*70}\n")
    
    # Create datasets
    train_dataset = BinaryChessDataset(
        metadata['binary_file'], 
        train_indices, 
        position_size=position_size,
        use_mtl=use_mtl,
        history_positions=history_positions
    )
    val_dataset = BinaryChessDataset(
        metadata['binary_file'], 
        val_indices,
        position_size=position_size,
        use_mtl=use_mtl,
        history_positions=history_positions
    )
    
    # Dataloaders
    prefetch_factor = config['hardware']['prefetch_factor']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['imitation_learning']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False,
        prefetch_factor=prefetch_factor if config['hardware']['num_workers'] > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['imitation_learning']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False,
        prefetch_factor=prefetch_factor if config['hardware']['num_workers'] > 0 else None
    )
    
    print(f"✓ DataLoaders created (prefetch_factor={prefetch_factor})")
    
    return train_loader, val_loader