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
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor


def board_to_tensor(board):
    """
    Convert chess.Board to tensor representation
    Returns: (12, 8, 8) tensor - 6 piece types x 2 colors
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = square // 8
            col = square % 8
            piece_idx = piece_to_idx[piece.piece_type]
            if piece.color == chess.WHITE:
                channel = piece_idx
            else:
                channel = piece_idx + 6
            tensor[channel, row, col] = 1.0
    
    return tensor


def board_to_compact(board):
    """
    Convert board to ultra-compact binary representation
    Each position: 64 squares × 4 bits = 32 bytes
    """
    piece_to_code = {
        (chess.PAWN, chess.WHITE): 1,
        (chess.KNIGHT, chess.WHITE): 2,
        (chess.BISHOP, chess.WHITE): 3,
        (chess.ROOK, chess.WHITE): 4,
        (chess.QUEEN, chess.WHITE): 5,
        (chess.KING, chess.WHITE): 6,
        (chess.PAWN, chess.BLACK): 7,
        (chess.KNIGHT, chess.BLACK): 8,
        (chess.BISHOP, chess.BLACK): 9,
        (chess.ROOK, chess.BLACK): 10,
        (chess.QUEEN, chess.BLACK): 11,
        (chess.KING, chess.BLACK): 12,
    }
    
    codes = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            code = piece_to_code[(piece.piece_type, piece.color)]
        else:
            code = 0
        codes.append(code)
    
    # Pack pairs of codes into bytes
    packed = bytearray(32)
    for i in range(0, 64, 2):
        packed[i // 2] = (codes[i] << 4) | codes[i + 1]
    
    return bytes(packed)


def compact_to_tensor(compact_board):
    """Convert compact representation back to tensor"""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    for i in range(32):
        byte = compact_board[i]
        code1 = (byte >> 4) & 0x0F
        code2 = byte & 0x0F
        
        square1 = i * 2
        square2 = i * 2 + 1
        
        for square, code in [(square1, code1), (square2, code2)]:
            if code == 0:
                continue
            
            row = square // 8
            col = square % 8
            
            if 1 <= code <= 6:
                channel = code - 1
            else:
                channel = code - 1
            
            tensor[channel, row, col] = 1.0
    
    return tensor


def move_to_index(move):
    """Convert chess.Move to index (0-4095)"""
    return move.from_square * 64 + move.to_square


def index_to_move(index):
    """Convert index back to move"""
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)


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


def process_extracted_game(game_data, min_elo, max_moves):
    """Process extracted game data"""
    try:
        if game_data is None:
            return []
        
        white_elo = game_data['white_elo']
        black_elo = game_data['black_elo']
        
        if white_elo == '?' or black_elo == '?':
            return []
        
        try:
            white_elo_int = int(white_elo)
            black_elo_int = int(black_elo)
            
            if white_elo_int < min_elo or black_elo_int < min_elo:
                return []
        except (ValueError, TypeError):
            return []
        
        result = game_data['result']
        if result == '1-0':
            outcome = 1.0
        elif result == '0-1':
            outcome = -1.0
        elif result == '1/2-1/2':
            outcome = 0.0
        else:
            return []
        
        data = []
        board = chess.Board()
        move_count = 0
        
        for move_uci in game_data['moves']:
            if move_count >= max_moves:
                break
            
            try:
                move = chess.Move.from_uci(move_uci)
                
                compact_board = board_to_compact(board)
                move_index = np.uint16(move_to_index(move))
                current_outcome = outcome if board.turn == chess.WHITE else -outcome
                
                data.append((compact_board, move_index, np.float32(current_outcome)))
                
                board.push(move)
                move_count += 1
            except:
                break
        
        return data
    except:
        return []


def process_game_batch(args):
    """Process a batch of extracted game data in parallel worker"""
    games_data, min_elo, max_moves = args
    all_positions = []
    
    for game_data in games_data:
        positions = process_extracted_game(game_data, min_elo, max_moves)
        all_positions.extend(positions)
    
    return all_positions


class BinaryDataWriter:
    """Write positions to binary file for maximum compression"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'wb')
        self.count = 0
    
    def write_position(self, compact_board, move_index, outcome):
        """Write one position (38 bytes total)"""
        self.file.write(compact_board)  # 32 bytes
        self.file.write(struct.pack('H', move_index))  # 2 bytes
        self.file.write(struct.pack('f', outcome))  # 4 bytes
        self.count += 1
    
    def close(self):
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def parse_single_pgn_file(args):
    """
    🚀 Process a single PGN file - designed for parallel execution
    Returns: (filename, positions_count, output_path)
    """
    pgn_path, output_path, min_elo, max_games, max_moves, file_index, select_top, sort_by_elo = args
    
    print(f"[Worker {file_index}] Processing: {pgn_path.name}")
    
    # Extract ALL games that meet min_elo threshold
    games_data = []
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            game_data = extract_game_data(game)
            if game_data:
                # Check min_elo threshold
                try:
                    white_elo = int(game_data['white_elo']) if game_data['white_elo'] != '?' else 0
                    black_elo = int(game_data['black_elo']) if game_data['black_elo'] != '?' else 0
                    
                    if white_elo >= min_elo and black_elo >= min_elo:
                        # Store with average Elo for sorting
                        avg_elo = (white_elo + black_elo) / 2
                        games_data.append((game_data, avg_elo))
                except (ValueError, TypeError):
                    pass
    
    print(f"[Worker {file_index}] Extracted {len(games_data)} games (Elo >= {min_elo}) from {pgn_path.name}")
    
    # 🎯 Sort by Elo and select top games
    if select_top and sort_by_elo and len(games_data) > 0:
        print(f"[Worker {file_index}] Sorting by average Elo...")
        games_data.sort(key=lambda x: x[1], reverse=True)  # Highest Elo first
        
        # Take top N games
        games_to_process = min(max_games, len(games_data))
        games_data = games_data[:games_to_process]
        
        if len(games_data) > 0:
            top_elo = games_data[0][1]
            bottom_elo = games_data[-1][1]
            print(f"[Worker {file_index}] Selected top {len(games_data)} games (Elo range: {bottom_elo:.0f}-{top_elo:.0f})")
    
    # Extract just the game_data (without Elo)
    games_data = [g[0] for g in games_data]
    
    # Process games in batches (using multiprocessing within this file)
    batch_size = max(1, len(games_data) // (mp.cpu_count() * 2))
    batches = []
    for i in range(0, len(games_data), batch_size):
        batch = games_data[i:i + batch_size]
        batches.append((batch, min_elo, max_moves))
    
    all_positions = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(process_game_batch, batch) for batch in batches]
        
        for future in as_completed(futures):
            positions = future.result()
            all_positions.extend(positions)
    
    # Write to binary file
    with BinaryDataWriter(output_path) as writer:
        for compact_board, move_index, outcome in all_positions:
            writer.write_position(compact_board, move_index, outcome)
    
    print(f"[Worker {file_index}] ✓ {pgn_path.name}: {len(all_positions)} positions → {output_path.name}")
    
    return pgn_path.name, len(all_positions), output_path


def parse_pgn_files(data_dir, config, num_workers=None):
    """
    🚀 Parse multiple PGN files in parallel - one process per file
    Extracts ALL games above min_elo, sorts by Elo, then takes top max_games
    Returns: metadata dict
    """
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / data_dir
    preprocessing_dir = script_dir / "data" / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    
    pgn_files = sorted(list(data_path.glob("*.pgn")))
    
    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {data_path}")
    
    print(f"\n{'='*70}")
    print(f"Found {len(pgn_files)} PGN file(s):")
    for pgn in pgn_files:
        size_mb = pgn.stat().st_size / (1024**2)
        print(f"  • {pgn.name} ({size_mb:.1f} MB)")
    print(f"{'='*70}\n")
    
    min_elo = config['data']['min_elo']
    max_games = config['data']['max_games']
    max_moves = config['data']['max_moves_per_game']
    select_top = config['data'].get('select_top_games', True)
    sort_by_elo = config['data'].get('sort_by_avg_elo', True)
    
    # Cache filename includes selection strategy
    cache_suffix = f"_top{max_games}" if select_top else f"_first{max_games}"
    binary_file = preprocessing_dir / f"positions_elo{min_elo}{cache_suffix}_moves{max_moves}.bin"
    metadata_file = preprocessing_dir / f"positions_elo{min_elo}{cache_suffix}_moves{max_moves}_meta.pkl"
    
    # Check cache
    if binary_file.exists() and metadata_file.exists():
        print(f"📦 Loading from cache: {binary_file.name}")
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            if metadata['total_positions'] > 0:
                size_mb = binary_file.stat().st_size / (1024**2)
                print(f"✓ Loaded {metadata['total_positions']:,} positions ({size_mb:.1f} MB)")
                return metadata
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            print("Will reprocess PGN files...")
    
    # Parallel processing configuration
    use_parallel = config['data'].get('parallel_pgn_processing', True)
    max_parallel_workers = config['data'].get('max_workers', None)
    
    if max_parallel_workers is None:
        max_parallel_workers = len(pgn_files)  # One worker per file
    
    games_per_file = max_games // len(pgn_files) if len(pgn_files) > 1 else max_games
    
    print(f"🚀 Processing Configuration:")
    print(f"  • Parallel processing: {'ENABLED' if use_parallel else 'DISABLED'}")
    print(f"  • Workers: {max_parallel_workers if use_parallel else 1}")
    print(f"  • Strategy: {'TOP games by Elo' if select_top else 'FIRST games found'}")
    print(f"  • Min Elo: {min_elo}")
    print(f"  • Target games per file: {games_per_file:,}")
    print()
    
    # Prepare tasks for parallel processing
    tasks = []
    temp_files = []
    
    for idx, pgn_file in enumerate(pgn_files):
        temp_output = preprocessing_dir / f"temp_{idx}_{pgn_file.stem}.bin"
        temp_files.append(temp_output)
        tasks.append((
            pgn_file,           # pgn_path
            temp_output,        # output_path
            min_elo,            # min_elo
            games_per_file,     # max_games (per file)
            max_moves,          # max_moves
            idx,                # file_index
            select_top,         # select_top_games
            sort_by_elo         # sort_by_avg_elo
        ))
    
    # Process files
    results = []
    total_positions = 0
    
    if use_parallel and len(pgn_files) > 1:
        # 🚀 PARALLEL: Process multiple PGN files simultaneously
        print(f"🚀 Processing {len(pgn_files)} PGN files in parallel...\n")
        
        with ProcessPoolExecutor(max_workers=max_parallel_workers) as executor:
            futures = [executor.submit(parse_single_pgn_file, task) for task in tasks]
            
            # Progress bar
            pbar = tqdm(total=len(futures), desc="Processing PGN files")
            
            for future in as_completed(futures):
                filename, positions_count, output_path = future.result()
                results.append((filename, positions_count, output_path))
                total_positions += positions_count
                pbar.update(1)
            
            pbar.close()
    else:
        # Sequential processing (for single file or if parallel disabled)
        print(f"Processing PGN files sequentially...\n")
        
        for task in tasks:
            filename, positions_count, output_path = parse_single_pgn_file(task)
            results.append((filename, positions_count, output_path))
            total_positions += positions_count
    
    # Summary
    print(f"\n{'='*70}")
    print("Processing Summary:")
    for filename, positions_count, _ in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"  • {filename}: {positions_count:,} positions")
    print(f"{'='*70}")
    print(f"Total: {total_positions:,} positions\n")
    
    if total_positions == 0:
        print("⚠️  WARNING: No positions extracted!")
        return {'binary_file': str(binary_file), 'total_positions': 0}
    
    # Merge temporary files
    if len(temp_files) > 1:
        print("📦 Merging binary files...")
        with open(binary_file, 'wb') as outfile:
            for temp_file in tqdm(temp_files, desc="Merging"):
                if temp_file.exists():
                    with open(temp_file, 'rb') as infile:
                        outfile.write(infile.read())
                    temp_file.unlink()
    else:
        if temp_files[0].exists():
            temp_files[0].rename(binary_file)
    
    # Save metadata
    metadata = {
        'binary_file': str(binary_file),
        'total_positions': total_positions,
        'position_size': 38
    }
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    final_size_mb = binary_file.stat().st_size / (1024**2)
    compression_ratio = 100 * (1 - final_size_mb / (total_positions * 3072 / 1024**2))
    
    print(f"✓ Final file: {binary_file.name}")
    print(f"✓ Size: {final_size_mb:.1f} MB")
    print(f"✓ Compression: {compression_ratio:.1f}% saved vs raw tensors")
    
    return metadata


class BinaryChessDataset(Dataset):
    """Memory-mapped binary dataset - no RAM needed!"""
    
    def __init__(self, binary_file, indices, position_size=38):
        self.binary_file = binary_file
        self.indices = indices
        self.position_size = position_size
        self._mmap = None
        self._file = None
    
    def _ensure_mmap(self):
        """Open memory-mapped file if not already open (per worker)"""
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
        
        compact_board = data[:32]
        move_index = struct.unpack('H', data[32:34])[0]
        outcome = struct.unpack('f', data[34:38])[0]
        
        board_tensor = compact_to_tensor(compact_board)
        
        return (
            torch.FloatTensor(board_tensor),
            torch.LongTensor([move_index])[0],
            torch.FloatTensor([outcome])
        )
    
    def __del__(self):
        if self._mmap is not None:
            self._mmap.close()
        if self._file is not None:
            self._file.close()


def create_dataloaders(metadata, config):
    """Create train and validation dataloaders from binary file"""
    
    if metadata['total_positions'] == 0:
        raise ValueError("Cannot create dataloaders with 0 positions.")
    
    total_positions = metadata['total_positions']
    all_indices = list(range(total_positions))
    
    # Split
    split_idx = int(len(all_indices) * config['data']['train_split'])
    split_idx = max(1, min(split_idx, len(all_indices) - 1))
    
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    print(f"📊 Dataset split:")
    print(f"  • Train: {len(train_indices):,} positions")
    print(f"  • Val: {len(val_indices):,} positions")
    
    # Create datasets
    train_dataset = BinaryChessDataset(metadata['binary_file'], train_indices)
    val_dataset = BinaryChessDataset(metadata['binary_file'], val_indices)
    
    # Dataloaders
    prefetch_factor = 4
    
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