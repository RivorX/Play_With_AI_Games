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
from concurrent.futures import ProcessPoolExecutor, as_completed


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
    Each position: 64 squares × 4 bits (16 piece types: empty + 6×2 colors + padding)
    = 32 bytes per position (vs 3072 bytes for tensor!)
    
    Piece encoding:
    0 = empty
    1-6 = white pieces (pawn, knight, bishop, rook, queen, king)
    7-12 = black pieces (pawn, knight, bishop, rook, queen, king)
    """
    # Pack two squares per byte (4 bits each)
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
    
    # Unpack
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
            
            if 1 <= code <= 6:  # White pieces
                channel = code - 1
            else:  # Black pieces (7-12)
                channel = code - 1
            
            tensor[channel, row, col] = 1.0
    
    return tensor


def move_to_index(move):
    """Convert chess.Move to index (0-4095) for from_square*64 + to_square"""
    return move.from_square * 64 + move.to_square


def index_to_move(index):
    """Convert index back to move (for decoding predictions)"""
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)


def process_game(game, min_elo, max_moves):
    """
    Process a single chess.pgn.Game object
    Returns list of compact positions
    """
    try:
        if game is None:
            return []
        
        # Filter by Elo
        white_elo = game.headers.get('WhiteElo', '?')
        black_elo = game.headers.get('BlackElo', '?')
        
        if white_elo == '?' or black_elo == '?':
            return []
        
        try:
            white_elo_int = int(white_elo)
            black_elo_int = int(black_elo)
            
            if white_elo_int < min_elo or black_elo_int < min_elo:
                return []
        except (ValueError, TypeError):
            return []
        
        # Get outcome
        result = game.headers.get('Result', '*')
        if result == '1-0':
            outcome = 1.0
        elif result == '0-1':
            outcome = -1.0
        elif result == '1/2-1/2':
            outcome = 0.0
        else:
            return []
        
        # Extract positions as compact format
        data = []
        board = game.board()
        move_count = 0
        
        for move in game.mainline_moves():
            if move_count >= max_moves:
                break
            
            # Store in ultra-compact format
            compact_board = board_to_compact(board)
            move_index = np.uint16(move_to_index(move))
            current_outcome = outcome if board.turn == chess.WHITE else -outcome
            
            # Store as: 32 bytes (board) + 2 bytes (move) + 4 bytes (outcome) = 38 bytes total
            data.append((compact_board, move_index, np.float32(current_outcome)))
            
            board.push(move)
            move_count += 1
        
        return data
    
    except Exception:
        return []


class BinaryDataWriter:
    """Write positions to binary file for maximum compression"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'wb')
        self.count = 0
    
    def write_position(self, compact_board, move_index, outcome):
        """Write one position (38 bytes total)"""
        self.file.write(compact_board)  # 32 bytes
        self.file.write(struct.pack('H', move_index))  # 2 bytes (unsigned short)
        self.file.write(struct.pack('f', outcome))  # 4 bytes (float)
        self.count += 1
    
    def close(self):
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def extract_game_data(game):
    """Extract serializable data from chess.pgn.Game"""
    try:
        if game is None:
            return None
        
        # Extract headers
        white_elo = game.headers.get('WhiteElo', '?')
        black_elo = game.headers.get('BlackElo', '?')
        result = game.headers.get('Result', '*')
        
        # Extract moves as UCI strings
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
    """Process extracted game data (serializable)"""
    try:
        if game_data is None:
            return []
        
        # Filter by Elo
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
        
        # Get outcome
        result = game_data['result']
        if result == '1-0':
            outcome = 1.0
        elif result == '0-1':
            outcome = -1.0
        elif result == '1/2-1/2':
            outcome = 0.0
        else:
            return []
        
        # Replay game and extract positions
        data = []
        board = chess.Board()
        move_count = 0
        
        for move_uci in game_data['moves']:
            if move_count >= max_moves:
                break
            
            try:
                move = chess.Move.from_uci(move_uci)
                
                # Store position before move
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


def parse_pgn_file_to_binary(pgn_path, output_path, config, num_workers=None):
    """
    Parse PGN file directly to binary format with parallel processing
    Returns: number of positions written
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    min_elo = config['data']['min_elo']
    max_games = config['data']['max_games']
    max_moves = config['data']['max_moves_per_game']
    
    # Use all available cores
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"Reading PGN file: {pgn_path}")
    print(f"Writing to: {output_path}")
    print(f"Using {num_workers} worker processes")
    
    # First pass: extract serializable data from games
    print("Extracting game data from PGN...")
    games_data = []
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        pbar = tqdm(desc="Extracting", total=max_games)
        for _ in range(max_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            game_data = extract_game_data(game)
            if game_data:
                games_data.append(game_data)
            
            pbar.update(1)
        pbar.close()
    
    print(f"Extracted {len(games_data)} games")
    
    # Split games into batches for parallel processing
    batch_size = max(1, len(games_data) // (num_workers * 4))  # 4 batches per worker
    batches = []
    for i in range(0, len(games_data), batch_size):
        batch = games_data[i:i + batch_size]
        batches.append((batch, min_elo, max_moves))
    
    print(f"Processing {len(batches)} batches with {num_workers} workers...")
    
    # Process batches in parallel
    all_positions = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_game_batch, batch) for batch in batches]
        
        pbar = tqdm(desc="Processing", total=len(batches))
        for future in as_completed(futures):
            positions = future.result()
            all_positions.extend(positions)
            pbar.update(1)
        pbar.close()
    
    # Write all positions to binary file
    print(f"Writing {len(all_positions)} positions to disk...")
    with BinaryDataWriter(output_path) as writer:
        for compact_board, move_index, outcome in tqdm(all_positions, desc="Writing"):
            writer.write_position(compact_board, move_index, outcome)
    
    print(f"Processed {len(games_data)} games")
    print(f"Valid positions: {len(all_positions)}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    return len(all_positions)


def parse_pgn_files(data_dir, config, num_workers=None):
    """
    Parse PGN files to binary format with memory mapping support
    Returns: metadata dict
    """
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / data_dir
    preprocessing_dir = script_dir / "data" / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    
    pgn_files = list(data_path.glob("*.pgn"))
    
    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {data_path}")
    
    print(f"Found {len(pgn_files)} PGN file(s): {[f.name for f in pgn_files]}")
    
    min_elo = config['data']['min_elo']
    max_games = config['data']['max_games']
    max_moves = config['data']['max_moves_per_game']
    
    binary_file = preprocessing_dir / f"positions_elo{min_elo}_games{max_games}_moves{max_moves}.bin"
    metadata_file = preprocessing_dir / f"positions_elo{min_elo}_games{max_games}_moves{max_moves}_meta.pkl"
    
    # Check if cache exists
    if binary_file.exists() and metadata_file.exists():
        print(f"Loading from cache: {binary_file}")
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            if metadata['total_positions'] > 0:
                print(f"Loaded metadata: {metadata['total_positions']} positions")
                print(f"File size: {binary_file.stat().st_size / (1024**2):.2f} MB")
                return metadata
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    # Process PGN files
    print("\nProcessing PGN files to binary format...")
    
    total_positions = 0
    games_per_file = max_games // len(pgn_files) if len(pgn_files) > 1 else max_games
    
    # Create temporary files for each PGN
    temp_files = []
    for idx, pgn_file in enumerate(pgn_files):
        print(f"\nProcessing: {pgn_file.name}")
        
        temp_output = preprocessing_dir / f"temp_{idx}.bin"
        
        file_config = config.copy()
        file_config['data'] = config['data'].copy()
        file_config['data']['max_games'] = games_per_file
        
        positions_count = parse_pgn_file_to_binary(
            pgn_file, temp_output, file_config, 
            num_workers=num_workers
        )
        total_positions += positions_count
        temp_files.append(temp_output)
        
        gc.collect()
    
    # Merge temporary files into one
    if len(temp_files) > 1:
        print("\nMerging files...")
        with open(binary_file, 'wb') as outfile:
            for temp_file in temp_files:
                with open(temp_file, 'rb') as infile:
                    outfile.write(infile.read())
                temp_file.unlink()  # Delete temp file
    else:
        temp_files[0].rename(binary_file)
    
    if total_positions == 0:
        print("\n" + "="*60)
        print("WARNING: No positions extracted!")
        print("="*60)
        return {'binary_file': str(binary_file), 'total_positions': 0}
    
    # Save metadata
    metadata = {
        'binary_file': str(binary_file),
        'total_positions': total_positions,
        'position_size': 38  # bytes per position
    }
    
    print(f"\nSaving metadata: {metadata_file}")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    final_size_mb = binary_file.stat().st_size / (1024**2)
    print(f"Total: {total_positions} positions")
    print(f"Final file size: {final_size_mb:.2f} MB")
    print(f"Compression: {100 * (1 - final_size_mb / (total_positions * 3072 / 1024**2)):.1f}% saved")
    
    return metadata


class BinaryChessDataset(Dataset):
    """Memory-mapped binary dataset - no RAM needed!"""
    
    def __init__(self, binary_file, indices, position_size=38):
        """
        Args:
            binary_file: path to binary data file
            indices: list of position indices for this split
            position_size: bytes per position (38)
        """
        self.binary_file = binary_file
        self.indices = indices
        self.position_size = position_size
        
        # Don't open file here - will be opened per worker
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
        # Ensure mmap is open in this worker
        self._ensure_mmap()
        
        position_idx = self.indices[idx]
        
        # Seek to position in file
        offset = position_idx * self.position_size
        
        # Read 38 bytes
        data = self._mmap[offset:offset + self.position_size]
        
        # Parse: 32 bytes (board) + 2 bytes (move) + 4 bytes (outcome)
        compact_board = data[:32]
        move_index = struct.unpack('H', data[32:34])[0]
        outcome = struct.unpack('f', data[34:38])[0]
        
        # Convert compact board to tensor
        board_tensor = compact_to_tensor(compact_board)
        
        return (
            torch.FloatTensor(board_tensor),
            torch.LongTensor([move_index])[0],
            torch.FloatTensor([outcome])
        )
    
    def __del__(self):
        """Clean up when dataset is destroyed"""
        if self._mmap is not None:
            self._mmap.close()
        if self._file is not None:
            self._file.close()


def create_dataloaders(metadata, config):
    """Create train and validation dataloaders from binary file"""
    
    if metadata['total_positions'] == 0:
        raise ValueError("Cannot create dataloaders with 0 positions.")
    
    total_positions = metadata['total_positions']
    
    # Create indices
    all_indices = list(range(total_positions))
    
    # Split
    split_idx = int(len(all_indices) * config['data']['train_split'])
    
    if split_idx == 0:
        split_idx = 1
    if split_idx >= len(all_indices):
        split_idx = len(all_indices) - 1
    
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    print(f"Train dataset: {len(train_indices)} positions")
    print(f"Val dataset: {len(val_indices)} positions")
    
    # Create datasets (memory-mapped, no RAM!)
    train_dataset = BinaryChessDataset(metadata['binary_file'], train_indices)
    val_dataset = BinaryChessDataset(metadata['binary_file'], val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['imitation_learning']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False,
        prefetch_factor=2 if config['hardware']['num_workers'] > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['imitation_learning']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False,
        prefetch_factor=2 if config['hardware']['num_workers'] > 0 else None
    )
    
    return train_loader, val_loader