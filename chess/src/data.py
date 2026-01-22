import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gc


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
    Args:
        game: chess.pgn.Game object
        min_elo: minimum Elo rating
        max_moves: maximum moves per game
    Returns:
        list of (board_tensor, move_index, outcome)
    """
    try:
        if game is None:
            return []
        
        # Filter by Elo
        white_elo = game.headers.get('WhiteElo', '?')
        black_elo = game.headers.get('BlackElo', '?')
        
        # Skip games without Elo or with '?' Elo
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
        
        # Extract positions and moves
        data = []
        board = game.board()
        move_count = 0
        
        for move in game.mainline_moves():
            if move_count >= max_moves:
                break
            
            # Current position
            board_tensor = board_to_tensor(board)
            move_index = move_to_index(move)
            
            # Flip outcome for black's perspective
            current_outcome = outcome if board.turn == chess.WHITE else -outcome
            
            data.append((board_tensor, move_index, current_outcome))
            
            board.push(move)
            move_count += 1
        
        return data
    
    except Exception as e:
        # Silently skip problematic games
        return []


def parse_pgn_file_sequential(pgn_path, config):
    """
    Parse PGN file sequentially (single-threaded but reliable)
    """
    min_elo = config['data']['min_elo']
    max_games = config['data']['max_games']
    max_moves = config['data']['max_moves_per_game']
    
    all_data = []
    games_processed = 0
    games_with_elo = 0
    
    print(f"Reading and processing PGN file: {pgn_path}")
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        pbar = tqdm(desc="Games", total=max_games)
        
        while games_processed < max_games:
            game = chess.pgn.read_game(f)
            
            if game is None:
                break
            
            games_processed += 1
            
            # Process game
            game_data = process_game(game, min_elo, max_moves)
            
            if game_data:
                games_with_elo += 1
                all_data.extend(game_data)
            
            pbar.update(1)
            
            if games_processed % 1000 == 0:
                pbar.set_postfix({
                    'positions': len(all_data),
                    'valid_games': games_with_elo
                })
        
        pbar.close()
    
    print(f"Processed {games_processed} games")
    print(f"Games with Elo >= {min_elo}: {games_with_elo}")
    print(f"Total positions: {len(all_data)}")
    
    return all_data


def parse_pgn_files(data_dir, config, num_workers=None):
    """
    Parse all PGN files in data directory with caching
    Returns: list of (board_tensor, move_index, outcome)
    """
    # Get base directory (relative to this file)
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / data_dir
    preprocessing_dir = script_dir / "data" / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PGN files
    pgn_files = list(data_path.glob("*.pgn"))
    
    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {data_path}")
    
    print(f"Found {len(pgn_files)} PGN file(s): {[f.name for f in pgn_files]}")
    
    min_elo = config['data']['min_elo']
    max_games = config['data']['max_games']
    max_moves = config['data']['max_moves_per_game']
    
    # Create cache filename based on config
    cache_file = preprocessing_dir / f"cache_elo{min_elo}_games{max_games}_moves{max_moves}.pkl"
    
    # Try to load from cache
    if cache_file.exists():
        print(f"Loading preprocessed data from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            if len(data) > 0:
                print(f"Loaded {len(data)} positions from cache")
                return data
            else:
                print("WARNING: Cache contains 0 positions. Reprocessing...")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            print("Reprocessing data...")
    
    # Process all PGN files
    all_data = []
    games_per_file = max_games // len(pgn_files) if len(pgn_files) > 1 else max_games
    
    for pgn_file in pgn_files:
        print(f"\nProcessing: {pgn_file.name}")
        
        # Create temporary config for this file
        file_config = config.copy()
        file_config['data'] = config['data'].copy()
        file_config['data']['max_games'] = games_per_file
        
        # Parse sequentially (more reliable than multiprocessing for PGN)
        file_data = parse_pgn_file_sequential(pgn_file, file_config)
        
        print(f"Extracted {len(file_data)} positions from {pgn_file.name}")
        all_data.extend(file_data)
        
        # Clean up to free memory
        del file_data
        gc.collect()
    
    print(f"\nTotal positions extracted: {len(all_data)}")
    
    if len(all_data) == 0:
        print("\n" + "="*60)
        print("WARNING: No positions extracted!")
        print("Possible reasons:")
        print(f"1. All games have Elo < {min_elo}")
        print("2. PGN file format is incorrect")
        print("3. Games don't have Elo ratings in headers")
        print("\nTry:")
        print("- Lower min_elo in config.yaml (try 1000 or 1500)")
        print("- Run: python chess/scripts/diagnose_pgn.py")
        print("- Check PGN file format")
        print("="*60)
        return []
    
    # Save to cache
    print(f"Saving to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Cache saved successfully")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return all_data


class ChessDataset(Dataset):
    """PyTorch Dataset for chess positions with lazy loading"""
    
    def __init__(self, data):
        """
        Args:
            data: list of (board_tensor, move_index, outcome)
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board_tensor, move_index, outcome = self.data[idx]
        
        # Convert to tensors only when needed
        return (
            torch.FloatTensor(board_tensor),
            torch.LongTensor([move_index])[0],
            torch.FloatTensor([outcome])
        )


def create_dataloaders(data, config):
    """Create train and validation dataloaders"""
    
    if len(data) == 0:
        raise ValueError(
            "Cannot create dataloaders with 0 positions.\n"
            "Please check:\n"
            "1. Your PGN file exists and is valid\n"
            "2. min_elo in config.yaml is not too high (try 1000-1500)\n"
            "3. Run 'python chess/scripts/diagnose_pgn.py' to check PGN format"
        )
    
    # Split data
    split_idx = int(len(data) * config['data']['train_split'])
    
    # Ensure at least 1 sample in each split
    if split_idx == 0:
        split_idx = 1
    if split_idx >= len(data):
        split_idx = len(data) - 1
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train dataset: {len(train_data)} positions")
    print(f"Val dataset: {len(val_data)} positions")
    
    # Create datasets
    train_dataset = ChessDataset(train_data)
    val_dataset = ChessDataset(val_data)
    
    # Create dataloaders with optimized settings
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