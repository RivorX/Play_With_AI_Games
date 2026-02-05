"""
Chess dataset and dataloader utilities.
Split from src.data to keep dataset logic separate from preprocessing.
"""

import chess
import mmap
import random
import struct

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.data_helpers import compact_to_tensor, get_turn_from_move_idx

class BinaryChessDataset(Dataset):
    """
    Memory-mapped dataset with POV and Dynamic Sliding Window support
    
    🆕 v4.2 KEY FEATURES:
    - POV (Point of View): All boards from current player's perspective
    - Dynamic Sliding Window: History assembled at load time using mmap
    - Stride support: Sample every Nth position for faster training
    - GameID tracking: Efficient history reconstruction
    """
    
    def __init__(self, binary_file, indices, position_size, use_mtl=None, 
                 history_positions=0, stride=1, stride_mode="fullmove_per_game_offset",
                 game_length_by_id=None):
        """
        Args:
            binary_file: Path to binary file
            indices: List of position indices to use
            position_size: Size of each position in bytes
            use_mtl: Whether MTL is enabled
            history_positions: Number of history positions to include (dynamic)
            stride: Sliding window stride (1 = all positions, 2 = every other, etc.)
            game_length_by_id: Optional dict {game_id: total_moves} for value weighting
        """
        # ✅ ADDED v4.2.1: Input validation
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if history_positions < 0:
            raise ValueError(f"history_positions must be >= 0, got {history_positions}")
        if position_size < 50:  # 🔧 FIXED: Minimum size with metadata (38B board + 12B metadata)
            raise ValueError(f"position_size too small: {position_size} (minimum 50 bytes)")
        
        self.binary_file = binary_file
        self.position_size = position_size
        self.history_positions = history_positions
        self.stride = stride
        self.stride_mode = stride_mode
        self.game_length_by_id = game_length_by_id
        
        # Auto-detect MTL
        # 🔧 v4.5 FIXED: base_size must match current binary layout:
        #   Board(38) + GameID(4) + MoveIdx(2) + MoveTarget(2) + Outcome(4) = 50
        #   Board is now 38 bytes: 32B pieces + 6B metadata (castling, ep, halfmove, fullmove)
        if use_mtl is None:
            base_size = 38 + 4 + 2 + 2 + 4  # = 50 (was 48 - CRITICAL FIX)
            self.use_mtl = (position_size == base_size + 12)  # 50 + 12 = 62
        else:
            self.use_mtl = use_mtl
        
        # 🆕 Apply stride filter to indices
        if self.stride > 1:
            print(f"  🔄 Applying stride={stride} (mode={self.stride_mode})")
            # Filter indices based on stride
            # We need to check MoveIdx for each position
            filtered_indices = self._filter_indices_by_stride(indices)
            self.indices = filtered_indices
            print(f"     Original positions: {len(indices):,}")
            print(f"     After stride filter: {len(self.indices):,}")
        else:
            self.indices = indices
        
        self._mmap = None
        self._file = None
    
    def _filter_indices_by_stride(self, indices):
        """
        Filter indices based on sliding window stride
        Modes:
        - ply: move_idx % stride == 0 (legacy, can drop one color for even stride)
        - fullmove: (move_idx // 2) % stride == 0 (keeps both colors)
        - fullmove_per_game_offset: per-game offset on fullmove index (keeps both colors)
        
        🔧 FIXED: Correct offset for new binary format
        """
        filtered = []
        
        # Open file temporarily to read move indices
        with open(self.binary_file, 'rb') as f:
            for idx in indices:
                offset = idx * self.position_size
                # 🔧 v4.5 FIXED Layout: [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + ...
                f.seek(offset + 38)  # Board (38B)
                header = f.read(6)   # GameID (4B) + MoveIdx (2B)
                if len(header) < 6:
                    continue
                
                game_id = struct.unpack('I', header[:4])[0]
                move_idx = struct.unpack('H', header[4:6])[0]
                
                fullmove_idx = move_idx // 2
                
                if self.stride_mode == "ply":
                    keep = (move_idx % self.stride == 0)
                elif self.stride_mode == "fullmove":
                    keep = (fullmove_idx % self.stride == 0)
                elif self.stride_mode == "fullmove_per_game_offset":
                    offset = game_id % self.stride
                    keep = (fullmove_idx % self.stride == offset)
                else:
                    # Fallback to safe default
                    keep = (fullmove_idx % self.stride == 0)
                
                if keep:
                    filtered.append(idx)
        
        return filtered
    
    def _ensure_mmap(self):
        """Open memory-mapped file (per worker)"""
        if self._mmap is None:
            self._file = open(self.binary_file, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get item with POV and dynamic sliding window history
        
        🔧 v4.3 FIXES:
        - Binary offsets updated for GameID uint32 (4 bytes instead of 2)
        - History walk iterates over raw file positions (position_idx-1, -2, …)
          which is correct: the binary file stores positions in game order, so
          adjacent positions in the file that share the same GameID belong to the
          same game.  The stride filter only affects which positions are *returned*
          as training samples — history is still assembled from every position in
          the file, exactly as intended by the sliding-window design.
        """
        self._ensure_mmap()
        
        position_idx = self.indices[idx]
        offset = position_idx * self.position_size
        data = self._mmap[offset:offset + self.position_size]
        
        # 🔧 v4.5 FIXED Layout:
        # [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [MTL (12B)]
        compact_board = data[:38]  # 🔧 FIXED: 38 bytes (was 36)
        game_id    = struct.unpack('I', data[38:42])[0]   # 🔧 FIXED: offset +2
        move_idx   = struct.unpack('H', data[42:44])[0]   # 🔧 FIXED: offset +2
        move_target = struct.unpack('H', data[44:46])[0]  # 🔧 FIXED: offset +2
        outcome    = struct.unpack('f', data[46:50])[0]   # 🔧 FIXED: offset +2
        total_moves = None
        if self.game_length_by_id is not None:
            total_moves = self.game_length_by_id.get(game_id, 0)
        
        # 🔧 v4.5: Validation
        if len(compact_board) != 38:
            raise ValueError(f"Invalid compact board size: {len(compact_board)} (expected 38)")
        
        # Determine whose turn it is
        is_black_turn = get_turn_from_move_idx(move_idx) == chess.BLACK
        
        # Convert current board to tensor with POV
        board_tensor = compact_to_tensor(compact_board, flip_perspective=is_black_turn)
        
        # 🔧 v4.5: Validate tensor shape (should be 16 planes now)
        if board_tensor.shape[0] != 16:
            raise ValueError(f"Invalid board tensor shape: {board_tensor.shape} (expected (16, 8, 8))")
        
        # DYNAMIC SLIDING WINDOW: Build history by walking backwards through raw file.
        # We walk position_idx-1, position_idx-2, … and stop as soon as the GameID
        # changes (= different game) or we run out of file.  This correctly assembles
        # history regardless of which positions the stride filter selected as samples.
        history_tensors = []
        
        if self.history_positions > 0:
            current_offset = position_idx - 1
            collected_history = 0
            
            while collected_history < self.history_positions and current_offset >= 0:
                hist_offset = current_offset * self.position_size
                hist_data = self._mmap[hist_offset:hist_offset + self.position_size]
                
                # 🔧 v4.5 FIXED: GameID is uint32 at [38:42]
                hist_game_id = struct.unpack('I', hist_data[38:42])[0]
                
                # Stop if different game
                if hist_game_id != game_id:
                    break
                
                hist_compact_board = hist_data[:38]  # 🔧 FIXED: 38 bytes
                
                # All history boards use the CURRENT player's POV for consistency
                hist_tensor = compact_to_tensor(hist_compact_board, flip_perspective=is_black_turn)
                
                history_tensors.insert(0, hist_tensor)  # oldest first
                collected_history += 1
                current_offset -= 1
            
            # Pad with zeros if not enough history available
            while len(history_tensors) < self.history_positions:
                empty_board = np.zeros((16, 8, 8), dtype=np.float32)  # 🔧 FIXED: 16 channels!
                history_tensors.insert(0, empty_board)
        
        # Stack: [oldest_history, …, newest_history, current]  →  (16*(H+1), 8, 8)
        if history_tensors:
            all_tensors = history_tensors + [board_tensor]
            stacked_board = np.concatenate(all_tensors, axis=0)
        else:
            stacked_board = board_tensor
        
        # 🔧 CRITICAL FIX: .copy() to avoid mmap non-resizable storage issue
        # DataLoader collate requires resizable tensors
        stacked_board = stacked_board.copy()
        
        if self.use_mtl:
            # 🔧 v4.5 FIXED: MTL labels with corrected offsets
            win      = struct.unpack('f', data[50:54])[0]  # 🔧 FIXED: offset +2
            material = struct.unpack('f', data[54:58])[0]  # 🔧 FIXED: offset +2
            check    = struct.unpack('f', data[58:62])[0]  # 🔧 FIXED: offset +2
            
            sample = {
                'board':    torch.from_numpy(stacked_board),
                'move':     torch.LongTensor([move_target])[0],
                'value':    torch.FloatTensor([outcome]),
                'move_idx': torch.LongTensor([move_idx])[0],  # 🔧 v4.4 FIX: Dodano dla move-weighted BCE
                'win':      torch.FloatTensor([win]),
                'material': torch.FloatTensor([material]),
                'check':    torch.FloatTensor([check])
            }
            if total_moves is not None:
                sample['total_moves'] = torch.LongTensor([total_moves])[0]
            return sample
        else:
            # 🆕 v4.4 FIX: Dodano move_idx dla move-weighted BCE loss
            if total_moves is not None:
                return (
                    torch.from_numpy(stacked_board),
                    torch.LongTensor([move_target])[0],
                    torch.FloatTensor([outcome]),
                    torch.LongTensor([move_idx])[0],  # 🔧 For move-weighted BCE
                    torch.LongTensor([total_moves])[0]
                )
            return (
                torch.from_numpy(stacked_board),
                torch.LongTensor([move_target])[0],
                torch.FloatTensor([outcome]),
                torch.LongTensor([move_idx])[0]  # 🔧 For move-weighted BCE
            )
    
    def __del__(self):
        if self._mmap is not None:
            self._mmap.close()
        if self._file is not None:
            self._file.close()


# ==============================================================================
# DATALOADER CREATION
# ==============================================================================

def _build_game_ranges(binary_file, position_size, total_positions):
    """
    Build contiguous (game_id, start_idx, end_idx) ranges by scanning the binary file.
    Assumes positions are stored in game order and GameID is at offset 38.
    """
    ranges = []
    
    with open(binary_file, 'rb') as f:
        prev_game_id = None
        start_idx = 0
        last_idx = -1
        
        for idx in range(total_positions):
            record = f.read(position_size)
            if len(record) < position_size:
                break
            
            last_idx = idx
            game_id = struct.unpack('I', record[38:42])[0]
            
            if prev_game_id is None:
                prev_game_id = game_id
                start_idx = idx
                continue
            
            if game_id != prev_game_id:
                ranges.append((prev_game_id, start_idx, idx))
                prev_game_id = game_id
                start_idx = idx
        
        if prev_game_id is not None and last_idx >= start_idx:
            ranges.append((prev_game_id, start_idx, last_idx + 1))
    
    return ranges


def _build_game_length_map(binary_file, position_size, total_positions):
    """
    Build {game_id: total_moves} mapping using contiguous ranges.
    """
    ranges = _build_game_ranges(binary_file, position_size, total_positions)
    return {game_id: end - start for game_id, start, end in ranges}


def _split_indices_by_game(metadata, config, return_game_ranges=False):
    """
    Split dataset by GameID to prevent val leakage from shared game history.
    Returns train/val indices and game counts for logging.
    """
    import random
    
    binary_file = metadata['binary_file']
    position_size = metadata['position_size']
    total_positions = metadata['total_positions']
    
    game_ranges = _build_game_ranges(binary_file, position_size, total_positions)
    if not game_ranges:
        raise ValueError("No games found for per-game split.")
    
    rng = random.Random(config['seed'])
    rng.shuffle(game_ranges)
    
    target_train_positions = int(total_positions * config['data']['train_split'])
    
    train_ranges = []
    val_ranges = []
    train_positions = 0
    
    for _, start, end in game_ranges:
        count = end - start
        if train_positions < target_train_positions:
            train_ranges.append((start, end))
            train_positions += count
        else:
            val_ranges.append((start, end))
    
    # Ensure both splits are non-empty
    if not val_ranges and train_ranges:
        val_ranges.append(train_ranges.pop())
    if not train_ranges and val_ranges:
        train_ranges.append(val_ranges.pop())
    
    train_indices = []
    for start, end in train_ranges:
        train_indices.extend(range(start, end))
    
    val_indices = []
    for start, end in val_ranges:
        val_indices.extend(range(start, end))
    
    if return_game_ranges:
        return train_indices, val_indices, len(game_ranges), len(train_ranges), len(val_ranges), game_ranges
    return train_indices, val_indices, len(game_ranges), len(train_ranges), len(val_ranges)


def create_dataloaders(metadata, config):
    """Create dataloaders with POV, MTL, and Dynamic Sliding Window support"""
    
    if metadata['total_positions'] == 0:
        raise ValueError("Cannot create dataloaders with 0 positions.")
    
    total_positions = metadata['total_positions']
    
    split_by_game = config['data'].get('split_by_game', False)
    game_count = train_game_count = val_game_count = None
    use_game_length = config['imitation_learning'].get('value_move_weight_use_game_length', False)
    game_length_by_id = None
    
    if split_by_game:
        if use_game_length:
            (train_indices, val_indices, game_count, train_game_count,
             val_game_count, game_ranges) = _split_indices_by_game(
                metadata, config, return_game_ranges=True
            )
            game_length_by_id = {game_id: end - start for game_id, start, end in game_ranges}
        else:
            train_indices, val_indices, game_count, train_game_count, val_game_count = _split_indices_by_game(metadata, config)
    else:
        all_indices = list(range(total_positions))
        
        # 🔧 FIXED: Random shuffle before split to balance Win/Draw/Loss distribution
        # Sequential split causes validation bias (last 10% may have different outcome distribution)
        import random
        random.seed(config['seed'])
        random.shuffle(all_indices)
        
        # Split
        split_idx = int(len(all_indices) * config['data']['train_split'])
        split_idx = max(1, min(split_idx, len(all_indices) - 1))
        
        train_indices = all_indices[:split_idx]
        val_indices = all_indices[split_idx:]
        if use_game_length:
            game_length_by_id = _build_game_length_map(
                metadata['binary_file'],
                metadata['position_size'],
                total_positions
            )
    
    # Get configuration
    position_size = metadata.get('position_size')
    use_mtl = metadata.get('use_mtl', False)
    history_positions = config['model'].get('history_positions', 0)
    stride = config['data'].get('sliding_window_stride', 1)
    stride_mode = config['data'].get('stride_mode', "fullmove_per_game_offset")
    
    # 🔧 v4.5 FIXED: Calculate actual input planes (16 per position with metadata)
    input_planes = 16 * (1 + history_positions)  # 16 planes (12 pieces + 4 metadata)
    
    print(f"\n{'='*70}")
    print("📊 Dataset Configuration:")
    print(f"  • Total positions (before stride): {total_positions:,}")
    print(f"  • Position size: {position_size} bytes (expected: {50 if not use_mtl else 62})")
    print(f"  • MTL: {use_mtl}")
    print(f"  • History positions: {history_positions} (dynamic)")
    print(f"  • Sliding window stride: {stride}")
    print(f"  • Stride mode: {stride_mode}")
    print(f"  • Split by game: {split_by_game}")
    if split_by_game and game_count is not None:
        print(f"  • Games: {game_count:,} (train: {train_game_count:,}, val: {val_game_count:,})")
    print(f"  • Input planes: {input_planes} (16 × {1 + history_positions})")
    print(f"  • Chess metadata: Castling, En Passant, Halfmove, Fullmove")
    print(f"{'='*70}\n")
    
    # Create datasets
    train_dataset = BinaryChessDataset(
        metadata['binary_file'], 
        train_indices, 
        position_size=position_size,
        use_mtl=use_mtl,
        history_positions=history_positions,
        stride=stride,
        stride_mode=stride_mode,
        game_length_by_id=game_length_by_id
    )
    val_dataset = BinaryChessDataset(
        metadata['binary_file'], 
        val_indices,
        position_size=position_size,
        use_mtl=use_mtl,
        history_positions=history_positions,
        stride=stride,
        stride_mode=stride_mode,
        game_length_by_id=game_length_by_id
    )
    
    print(f"  • Train positions (after stride): {len(train_dataset):,}")
    print(f"  • Val positions (after stride): {len(val_dataset):,}")
    print(f"{'='*70}\n")
    
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
