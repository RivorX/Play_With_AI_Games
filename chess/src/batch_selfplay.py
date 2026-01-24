"""
🎯 BATCH SELF-PLAY WITH FULL MCTS - AlphaZero Style
Plays multiple games using MCTS for move selection

✅ CORRECT IMPLEMENTATION:
- Uses MCTS for all move selections (not raw network)
- Training targets = MCTS visit distributions
- High-quality training data
"""

import torch
import chess
import numpy as np
import pickle
from src.data import board_to_tensor, move_to_index


class BatchSelfPlayMCTS:
    """
    ✅ PROPER Self-Play with MCTS (AlphaZero approach)
    
    Each move is selected using MCTS search, not raw network policy.
    This generates much higher quality training data.
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        
        # Import here to avoid circular dependency
        from src.mcts import BatchMCTS
        
        # 🎯 CREATE MCTS ENGINE
        self.mcts = BatchMCTS(model, config, device)
        
        # Ensure model uses channels_last for GPU optimization
        if device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # MCTS parameters
        self.num_simulations = config['reinforcement_learning']['mcts_simulations']
        self.temp_threshold = config['reinforcement_learning']['mcts_temperature_threshold']
        self.temperature = config['reinforcement_learning'].get('mcts_temperature', 1.0)
        
        # 🆕 Temperature schedule support
        self.use_temp_schedule = config['reinforcement_learning'].get('use_temperature_schedule', False)
        if self.use_temp_schedule:
            self.temp_start = config['reinforcement_learning'].get('temperature_start', 1.5)
            self.temp_end = config['reinforcement_learning'].get('temperature_end', 0.5)
    
    def play_games(self, num_games):
        """
        ✅ Play games using MCTS for move selection
        
        Args:
            num_games: Number of games to play
        
        Returns:
            all_positions: List of (board_tensor, policy_target, value) tuples
            game_lengths: List of game lengths
        """
        all_positions = []
        game_lengths = []
        
        for game_idx in range(num_games):
            positions, game_length = self._play_single_game()
            all_positions.extend(positions)
            game_lengths.append(game_length)
            
            if (game_idx + 1) % 10 == 0:
                print(f"  Completed {game_idx + 1}/{num_games} games (avg length: {np.mean(game_lengths[-10:]):.1f} moves)")
        
        return all_positions, game_lengths
    
    def _play_single_game(self):
        """
        Play one game using MCTS for move selection
        
        Returns:
            positions: List of training positions from this game
            game_length: Number of moves in the game
        """
        board = chess.Board()
        game_history = []
        move_count = 0
        max_moves = 200
        
        while not board.is_game_over() and move_count < max_moves:
            # 🎯 USE MCTS TO SELECT MOVE
            visit_counts = self.mcts.search(
                board, 
                num_simulations=self.num_simulations
            )
            
            # Temperature-based move selection
            # High temperature early = more exploration
            # Low temperature later = more exploitation
            temperature = self.temperature if move_count < self.temp_threshold else 0.01
            move = self._select_move_from_visits(visit_counts, temperature)
            
            # 🎯 TRAINING TARGET = MCTS VISIT DISTRIBUTION (not raw network policy!)
            policy_target = torch.zeros(4096, dtype=torch.float32)
            total_visits = sum(visit_counts.values())
            
            for m, visits in visit_counts.items():
                policy_target[move_to_index(m)] = visits / total_visits
            
            # Store position for training
            board_tensor = torch.from_numpy(board_to_tensor(board))
            game_history.append((board_tensor, policy_target, board.turn))
            
            # Make the move
            board.push(move)
            move_count += 1
        
        # Reset MCTS tree for next game (important for memory!)
        self.mcts.reset_tree()
        
        # Compute game outcome for value targets
        result = board.result()
        if result == '1-0':
            outcome = 1.0
        elif result == '0-1':
            outcome = -1.0
        else:
            outcome = 0.0
        
        # Create final training positions with outcomes
        positions = []
        for board_tensor, policy_target, turn in game_history:
            # Value from perspective of player to move
            value = outcome if turn == chess.WHITE else -outcome
            positions.append((
                board_tensor,
                policy_target,
                torch.tensor([value], dtype=torch.float32)
            ))
        
        return positions, len(game_history)
    
    def _select_move_from_visits(self, visit_counts, temperature):
        """
        Select move based on MCTS visit counts with temperature
        
        Args:
            visit_counts: Dict[chess.Move, int] - visit count for each move
            temperature: float - exploration parameter
        
        Returns:
            chess.Move - selected move
        """
        moves = list(visit_counts.keys())
        visits = np.array([visit_counts[m] for m in moves])
        
        if temperature == 0 or len(moves) == 1:
            # Deterministic: choose most visited
            best_idx = np.argmax(visits)
            return moves[best_idx]
        else:
            # Stochastic: sample proportional to visits^(1/temp)
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]


# ============================================================================
# PARALLEL WORKER FOR MULTIPROCESSING
# ============================================================================

def play_games_mcts_worker(rank, model_state, config, device_id, num_games, result_file_path):
    """
    🚀 Worker function for parallel MCTS self-play
    
    Each worker:
    1. Loads model
    2. Creates its own MCTS engine
    3. Plays games
    4. Saves results to file
    
    Args:
        rank: Worker ID
        model_state: Model state dict
        config: Config dict
        device_id: GPU device ID (or 'cpu')
        num_games: Number of games to play
        result_file_path: Path to save results
    """
    try:
        from src.model import ChessNet
        
        # Setup device
        if device_id == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        
        print(f"Worker {rank}: Starting on {device}")
        
        # Load model
        model = ChessNet(config).to(device)
        if device.type == 'cuda':
            model = model.to(memory_format=torch.channels_last)
        
        model.load_state_dict(model_state)
        model.eval()
        
        # Create self-play engine with MCTS
        engine = BatchSelfPlayMCTS(model, config, device)
        
        print(f"Worker {rank}: Playing {num_games} games with MCTS ({config['reinforcement_learning']['mcts_simulations']} sims/move)")
        
        # Play games
        positions, game_lengths = engine.play_games(num_games)
        
        # Save to file (avoids shared memory issues on Windows)
        with open(result_file_path, 'wb') as f:
            pickle.dump((positions, game_lengths), f)
        
        print(f"Worker {rank}: ✅ Generated {len(positions)} positions from {len(game_lengths)} games")
        print(f"Worker {rank}: Saved to {result_file_path}")
    
    except Exception as e:
        print(f"❌ Worker {rank} failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save empty result to avoid blocking
        with open(result_file_path, 'wb') as f:
            pickle.dump(([], []), f)


# ============================================================================
# BACKWARDS COMPATIBILITY WRAPPER
# ============================================================================

def play_games_batch_worker_safe(rank, model_state, config, device_id, num_games, result_file_path):
    """
    ✅ Backwards compatible wrapper that uses PROPER MCTS
    
    This replaces the old fast-but-wrong version
    """
    return play_games_mcts_worker(rank, model_state, config, device_id, num_games, result_file_path)


# ============================================================================
# LEGACY FAST MODE (NOT RECOMMENDED - kept for comparison only)
# ============================================================================

class BatchSelfPlayFast:
    """
    ⚠️ FAST BUT LOW QUALITY: Direct network policy (no MCTS)
    
    This is the OLD implementation - kept only for speed comparison.
    NOT RECOMMENDED for actual training!
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        
        if device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
        
        print("⚠️ WARNING: Using FAST mode (no MCTS) - training quality will be lower!")
    
    def play_games(self, num_games):
        """
        ⚠️ Play games using ONLY network policy (no MCTS)
        
        Fast but generates lower quality training data!
        """
        from src.data import board_to_tensor, move_to_index
        
        boards = [chess.Board() for _ in range(num_games)]
        game_histories = [[] for _ in range(num_games)]
        move_counts = [0] * num_games
        
        max_moves = 200
        temp_threshold = self.config['reinforcement_learning']['mcts_temperature_threshold']
        
        # Pre-allocate tensor buffer
        max_batch_size = num_games
        board_buffer = torch.zeros(
            (max_batch_size, 12, 8, 8), 
            dtype=torch.float32,
            device='cpu',
            pin_memory=(self.device.type == 'cuda')
        )
        
        while True:
            active = [
                i for i in range(num_games)
                if not boards[i].is_game_over() and move_counts[i] < max_moves
            ]
            
            if not active:
                break
            
            batch_size = len(active)
            legal_moves_list = []
            
            for idx, game_idx in enumerate(active):
                tensor = board_to_tensor(boards[game_idx])
                board_buffer[idx].copy_(torch.from_numpy(tensor))
                legal_moves_list.append(list(boards[game_idx].legal_moves))
            
            batch_tensors = board_buffer[:batch_size].to(
                self.device, 
                memory_format=torch.channels_last,
                non_blocking=True
            )
            
            with torch.no_grad():
                policy_logits, values = self.model(batch_tensors)
                policies = torch.exp(policy_logits).cpu().numpy()
            
            for idx, game_idx in enumerate(active):
                policy = policies[idx]
                legal_moves = legal_moves_list[idx]
                
                if not legal_moves:
                    continue
                
                legal_probs = np.array([policy[move_to_index(m)] for m in legal_moves])
                
                if legal_probs.sum() > 1e-10:
                    legal_probs = legal_probs / legal_probs.sum()
                else:
                    legal_probs = np.ones(len(legal_moves)) / len(legal_moves)
                
                legal_probs = legal_probs / legal_probs.sum()
                
                use_temp = move_counts[game_idx] < temp_threshold
                if use_temp and np.random.rand() < 0.5:
                    if np.abs(legal_probs.sum() - 1.0) < 0.01:
                        move_idx = np.random.choice(len(legal_moves), p=legal_probs)
                    else:
                        move_idx = np.argmax(legal_probs)
                else:
                    move_idx = np.argmax(legal_probs)
                
                move = legal_moves[move_idx]
                
                board_tensor = torch.from_numpy(board_to_tensor(boards[game_idx]))
                policy_target = torch.zeros(4096, dtype=torch.float32)
                policy_target[move_to_index(move)] = 1.0
                
                game_histories[game_idx].append(
                    (board_tensor, policy_target, boards[game_idx].turn)
                )
                
                boards[game_idx].push(move)
                move_counts[game_idx] += 1
        
        all_positions = []
        game_lengths = []
        
        for game_idx in range(num_games):
            result = boards[game_idx].result()
            
            if result == '1-0':
                outcome = 1.0
            elif result == '0-1':
                outcome = -1.0
            else:
                outcome = 0.0
            
            history_len = len(game_histories[game_idx])
            
            for board_tensor, policy_target, turn in game_histories[game_idx]:
                value = outcome if turn == chess.WHITE else -outcome
                all_positions.append((
                    board_tensor,
                    policy_target,
                    torch.tensor([value], dtype=torch.float32)
                ))
            
            game_lengths.append(history_len)
        
        return all_positions, game_lengths


# For backwards compatibility
BatchSelfPlay = BatchSelfPlayMCTS  # Use MCTS version by default