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
import math
import pickle
from src.data import board_to_tensor, move_to_index
from src.utils.data_helpers import ACTION_SIZE
from src.mcts import MCTSNode


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
        self.verbose = bool(config.get('reinforcement_learning', {}).get('self_play_worker_verbose', False))
        
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
            
            if self.verbose and (game_idx + 1) % 10 == 0:
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
            policy_target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
            total_visits = sum(visit_counts.values())
            
            for m, visits in visit_counts.items():
                policy_target[move_to_index(m, board)] = visits / total_visits
            
            # Store position for training (with history, matches model input)
            board_tensor = torch.from_numpy(self.mcts._build_history_tensor(board))
            game_history.append((board_tensor, policy_target, board.turn))
            
            # Update history BEFORE making the move (matches play.py)
            self.mcts.update_history(board)
            
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


# =============================================================================
# TRUE BATCH MCTS (MULTI-GAME) FOR BETTER GPU UTILIZATION
# =============================================================================

class MultiGameBatchMCTS:
    """
    Batch MCTS that searches multiple games in parallel and batches
    leaf evaluations across all games for higher GPU utilization.
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.c_puct = config['reinforcement_learning']['mcts_c_puct']
        self.eval_batch_size = config['reinforcement_learning'].get('mcts_batch_size', 32)

        # History configuration (POV)
        self.history_positions = config['model'].get('history_positions', 0)

        # Tree reuse
        self.reuse_tree = config['reinforcement_learning'].get('mcts_reuse_tree', True)

        # Dirichlet noise params
        self.dirichlet_alpha = config['reinforcement_learning'].get('mcts_dirichlet_alpha', 0.3)
        self.dirichlet_weight = config['reinforcement_learning'].get('mcts_dirichlet_weight', 0.0)

        # Inference optimization (AMP on GPU)
        self.use_amp = config.get('hardware', {}).get('use_amp', False) and self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if config.get('hardware', {}).get('use_bfloat16', False) else torch.float16

    def _build_history_tensor(self, current_board, board_history):
        """
        Build tensor with history using POV-aware board_to_tensor.

        Args:
            current_board: chess.Board for current position
            board_history: list of previous boards (from real game history)
        """
        if self.history_positions == 0:
            return board_to_tensor(current_board)

        history_tensors = []
        if board_history:
            history_boards = board_history[-self.history_positions:]
            for hist_fen in history_boards:
                # 🔥 OPTIMIZATION: Reconstruct board from FEN string (saves RAM)
                hist_board = chess.Board(hist_fen) if isinstance(hist_fen, str) else hist_fen
                hist_tensor = board_to_tensor(
                    hist_board,
                    flip_perspective=(current_board.turn == chess.BLACK)
                )
                history_tensors.append(hist_tensor)

        while len(history_tensors) < self.history_positions:
            empty_tensor = np.zeros((16, 8, 8), dtype=np.float32)
            history_tensors.insert(0, empty_tensor)

        current_tensor = board_to_tensor(current_board)
        history_tensors.append(current_tensor)

        return np.concatenate(history_tensors, axis=0)

    def _select_child(self, node):
        """Select child with highest UCB score (optimized)"""
        best_score = -float('inf')
        best_child = None

        # Precalculate parent term to avoid doing it for every child
        parent_sqrt = math.sqrt(node.visit_count + node.virtual_loss + 1)
        c_puct = self.c_puct

        for _, child in node.children.items():
            # Inline value() and ucb_score() for speed
            cv = child.visit_count + child.virtual_loss
            if cv == 0:
                q_value = 0.0
            else:
                q_value = (child.value_sum - child.virtual_loss) / cv
                
            u_value = c_puct * child.prior * parent_sqrt / (1 + cv)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backpropagate(self, search_path, value):
        """Backpropagate value"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def search_many(self, game_states, num_simulations):
        """
        Run MCTS for multiple games and batch leaf evaluations across games.

        Args:
            game_states: list of dicts with keys: board, root, board_history
            num_simulations: simulations per game
        Returns:
            List[Dict[chess.Move, int]] visit counts for each game in order
        """
        if not game_states:
            return []

        # Initialize / reuse roots per game
        for gs in game_states:
            board = gs['board']
            root = gs.get('root', None)

            if self.reuse_tree and root is not None:
                # 🔥 OPTIMIZATION: Use cached FEN
                target_fen = board.fen()
                for _, child in root.children.items():
                    if child.get_fen() == target_fen:
                        _ = child.board  # Ensure board is instantiated
                        root = child
                        root.parent = None
                        break
                else:
                    root = MCTSNode(board)
            else:
                root = MCTSNode(board)

            gs['root'] = root
            gs['_noise_added'] = False

        remaining = [num_simulations] * len(game_states)
        total_remaining = num_simulations * len(game_states)
        game_ptr = 0

        while total_remaining > 0:
            # Cap per-game quota per round so that backprop happens before all
            # simulations of a single game are consumed  (fixes: when
            # eval_batch_size >= num_simulations, all sims land in one batch,
            # root is never traversed after expansion, children stay at 0 visits).
            max_remaining_any_game = max(remaining) if remaining else 1
            slots_per_game = max(1, min(
                self.eval_batch_size // max(1, len(game_states)),
                max_remaining_any_game // 4,
            ))
            batch_size = min(self.eval_batch_size, total_remaining,
                             slots_per_game * len(game_states))
            leaf_nodes = []
            search_paths = []
            leaf_game_indices = []

            for _ in range(batch_size):
                # Find next game with remaining sims
                found = False
                for _ in range(len(game_states)):
                    if remaining[game_ptr] > 0:
                        found = True
                        break
                    game_ptr = (game_ptr + 1) % len(game_states)

                if not found:
                    break

                gs_idx = game_ptr
                gs = game_states[gs_idx]

                node = gs['root']
                search_path = [node]
                node.add_virtual_loss()

                while not node.is_leaf() and not node.is_game_over:
                    node = self._select_child(node)
                    node.add_virtual_loss()
                    search_path.append(node)

                search_paths.append(search_path)
                leaf_nodes.append(node)
                leaf_game_indices.append(gs_idx)

                remaining[gs_idx] -= 1
                total_remaining -= 1
                game_ptr = (game_ptr + 1) % len(game_states)

            if not leaf_nodes:
                break

            values = self._batch_expand_and_evaluate(leaf_nodes, leaf_game_indices, game_states)

            for search_path, value in zip(search_paths, values):
                self._backpropagate(search_path, value)
                for node in search_path:
                    node.remove_virtual_loss()

        return [
            {move: child.visit_count for move, child in gs['root'].children.items()}
            for gs in game_states
        ]

    def _batch_expand_and_evaluate(self, nodes, game_indices, game_states):
        """
        Batch expansion + evaluation for leaf nodes across games.
        """
        # The same leaf can appear multiple times in one batch.
        # Evaluate each unique node once and fan-out value to duplicates.
        node_occurrences = {}
        unique_entries = []
        for idx, node in enumerate(nodes):
            node_id = id(node)
            if node_id not in node_occurrences:
                node_occurrences[node_id] = [idx]
                unique_entries.append((node, game_indices[idx]))
            else:
                node_occurrences[node_id].append(idx)

        terminal_values = {}
        non_terminal_nodes = []
        non_terminal_game_indices = []

        for node, gi in unique_entries:
            if node.is_game_over:
                result = node.board.result()
                if result == '1-0':
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif result == '0-1':
                    value = -1.0 if node.board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0
                terminal_values[id(node)] = value
            else:
                non_terminal_nodes.append(node)
                non_terminal_game_indices.append(gi)

        values_by_node_id = {}

        if non_terminal_nodes:
            board_tensors = torch.stack([
                torch.from_numpy(
                    self._build_history_tensor(
                        node.board,
                        game_states[gi]['board_history']
                    )
                )
                for node, gi in zip(non_terminal_nodes, non_terminal_game_indices)
            ]).to(self.device, memory_format=torch.channels_last, non_blocking=True)

            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                        policy_logits_batch, values_batch = self.model(board_tensors, return_aux=False)
                else:
                    policy_logits_batch, values_batch = self.model(board_tensors, return_aux=False)
                
                # 🔥 OPTIMIZATION: Transfer entire batch to CPU at once, not row by row
                if values_batch.dim() == 2 and values_batch.shape[1] == 3:
                    # WDL output: compute softmax on GPU before transfer
                    wdl_probs = torch.softmax(values_batch, dim=1)
                    values_batch = (wdl_probs[:, 0] - wdl_probs[:, 2])
                
                # Convert to float32 before numpy() because numpy doesn't support bfloat16
                policy_logits_batch = policy_logits_batch.float().cpu().numpy()
                values_batch = values_batch.float().cpu().numpy()

            for idx, node in enumerate(non_terminal_nodes):
                policy_logits = policy_logits_batch[idx]
                value = float(values_batch[idx])

                legal_moves = list(node.board.legal_moves)
                legal_indices = [move_to_index(m, node.board) for m in legal_moves]

                if legal_moves:
                    legal_logits = policy_logits[legal_indices]
                    legal_probs = np.exp(legal_logits - legal_logits.max())
                    legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
                else:
                    legal_probs = np.array([])

                gi = non_terminal_game_indices[idx]
                add_noise = (
                    self.dirichlet_weight > 0 and
                    node is game_states[gi]['root'] and
                    not game_states[gi].get('_noise_added', False)
                )

                if add_noise and len(legal_moves) > 0:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
                    legal_probs = (1 - self.dirichlet_weight) * legal_probs + self.dirichlet_weight * noise
                    game_states[gi]['_noise_added'] = True

                # Expand only if not already expanded (avoid overwriting priors in same batch)
                if not node.expanded:
                    for move, prior in zip(legal_moves, legal_probs):
                        node.children[move] = MCTSNode(
                            board=None,
                            parent=node,
                            move=move,
                            prior=prior,
                            copy_board=False,
                        )
                    node.expanded = True

                values_by_node_id[id(node)] = value

        all_values = [0.0] * len(nodes)
        for node_id, indices in node_occurrences.items():
            value = values_by_node_id.get(node_id, terminal_values.get(node_id, 0.0))
            for idx in indices:
                all_values[idx] = value

        return all_values


class BatchSelfPlayMCTSBatch:
    """
    True batch self-play: multiple games in parallel, shared GPU eval batches.
    """

    def __init__(self, model, config, device, max_batch_games_per_worker):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()

        # Use multi-game MCTS
        self.mcts = MultiGameBatchMCTS(model, config, device)

        if device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)

        self.num_simulations = config['reinforcement_learning']['mcts_simulations']
        self.temp_threshold = config['reinforcement_learning']['mcts_temperature_threshold']
        self.temperature = config['reinforcement_learning'].get('mcts_temperature', 1.0)

        self.max_batch_games_per_worker = max(1, int(max_batch_games_per_worker))
        self._progress_file = None  # Set externally to enable progress reporting
        self._games_completed = 0

    def play_games(self, num_games):
        all_positions = []
        game_lengths = []
        self._games_completed = 0

        games_left = num_games
        batch_idx = 0

        while games_left > 0:
            batch_size = min(self.max_batch_games_per_worker, games_left)
            batch_idx += 1

            positions, lengths = self._play_batch(batch_size)
            all_positions.extend(positions)
            game_lengths.extend(lengths)
            self._games_completed += len(lengths)

            # Report progress to file if requested
            if self._progress_file is not None:
                try:
                    with open(self._progress_file, 'w') as _pf:
                        _pf.write(str(self._games_completed))
                except Exception:
                    pass

            games_left -= batch_size

        return all_positions, game_lengths

    def _play_batch(self, batch_size):
        max_moves = 200

        game_states = []
        for _ in range(batch_size):
            game_states.append({
                'board': chess.Board(),
                'board_history': [],
                'root': None,
                'game_history': [],
                'move_count': 0,
                'done': False
            })

        while True:
            active_indices = [i for i, gs in enumerate(game_states) if not gs['done']]
            if not active_indices:
                break

            active_states = [game_states[i] for i in active_indices]

            visit_counts_list = self.mcts.search_many(
                active_states,
                num_simulations=self.num_simulations
            )

            for idx, visit_counts in zip(active_indices, visit_counts_list):
                gs = game_states[idx]
                board = gs['board']

                if not visit_counts:
                    gs['done'] = True
                    continue

                temperature = self.temperature if gs['move_count'] < self.temp_threshold else 0.01
                move = self._select_move_from_visits(visit_counts, temperature)

                policy_target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
                total_visits = sum(visit_counts.values())

                if total_visits > 0:
                    for m, visits in visit_counts.items():
                        policy_target[move_to_index(m, board)] = visits / total_visits
                else:
                    # Fallback: uniform over legal moves
                    n = len(visit_counts)
                    for m in visit_counts:
                        policy_target[move_to_index(m, board)] = 1.0 / n

                board_tensor = torch.from_numpy(
                    self.mcts._build_history_tensor(board, gs['board_history'])
                )
                gs['game_history'].append((board_tensor, policy_target, board.turn))

                # Update history BEFORE making the move
                # 🔥 OPTIMIZATION: Store FEN instead of full board copy to save RAM
                gs['board_history'].append(board.fen())
                max_history = self.mcts.history_positions + 10
                if len(gs['board_history']) > max_history:
                    gs['board_history'] = gs['board_history'][-max_history:]

                board.push(move)
                gs['move_count'] += 1

                if board.is_game_over() or gs['move_count'] >= max_moves:
                    gs['done'] = True
                    # Free up memory immediately
                    gs['root'] = None

        positions = []
        game_lengths = []

        for gs in game_states:
            board = gs['board']
            result = board.result()
            if result == '1-0':
                outcome = 1.0
            elif result == '0-1':
                outcome = -1.0
            else:
                outcome = 0.0

            history_len = len(gs['game_history'])

            for board_tensor, policy_target, turn in gs['game_history']:
                value = outcome if turn == chess.WHITE else -outcome
                positions.append((
                    board_tensor,
                    policy_target,
                    torch.tensor([value], dtype=torch.float32)
                ))

            game_lengths.append(history_len)

        return positions, game_lengths

    def _select_move_from_visits(self, visit_counts, temperature):
        moves = list(visit_counts.keys())
        visits = np.array([visit_counts[m] for m in moves], dtype=np.float64)

        if temperature == 0 or len(moves) == 1:
            best_idx = np.argmax(visits)
            return moves[best_idx]
        else:
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            if total <= 0 or not np.isfinite(total):
                # Fallback: all visits are zero or overflow — pick uniformly
                return moves[np.random.randint(len(moves))]
            probs = visits_temp / total
            # Renormalize to fix any floating-point drift
            probs = np.clip(probs, 0.0, 1.0)
            probs /= probs.sum()
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
        rl_cfg = config.get('reinforcement_learning', {})
        worker_verbose = bool(rl_cfg.get('self_play_worker_verbose', False))

        def _wlog(message):
            if worker_verbose:
                print(f"Worker {rank}: {message}")
        # Silence model summary in workers
        config['model'] = {**config.get('model', {}), 'print_summary': False}
        
        # Setup device
        if device_id == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
        
        # Limit CPU threads per worker to avoid oversubscription
        if device.type == 'cpu':
            self_play_threads = rl_cfg.get('self_play_torch_threads', None)
            if self_play_threads is not None:
                try:
                    self_play_threads = int(self_play_threads)
                    if self_play_threads > 0:
                        torch.set_num_threads(self_play_threads)
                        try:
                            torch.set_num_interop_threads(max(1, min(4, self_play_threads)))
                        except Exception:
                            pass
                except Exception:
                    pass
        
        _wlog(f"Starting on {device}")
        
        # Load model
        model = ChessNet(config).to(device)
        if device.type == 'cuda':
            model = model.to(memory_format=torch.channels_last)
        
        # CoordConv buffers (coord_x/coord_y) can cause overlap copy errors on load.
        # They are deterministic, so we safely skip them.
        if model_state:
            model_state = {
                k: v for k, v in model_state.items()
                if not (k.endswith('coord_x') or k.endswith('coord_y'))
            }
        incompatible = model.load_state_dict(model_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            extra_missing = [k for k in incompatible.missing_keys if not (k.endswith('coord_x') or k.endswith('coord_y'))]
            if extra_missing or incompatible.unexpected_keys:
                print(f"⚠️ Worker {rank}: state_dict mismatch. Missing={extra_missing}, Unexpected={incompatible.unexpected_keys}")
        model.eval()
        
        # Create self-play engine with MCTS
        use_batch = rl_cfg.get('use_batch_selfplay', False)
        max_batch_games = rl_cfg.get('max_batch_games_per_worker', 256)
        if use_batch:
            engine = BatchSelfPlayMCTSBatch(model, config, device, max_batch_games)
            actual_max = min(max_batch_games, num_games)
            _wlog(f"Batch self-play enabled (max {max_batch_games}, actual {actual_max})")
        else:
            engine = BatchSelfPlayMCTS(model, config, device)

        # Progress reporting: worker writes number of completed games to a .progress file
        # so the main process can track live progress via tqdm.
        progress_file_path = result_file_path.replace('.pkl', '.progress')

        def _write_progress(n):
            try:
                with open(progress_file_path, 'w') as _pf:
                    _pf.write(str(n))
            except Exception:
                pass

        if hasattr(engine, '_progress_file'):
            engine._progress_file = progress_file_path

        _wlog(
            f"Playing {num_games} games with MCTS "
            f"({config['reinforcement_learning']['mcts_simulations']} sims/move)"
        )

        save_every = rl_cfg.get('self_play_save_every_games_resolved', None)
        if save_every is None:
            save_mult = rl_cfg.get('self_play_save_every_games', 0)
            try:
                save_mult = float(save_mult)
            except Exception:
                save_mult = 0
            if save_mult and save_mult > 0:
                games_per_iter = rl_cfg.get('games_per_iteration', 0)
                try:
                    games_per_iter = int(games_per_iter)
                except Exception:
                    games_per_iter = 0
                save_every = max(1, int(round(games_per_iter * save_mult)))
            else:
                save_every = 0

        total_positions = 0
        total_games = 0

        if save_every and save_every > 0:
            # Truncate file first
            with open(result_file_path, 'wb') as f:
                pass

            games_left = num_games
            while games_left > 0:
                chunk_games = min(save_every, games_left)
                # Temporarily disable engine's own progress file to avoid double-counting;
                # we'll update progress manually after each chunk instead.
                if hasattr(engine, '_progress_file'):
                    engine._progress_file = None
                positions, game_lengths = engine.play_games(chunk_games)

                with open(result_file_path, 'ab') as f:
                    pickle.dump((positions, game_lengths), f)

                total_positions += len(positions)
                total_games += len(game_lengths)
                _write_progress(total_games)
                games_left -= chunk_games
        else:
            # Play games in one shot — engine writes progress itself
            if hasattr(engine, '_progress_file'):
                engine._progress_file = progress_file_path
            positions, game_lengths = engine.play_games(num_games)
            total_positions = len(positions)
            total_games = len(game_lengths)
            _write_progress(total_games)

            # Save to file (avoids shared memory issues on Windows)
            with open(result_file_path, 'wb') as f:
                pickle.dump((positions, game_lengths), f)

        _wlog(f"Generated {total_positions} positions from {total_games} games")
        _wlog(f"Saved to {result_file_path}")
    
    except KeyboardInterrupt:
        # Avoid noisy tracebacks when user interrupts training.
        try:
            with open(result_file_path, 'wb') as f:
                pickle.dump(([], []), f)
        finally:
            raise SystemExit(130)

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
        from src.utils.data_helpers import ACTION_SIZE
        
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
                
                legal_probs = np.array([policy[move_to_index(m, boards[game_idx])] for m in legal_moves])
                
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
                policy_target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
                policy_target[move_to_index(move, boards[game_idx])] = 1.0
                
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
