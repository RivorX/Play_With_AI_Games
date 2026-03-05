"""
MCTS (Monte Carlo Tree Search) - v4.2
🆕 UPDATED: Compatible with POV + Dynamic Sliding Window
- 🎯 POV: Automatic perspective handling
- 🔄 History: Proper history tracking and assembly
- 🚀 Batch evaluation with history support
"""

import chess
import numpy as np
import math
from src.utils.data_helpers import board_to_tensor, move_to_index
import torch


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, board=None, parent=None, move=None, prior=0, copy_board=True):
        # In expansion we already pass a copied board, so allow skipping extra copy.
        if board is not None:
            self._board = board.copy() if copy_board else board
        else:
            self._board = None
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False
        
        # 🚀 Virtual Loss (no lock - single-threaded per worker)
        self.virtual_loss = 0
        
        # 🔥 Cache FEN for tree reuse speedup
        self._fen_cache = None
        
        # 🔥 Cache game over status
        self._is_game_over = None
        # Cache current board tensor (side-to-move POV) reused across evaluations.
        self._board_tensor = None
    
    @property
    def board(self):
        """Lazy evaluation of board state to save CPU and RAM"""
        if self._board is None:
            self._board = self.parent.board.copy()
            self._board.push(self.move)
        return self._board
        
    @property
    def is_game_over(self):
        """Cached game over check"""
        if self._is_game_over is None:
            self._is_game_over = self.board.is_game_over()
        return self._is_game_over
    
    def value(self):
        """Average value with virtual loss"""
        if self.visit_count == 0:
            return 0
        return (self.value_sum - self.virtual_loss) / (self.visit_count + self.virtual_loss)
    
    def is_leaf(self):
        return not self.expanded
    
    def add_virtual_loss(self, n=1):
        """Add virtual loss (no lock needed - single-threaded)"""
        self.virtual_loss += n
    
    def remove_virtual_loss(self, n=1):
        """Remove virtual loss after backup"""
        self.virtual_loss -= n
    
    def get_fen(self):
        """Get cached FEN (lazy evaluation)"""
        if self._fen_cache is None:
            self._fen_cache = self.board.fen()
        return self._fen_cache


class BatchMCTS:
    """
    🚀 Optimized Batch MCTS with:
    - Tree reuse between moves
    - Virtual loss for parallel search
    - Efficient batch evaluation
    - 🆕 v4.2: History support with POV
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.c_puct = config['reinforcement_learning']['mcts_c_puct']
        self.eval_batch_size = config['reinforcement_learning'].get('mcts_batch_size', 32)
        
        # 🆕 v4.2: History configuration
        self.history_positions = config['model'].get('history_positions', 0)
        
        # 🚀 Tree reuse
        self.root = None
        self.reuse_tree = config['reinforcement_learning'].get('mcts_reuse_tree', True)
        
        # 🆕 v4.2: Board history tracking
        # Store list of boards leading to current position
        self.board_history = []
        
        # 🔥 Inference optimization (AMP on GPU)
        self.use_amp = config.get('hardware', {}).get('use_amp', False) and self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if config.get('hardware', {}).get('use_bfloat16', False) else torch.float16
        self.dirichlet_alpha = config['reinforcement_learning'].get('mcts_dirichlet_alpha', 0.3)
        self.dirichlet_weight = config['reinforcement_learning'].get('mcts_dirichlet_weight', 0.0)

        # Reused immutable zero-history plane to reduce per-call allocations.
        self._empty_history_tensor = np.zeros((16, 8, 8), dtype=np.float32)

    @staticmethod
    def _is_encoded_history_entry(entry):
        return (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], np.ndarray)
            and isinstance(entry[1], np.ndarray)
        )

    def _encode_history_entry(self, board_or_fen):
        """
        Convert history item to cached tensors for both POVs:
        (white_pov_tensor, black_pov_tensor).
        """
        if self._is_encoded_history_entry(board_or_fen):
            return board_or_fen
        if isinstance(board_or_fen, str):
            board_obj = chess.Board(board_or_fen)
        else:
            board_obj = board_or_fen
        if not isinstance(board_obj, chess.Board):
            raise TypeError(f"Unsupported history entry type: {type(board_or_fen)}")

        return (
            board_to_tensor(board_obj, flip_perspective=False),
            board_to_tensor(board_obj, flip_perspective=True),
        )
    
    def _build_history_tensor(self, current_board, current_tensor=None):
        """
        🆕 v4.2: Build tensor with history using POV-aware board_to_tensor
        
        Similar to play.py, but uses self.board_history
        
        Args:
            current_board: chess.Board for current position
        
        Returns:
            numpy array: (input_planes, 8, 8) tensor
        """
        if self.history_positions == 0:
            # No history - just current board
            return board_to_tensor(current_board)
        
        # Build history list
        history_tensors = []
        use_black_pov = current_board.turn == chess.BLACK
        
        # Get last N boards from history
        if self.board_history:
            history_boards = self.board_history[-self.history_positions:]
            # Convert history boards to tensors with POV
            for hist_entry in history_boards:
                encoded = self._encode_history_entry(hist_entry)
                hist_tensors = encoded[1] if use_black_pov else encoded[0]
                history_tensors.append(hist_tensors)
        
        # Pad with ZEROS if not enough history (matching training data!)
        pad_count = max(0, self.history_positions - len(history_tensors))
        if pad_count:
            # 🔧 v4.5 FIX: Use zeros, not chess.Board() - matches BinaryChessDataset padding
            history_tensors = [self._empty_history_tensor] * pad_count + history_tensors
        
        # Add current board
        if current_tensor is None:
            current_tensor = board_to_tensor(current_board)
        history_tensors.append(current_tensor)
        
        # Stack: [oldest_history, ..., newest_history, current]
        # 🔧 v4.5 FIX: Shape is now (16 * (history_positions + 1), 8, 8) - 16 planes per position
        return np.concatenate(history_tensors, axis=0)

    @staticmethod
    def _current_tensor_for_node(node):
        cached = getattr(node, "_board_tensor", None)
        if cached is None:
            cached = board_to_tensor(node.board)
            node._board_tensor = cached
        return cached

    def advance_root(self, move):
        """
        Advance tree root to a known played move.

        This avoids expensive FEN matching on the next search call when the move
        is known by the caller (e.g. self-play/game loop).
        """
        if self.root is None:
            return
        child = self.root.children.get(move)
        if child is None:
            self.root = None
            return
        _ = child.board  # Materialize board once, then detach subtree.
        child.parent = None
        self.root = child
    
    def search(self, board, num_simulations, temperature=1.0):
        """
        Run MCTS with tree reuse and batch evaluation
        🆕 v4.2: Now with history support
        """
        # 🚀 Tree reuse: if root exists and matches board, reuse it
        if self.reuse_tree and self.root is not None:
            # 🔥 OPTIMIZATION: Use cached FEN instead of generating twice
            target_fen = board.fen()
            if self.root.get_fen() == target_fen:
                # Root already at requested position.
                pass
            else:
                # Try to find current position in existing tree
                for _, child in self.root.children.items():
                    if child.get_fen() == target_fen:
                        # Found it! Reuse this subtree
                        # Ensure board is instantiated before detaching parent
                        _ = child.board
                        self.root = child
                        self.root.parent = None  # Make it new root
                        break
                else:
                    # Position not found, create new tree
                    self.root = MCTSNode(board)
        else:
            self.root = MCTSNode(board)
        
        # Add Dirichlet noise to root
        add_noise = self.dirichlet_weight > 0
        
        # Batch simulations
        for batch_start in range(0, num_simulations, self.eval_batch_size):
            batch_size = min(self.eval_batch_size, num_simulations - batch_start)
            
            search_paths = []
            leaf_nodes = []
            
            for _ in range(batch_size):
                node = self.root
                search_path = [node]
                
                # 🚀 Add virtual loss during traversal
                node.add_virtual_loss()
                
                # Selection
                while not node.is_leaf():
                    node = self._select_child(node)
                    node.add_virtual_loss()
                    search_path.append(node)
                
                search_paths.append(search_path)
                leaf_nodes.append(node)
            
            # Batch evaluation
            values = self._batch_expand_and_evaluate(
                leaf_nodes,
                add_noise=(leaf_nodes[0] == self.root and add_noise)
            )
            
            # Backpropagation + remove virtual loss
            for search_path, value in zip(search_paths, values):
                self._backpropagate(search_path, value)
                # Remove virtual loss
                for node in search_path:
                    node.remove_virtual_loss()
        
        return {move: child.visit_count for move, child in self.root.children.items()}
    
    def _select_child(self, node):
        """Select child with highest UCB score (optimized)"""
        best_score = -float('inf')
        best_child = None
        
        # Precalculate parent term to avoid doing it for every child
        parent_sqrt = math.sqrt(node.visit_count + node.virtual_loss + 1)
        c_puct = self.c_puct
        
        for move, child in node.children.items():
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
    
    def _batch_expand_and_evaluate(self, nodes, add_noise=False):
        """
        🆕 v4.2: Batch expansion and evaluation with history support
        """
        # The same leaf can be selected multiple times in one search batch.
        # Evaluate each unique node once and reuse its value for duplicates.
        node_occurrences = {}
        unique_nodes = []
        for idx, node in enumerate(nodes):
            node_id = id(node)
            if node_id not in node_occurrences:
                node_occurrences[node_id] = [idx]
                unique_nodes.append(node)
            else:
                node_occurrences[node_id].append(idx)

        terminal_values = {}
        non_terminal_nodes = []

        for node in unique_nodes:
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

        values_by_node_id = {}

        if non_terminal_nodes:
            # 🆕 v4.2: Stack tensors WITH HISTORY
            boards_np = np.stack(
                [
                    self._build_history_tensor(
                        node.board,
                        current_tensor=self._current_tensor_for_node(node),
                    )
                    for node in non_terminal_nodes
                ],
                axis=0,
            )
            board_tensors = torch.from_numpy(boards_np).to(
                self.device,
                memory_format=torch.channels_last,
                non_blocking=True,
            )

            # Single GPU call (inference optimized)
            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                        policy_logits_batch, values_batch = self.model(
                            board_tensors,
                            apply_log_softmax=False,
                        )
                else:
                    policy_logits_batch, values_batch = self.model(
                        board_tensors,
                        apply_log_softmax=False,
                    )
                
                # 🔥 OPTIMIZATION: Transfer entire batch to CPU at once, not row by row
                if values_batch.dim() == 2 and values_batch.shape[1] == 3:
                    # WDL output: compute softmax on GPU before transfer
                    wdl_probs = torch.softmax(values_batch, dim=1)
                    values_batch = (wdl_probs[:, 0] - wdl_probs[:, 2])
                
                # Convert to float32 before numpy() because numpy doesn't support bfloat16
                policy_logits_batch = policy_logits_batch.float().cpu().numpy()
                values_batch = values_batch.float().cpu().numpy()

            # Process results
            for idx, node in enumerate(non_terminal_nodes):
                policy_logits = policy_logits_batch[idx]
                
                # Scalar output (already processed if WDL)
                value = float(values_batch[idx])

                legal_moves = list(node.board.legal_moves)
                legal_indices = [move_to_index(m, node.board) for m in legal_moves]

                # Compute probabilities only for legal moves (faster than full ACTION_SIZE)
                legal_logits = policy_logits[legal_indices]
                legal_probs = np.exp(legal_logits - legal_logits.max())
                legal_probs = legal_probs / (legal_probs.sum() + 1e-8)

                # Dirichlet noise
                if add_noise and node is self.root and len(legal_moves) > 0:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
                    legal_probs = (1 - self.dirichlet_weight) * legal_probs + self.dirichlet_weight * noise

                if not node.expanded:
                    # Create children lazily (saves massive CPU and RAM)
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
    
    def _backpropagate(self, search_path, value):
        """Backpropagate value"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value
    
    def update_history(self, board):
        """
        🆕 v4.2: Update board history after a move
        Call this from play.py after each move
        
        Args:
            board: chess.Board that was just played
        """
        # Store cached tensors for both POVs to avoid repeated FEN parse + tensor rebuild.
        self.board_history.append(self._encode_history_entry(board))
        
        # Keep only last N positions needed for history
        max_history = self.history_positions + 10  # Keep a few extra for safety
        if len(self.board_history) > max_history:
            self.board_history = self.board_history[-max_history:]
    
    def reset_tree(self):
        """Reset tree (call after game ends)"""
        self.root = None
        self.board_history = []  # 🆕 v4.2: Also reset history


def select_move_by_visits(visit_counts, temperature=1.0):
    """Select move based on visit counts with temperature"""
    moves = list(visit_counts.keys())
    visits = np.array([visit_counts[m] for m in moves])
    
    if temperature == 0 or len(moves) == 1:
        best_idx = np.argmax(visits)
        return moves[best_idx], visits
    else:
        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / visits_temp.sum()
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx], probs


# Backwards compatibility
MCTS = BatchMCTS
