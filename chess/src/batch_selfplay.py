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


_EMPTY_HISTORY_TENSOR = np.zeros((16, 8, 8), dtype=np.float32)


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(self, board=None, parent=None, move=None, prior=0.0, copy_board=True):
        if board is not None:
            self._board = board.copy() if copy_board else board
        else:
            self._board = None
        self.parent = parent
        self.move = move
        self.prior = prior
        self.base_prior = prior

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False
        self.virtual_loss = 0

        self._fen_cache = None
        self._is_game_over = None
        self._board_tensor = None
        self._legal_moves = None
        self._legal_indices = None

    @property
    def board(self):
        if self._board is None:
            self._board = self.parent.board.copy()
            self._board.push(self.move)
        return self._board

    @property
    def is_game_over(self):
        if self._is_game_over is None:
            self._is_game_over = self.board.is_game_over(claim_draw=True)
        return self._is_game_over

    def is_leaf(self):
        return not self.expanded

    def add_virtual_loss(self, n=1):
        self.virtual_loss += n

    def remove_virtual_loss(self, n=1):
        self.virtual_loss -= n

    def get_fen(self):
        if self._fen_cache is None:
            self._fen_cache = self.board.fen()
        return self._fen_cache

    def get_legal_moves_and_indices(self):
        if self._legal_moves is None or self._legal_indices is None:
            legal_moves = tuple(self.board.legal_moves)
            legal_indices = np.fromiter(
                (move_to_index(move, self.board) for move in legal_moves),
                dtype=np.int32,
                count=len(legal_moves),
            )
            self._legal_moves = legal_moves
            self._legal_indices = legal_indices
        return self._legal_moves, self._legal_indices


def _build_sparse_policy_target_from_visits(visit_counts, board):
    """
    Convert MCTS visit counts into sparse policy targets.

    Returns:
        (indices, probs)
        indices: int16 tensor of action indices
        probs: float32 tensor of normalized visit probabilities
    """
    if not visit_counts:
        return (
            torch.empty(0, dtype=torch.int16),
            torch.empty(0, dtype=torch.float32),
        )

    moves = list(visit_counts.keys())
    indices = np.empty(len(moves), dtype=np.int16)
    visits = np.empty(len(moves), dtype=np.float32)

    for i, move in enumerate(moves):
        indices[i] = move_to_index(move, board)
        visits[i] = float(visit_counts[move])

    total_visits = float(visits.sum())
    if total_visits > 0:
        probs = visits / total_visits
    else:
        probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)

    return (
        torch.from_numpy(indices),
        torch.from_numpy(probs.astype(np.float32, copy=False)),
    )


def _move_selection_temperature(board, base_temperature, threshold_fullmoves):
    """
    Use exploratory sampling for the first N full moves of the game.
    """
    if threshold_fullmoves is None:
        return base_temperature
    return base_temperature if board.fullmove_number <= threshold_fullmoves else 0.01


def _select_move_from_visits_safe(visit_counts, temperature):
    """
    Select move from visit counts with numeric safeguards.
    """
    moves = list(visit_counts.keys())
    visits = np.fromiter(visit_counts.values(), dtype=np.float64, count=len(moves))

    if temperature == 0 or len(moves) == 1:
        return moves[int(np.argmax(visits))]

    visits_temp = visits ** (1.0 / temperature)
    total = float(visits_temp.sum())
    if total <= 0 or not np.isfinite(total):
        return moves[np.random.randint(len(moves))]

    probs = visits_temp / total
    probs = np.clip(probs, 0.0, 1.0)
    probs_sum = float(probs.sum())
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        return moves[np.random.randint(len(moves))]

    probs /= probs_sum
    idx = np.random.choice(len(moves), p=probs)
    return moves[idx]


def select_move_by_visits(visit_counts, temperature=1.0):
    """Compatibility helper used by UI/eval code paths."""
    moves = list(visit_counts.keys())
    visits = np.fromiter(visit_counts.values(), dtype=np.float64, count=len(moves))

    if temperature == 0 or len(moves) == 1:
        best_idx = int(np.argmax(visits))
        return moves[best_idx], visits

    visits_temp = visits ** (1.0 / temperature)
    total = float(visits_temp.sum())
    if total <= 0 or not np.isfinite(total):
        probs = np.full(len(moves), 1.0 / max(1, len(moves)), dtype=np.float64)
    else:
        probs = visits_temp / total
        probs = np.clip(probs, 0.0, 1.0)
        probs_sum = float(probs.sum())
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.full(len(moves), 1.0 / max(1, len(moves)), dtype=np.float64)
        else:
            probs = probs / probs_sum

    idx = int(np.random.choice(len(moves), p=probs))
    return moves[idx], probs


def _resolve_selfplay_max_moves(config):
    rl_cfg = config.get('reinforcement_learning', {})
    raw_value = rl_cfg.get('self_play_max_moves', 300)
    try:
        return max(1, int(raw_value))
    except Exception:
        return 300


def _build_history_tensor_from_encoded(
    turn,
    encoded_history,
    history_count,
    history_positions,
    empty_history_tensor,
):
    """Build model input from cached history tensors without FEN parsing."""
    use_black_pov = (turn == chess.BLACK)
    available = len(encoded_history)
    if available <= 0:
        return empty_history_tensor

    current_idx = max(0, min(int(history_count), available - 1))
    current_encoded = encoded_history[current_idx]
    current_tensor = current_encoded[1] if use_black_pov else current_encoded[0]

    if history_positions <= 0:
        return current_tensor

    end = max(0, min(int(history_count), available))
    start = max(0, end - int(history_positions))
    history_slice = encoded_history[start:end]

    history_tensors = []
    for encoded in history_slice:
        history_tensors.append(encoded[1] if use_black_pov else encoded[0])

    pad_count = max(0, int(history_positions) - len(history_tensors))
    if pad_count:
        history_tensors = [empty_history_tensor] * pad_count + history_tensors

    history_tensors.append(current_tensor)
    return np.concatenate(history_tensors, axis=0)


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
        self.use_dynamic_c_puct = bool(
            config['reinforcement_learning'].get('mcts_dynamic_c_puct', True)
        )
        self.c_puct_base = float(
            config['reinforcement_learning'].get('mcts_c_puct_base', 19652)
        )
        self.c_puct_init = float(
            config['reinforcement_learning'].get('mcts_c_puct_init', self.c_puct)
        )
        c_puct_max_raw = config['reinforcement_learning'].get('mcts_c_puct_max', None)
        self.c_puct_max = float(c_puct_max_raw) if c_puct_max_raw is not None else None
        self.use_fpu = bool(config['reinforcement_learning'].get('mcts_use_fpu', True))
        self.fpu_reduction = float(
            config['reinforcement_learning'].get('mcts_fpu_reduction', 0.30)
        )
        self.fpu_absolute = config['reinforcement_learning'].get('mcts_fpu_absolute', None)
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

    def _build_history_tensor(self, current_board, board_history, current_tensor=None):
        """
        Build tensor with history using POV-aware board_to_tensor.

        Args:
            current_board: chess.Board for current position
            board_history: list of previous boards (from real game history)
        """
        if self.history_positions == 0:
            return board_to_tensor(current_board)

        history_tensors = []
        use_black_pov = current_board.turn == chess.BLACK
        if board_history:
            history_boards = board_history[-self.history_positions:]
            for hist_entry in history_boards:
                encoded = self._encode_history_entry(hist_entry)
                hist_tensor = encoded[1] if use_black_pov else encoded[0]
                history_tensors.append(hist_tensor)

        pad_count = max(0, self.history_positions - len(history_tensors))
        if pad_count:
            history_tensors = [self._empty_history_tensor] * pad_count + history_tensors

        if current_tensor is None:
            current_tensor = board_to_tensor(current_board)
        history_tensors.append(current_tensor)

        return np.concatenate(history_tensors, axis=0)

    @staticmethod
    def _current_tensor_for_node(node):
        cached = getattr(node, "_board_tensor", None)
        if cached is None:
            cached = board_to_tensor(node.board)
            node._board_tensor = cached
        return cached

    def _select_child(self, node):
        """Select child with highest UCB score (optimized)"""
        best_score = -float('inf')
        best_child = None

        # Precalculate parent term to avoid doing it for every child
        parent_visits = node.visit_count + node.virtual_loss
        parent_sqrt = math.sqrt(parent_visits + 1)
        if self.use_dynamic_c_puct:
            c_puct = math.log((parent_visits + self.c_puct_base + 1.0) / self.c_puct_base) + self.c_puct_init
            if self.c_puct_max is not None:
                c_puct = min(c_puct, self.c_puct_max)
        else:
            c_puct = self.c_puct

        fpu_value = 0.0
        if self.use_fpu:
            if self.fpu_absolute is not None:
                fpu_value = float(self.fpu_absolute)
            elif parent_visits > 0:
                parent_q = (node.value_sum - node.virtual_loss) / parent_visits
                explored_prior = 0.0
                for child in node.children.values():
                    if (child.visit_count + child.virtual_loss) > 0:
                        explored_prior += child.prior
                unexplored_prior = max(0.0, 1.0 - explored_prior)
                fpu_value = parent_q - self.fpu_reduction * math.sqrt(unexplored_prior)

        for _, child in node.children.items():
            # Inline value() and ucb_score() for speed
            cv = child.visit_count + child.virtual_loss
            if cv == 0:
                q_value = fpu_value
            else:
                q_value = (child.value_sum - child.virtual_loss) / cv
                
            u_value = c_puct * child.prior * parent_sqrt / (1 + cv)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _apply_root_noise(self, node):
        """Apply fresh Dirichlet noise to an already expanded root node."""
        if self.dirichlet_weight <= 0 or not node.children:
            return

        children = list(node.children.values())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        mix = self.dirichlet_weight

        for child, noise_value in zip(children, noise):
            child.prior = (1.0 - mix) * child.base_prior + mix * float(noise_value)

    def _backpropagate(self, search_path, value):
        """Backpropagate value"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def search_many(self, game_states, num_simulations, add_root_noise=False):
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
            root_synced = bool(gs.get('_root_synced', False))

            if self.reuse_tree and root is not None:
                if root_synced:
                    gs['_root_synced'] = False
                else:
                    target_fen = board.fen()
                    if root.get_fen() == target_fen:
                        pass
                    else:
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
                gs['_root_synced'] = False

            gs['root'] = root
            if add_root_noise and root.expanded:
                self._apply_root_noise(root)
                gs['_needs_root_noise_on_expand'] = False
            else:
                gs['_needs_root_noise_on_expand'] = bool(add_root_noise and not root.expanded)

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
            selected_this_batch = [0] * len(game_states)

            for _ in range(batch_size):
                # Find next game with remaining sims
                found = False
                for _ in range(len(game_states)):
                    current_gs = game_states[game_ptr]
                    if (
                        remaining[game_ptr] > 0
                        and not (
                            selected_this_batch[game_ptr] > 0
                            and current_gs['root'] is not None
                            and not current_gs['root'].expanded
                        )
                    ):
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

                while not node.is_leaf():
                    node = self._select_child(node)
                    node.add_virtual_loss()
                    search_path.append(node)

                search_paths.append(search_path)
                leaf_nodes.append(node)
                leaf_game_indices.append(gs_idx)
                selected_this_batch[gs_idx] += 1

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
                result = node.board.result(claim_draw=True)
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
            legal_moves_per_node = []
            legal_indices_per_node = []
            legal_counts = []
            max_legal_count = 0
            for node in non_terminal_nodes:
                legal_moves, legal_indices = node.get_legal_moves_and_indices()
                legal_moves_per_node.append(legal_moves)
                legal_indices_per_node.append(legal_indices)
                legal_count = len(legal_indices)
                legal_counts.append(legal_count)
                if legal_count > max_legal_count:
                    max_legal_count = legal_count

            boards_np = np.stack(
                [
                    self._build_history_tensor(
                        node.board,
                        game_states[gi]['board_history'],
                        current_tensor=self._current_tensor_for_node(node),
                    )
                    for node, gi in zip(non_terminal_nodes, non_terminal_game_indices)
                ],
                axis=0,
            )
            board_tensors = torch.from_numpy(boards_np).to(
                self.device,
                memory_format=torch.channels_last,
                non_blocking=True,
            )

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

                # Gather only legal move logits on GPU before transferring to CPU.
                if max_legal_count > 0:
                    legal_index_matrix = np.zeros(
                        (len(non_terminal_nodes), max_legal_count),
                        dtype=np.int64,
                    )
                    for row_idx, legal_indices in enumerate(legal_indices_per_node):
                        legal_count = legal_counts[row_idx]
                        if legal_count:
                            legal_index_matrix[row_idx, :legal_count] = legal_indices
                    legal_index_tensor = torch.from_numpy(legal_index_matrix).to(
                        self.device,
                        non_blocking=True,
                    )
                    legal_logits_batch = torch.gather(policy_logits_batch, 1, legal_index_tensor)
                    legal_logits_batch = legal_logits_batch.to(dtype=torch.float16).cpu().numpy()
                else:
                    legal_logits_batch = None
                values_batch = values_batch.float().cpu().numpy()

            for idx, node in enumerate(non_terminal_nodes):
                value = float(values_batch[idx])

                legal_moves = legal_moves_per_node[idx]
                legal_count = legal_counts[idx]

                if legal_moves:
                    legal_logits = legal_logits_batch[idx, :legal_count].astype(np.float32, copy=False)
                    legal_probs = np.exp(legal_logits - legal_logits.max())
                    legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
                else:
                    legal_probs = np.array([])

                gi = non_terminal_game_indices[idx]
                add_noise = (
                    game_states[gi].get('_needs_root_noise_on_expand', False)
                    and node is game_states[gi]['root']
                )

                if add_noise and len(legal_moves) > 0:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
                    legal_probs = (1 - self.dirichlet_weight) * legal_probs + self.dirichlet_weight * noise
                    game_states[gi]['_needs_root_noise_on_expand'] = False

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


class BatchMCTS:
    """
    Single-game compatibility wrapper over MultiGameBatchMCTS.

    Keeps the old API (`search`, `advance_root`, `update_history`, `reset_tree`)
    while using the same shared multi-game MCTS core.
    """

    def __init__(self, model, config, device):
        self.config = config
        self._multi = MultiGameBatchMCTS(model, config, device)
        self.root = None
        self._root_synced = False
        self.board_history = []

    def search(self, board, num_simulations, temperature=1.0, add_root_noise=False):
        del temperature  # visit selection temperature is handled by caller

        game_state = {
            'board': board,
            'root': self.root,
            '_root_synced': self._root_synced,
            'board_history': self.board_history,
        }
        visit_counts = self._multi.search_many(
            [game_state],
            num_simulations=num_simulations,
            add_root_noise=add_root_noise,
        )
        self.root = game_state.get('root')
        self._root_synced = bool(game_state.get('_root_synced', False))
        return visit_counts[0] if visit_counts else {}

    def advance_root(self, move):
        if self.root is None:
            return
        child = self.root.children.get(move)
        if child is None:
            self.root = None
            self._root_synced = False
            return
        _ = child.board
        child.parent = None
        self.root = child
        self._root_synced = True

    def update_history(self, board):
        self.board_history.append(self._multi._encode_history_entry(board))
        max_history = int(self.config.get('model', {}).get('history_positions', 0)) + 10
        if len(self.board_history) > max_history:
            self.board_history = self.board_history[-max_history:]

    def reset_tree(self):
        self.root = None
        self._root_synced = False
        self.board_history = []


class BatchSelfPlayMCTSBatch:
    """
    True batch self-play: multiple games in parallel, shared GPU eval batches.
    """

    def __init__(self, model, config, device, max_batch_games_per_worker, opponent_model=None):
        self.model = model
        self.opponent_model = opponent_model
        self.config = config
        self.device = device
        self.model.eval()
        if self.opponent_model is not None:
            self.opponent_model.eval()
        rl_cfg = config.get('reinforcement_learning', {})

        # Use multi-game MCTS (dual-model optional: primary vs opponent)
        self.mcts = MultiGameBatchMCTS(model, config, device)
        self.opponent_mcts = MultiGameBatchMCTS(opponent_model, config, device) if opponent_model is not None else None
        self.dual_model_enabled = bool(self.opponent_mcts is not None and rl_cfg.get('league_dual_color_models_enabled', True))
        self.train_primary_positions_only = bool(
            rl_cfg.get('league_train_primary_positions_only', True)
        )

        if device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
            if self.opponent_model is not None:
                self.opponent_model = self.opponent_model.to(memory_format=torch.channels_last)

        self.num_simulations = config['reinforcement_learning']['mcts_simulations']
        self.temp_threshold = config['reinforcement_learning']['mcts_temperature_threshold']
        self.temperature = config['reinforcement_learning'].get('mcts_temperature', 1.0)
        self.max_moves = _resolve_selfplay_max_moves(config)

        # Optional draw shaping to reduce drawish equilibria in very long games.
        self.draw_value_shaping_enabled = bool(
            rl_cfg.get('draw_value_shaping_enabled', False)
        )
        self.draw_value_shaping_after_moves = int(
            rl_cfg.get('draw_value_shaping_after_moves', max(1, self.max_moves // 2))
        )
        self.draw_value_shaping_max_penalty = float(
            rl_cfg.get('draw_value_shaping_max_penalty', 0.08)
        )
        self.draw_value_shaping_power = float(
            rl_cfg.get('draw_value_shaping_power', 1.5)
        )

        # Optional safe resign logic to force decisive outcomes without
        # triggering on noisy single-step value estimates.
        self.resign_enabled = bool(rl_cfg.get('resign_enabled', False))
        self.resign_threshold = float(rl_cfg.get('resign_threshold', -0.9))
        self.resign_min_moves = int(rl_cfg.get('resign_min_moves', 40))
        self.resign_check_interval = max(1, int(rl_cfg.get('resign_check_interval', 4)))
        self.resign_required_confirmations = max(
            1,
            int(rl_cfg.get('resign_required_confirmations', 3)),
        )

        self.max_batch_games_per_worker = max(1, int(max_batch_games_per_worker))
        self._progress_file = None  # Set externally to enable progress reporting
        self._games_completed = 0
        self._progress_base = 0

    def _is_primary_to_move(self, gs):
        return gs['board'].turn == gs['primary_color']

    def _active_mcts_for_game(self, gs):
        if not self.dual_model_enabled:
            return self.mcts
        return self.mcts if self._is_primary_to_move(gs) else self.opponent_mcts

    def _active_root_keys_for_game(self, gs):
        if not self.dual_model_enabled:
            return 'root', '_root_synced', None, None
        if self._is_primary_to_move(gs):
            return 'root_primary', '_root_primary_synced', 'root_opponent', '_root_opponent_synced'
        return 'root_opponent', '_root_opponent_synced', 'root_primary', '_root_primary_synced'

    def _compute_draw_value_target(self, move_count):
        """Return shaped draw target in [-1, 0], stronger only for long draws."""
        if not self.draw_value_shaping_enabled:
            return 0.0
        if self.draw_value_shaping_max_penalty <= 0:
            return 0.0
        if move_count <= self.draw_value_shaping_after_moves:
            return 0.0

        den = max(1, self.max_moves - self.draw_value_shaping_after_moves)
        progress = (move_count - self.draw_value_shaping_after_moves) / den
        progress = max(0.0, min(1.0, progress))
        shaped = -self.draw_value_shaping_max_penalty * (progress ** self.draw_value_shaping_power)
        return float(max(-1.0, min(0.0, shaped)))

    def play_games(self, num_games):
        all_positions = []
        game_lengths = []
        self._games_completed = 0
        total_dropped_positions = 0
        total_truncated_games = 0
        total_claimable_draw_ended_games = 0
        total_completed_length_sum = 0
        total_truncated_length_sum = 0
        total_completed_white_wins = 0
        total_completed_black_wins = 0
        total_completed_draws = 0
        total_primary_wins = 0
        total_opponent_wins = 0
        total_model_draws = 0
        total_primary_white_games = 0
        total_primary_white_wins = 0
        total_primary_black_games = 0
        total_primary_black_wins = 0

        games_left = num_games
        batch_idx = 0

        while games_left > 0:
            batch_size = min(self.max_batch_games_per_worker, games_left)
            batch_idx += 1

            positions, lengths, batch_stats = self._play_batch(batch_size)
            all_positions.extend(positions)
            game_lengths.extend(lengths)
            total_dropped_positions += int(batch_stats.get('dropped_positions', 0))
            total_truncated_games += int(batch_stats.get('truncated_games', 0))
            total_claimable_draw_ended_games += int(batch_stats.get('claimable_draw_ended_games', 0))
            total_completed_length_sum += int(batch_stats.get('completed_length_sum', 0))
            total_truncated_length_sum += int(batch_stats.get('truncated_length_sum', 0))
            total_completed_white_wins += int(batch_stats.get('completed_white_wins', 0))
            total_completed_black_wins += int(batch_stats.get('completed_black_wins', 0))
            total_completed_draws += int(batch_stats.get('completed_draws', 0))
            total_primary_wins += int(batch_stats.get('primary_model_wins', 0))
            total_opponent_wins += int(batch_stats.get('opponent_model_wins', 0))
            total_model_draws += int(batch_stats.get('model_match_draws', 0))
            total_primary_white_games += int(batch_stats.get('primary_white_games', 0))
            total_primary_white_wins += int(batch_stats.get('primary_white_wins', 0))
            total_primary_black_games += int(batch_stats.get('primary_black_games', 0))
            total_primary_black_wins += int(batch_stats.get('primary_black_wins', 0))

            games_left -= batch_size

        total_games = len(game_lengths)
        self.last_selfplay_stats = {
            'total_games': total_games,
            'truncated_games': total_truncated_games,
            'completed_games': max(0, total_games - total_truncated_games),
            'dropped_positions': int(total_dropped_positions),
            'claimable_draw_ended_games': int(total_claimable_draw_ended_games),
            'completed_length_sum': int(total_completed_length_sum),
            'truncated_length_sum': int(total_truncated_length_sum),
            'completed_white_wins': int(total_completed_white_wins),
            'completed_black_wins': int(total_completed_black_wins),
            'completed_draws': int(total_completed_draws),
            'primary_model_wins': int(total_primary_wins),
            'opponent_model_wins': int(total_opponent_wins),
            'model_match_draws': int(total_model_draws),
            'primary_white_games': int(total_primary_white_games),
            'primary_white_wins': int(total_primary_white_wins),
            'primary_black_games': int(total_primary_black_games),
            'primary_black_wins': int(total_primary_black_wins),
        }
        return all_positions, game_lengths

    def _report_progress(self):
        if self._progress_file is None:
            return
        try:
            with open(self._progress_file, 'w') as _pf:
                _pf.write(str(self._progress_base + self._games_completed))
        except Exception:
            pass

    def _mark_game_completed(self, gs):
        if gs.get('_completion_reported', False):
            return
        gs['_completion_reported'] = True
        self._games_completed += 1
        self._report_progress()

    def _play_batch(self, batch_size):
        max_moves = self.max_moves

        game_states = []
        for game_idx in range(batch_size):
            primary_color = chess.WHITE if (game_idx % 2 == 0) else chess.BLACK
            game_states.append({
                'board': chess.Board(),
                'board_history': [],
                'root': None,
                '_root_synced': False,
                'root_primary': None,
                '_root_primary_synced': False,
                'root_opponent': None,
                '_root_opponent_synced': False,
                'primary_color': primary_color,
                'game_history': [],
                'move_count': 0,
                'resign_confirmations': 0,
                'resigned_outcome': None,
                'done': False,
                '_completion_reported': False,
            })

        while True:
            active_indices = [i for i, gs in enumerate(game_states) if not gs['done']]
            if not active_indices:
                break

            if self.resign_enabled:
                resign_candidate_indices_primary = []
                resign_tensors_primary = []
                resign_candidate_indices_opponent = []
                resign_tensors_opponent = []

                for idx in active_indices:
                    gs = game_states[idx]
                    if gs['move_count'] < self.resign_min_moves:
                        continue

                    mcts_eval = self._active_mcts_for_game(gs)
                    tensor = mcts_eval._build_history_tensor(
                        gs['board'],
                        gs['board_history'],
                    )
                    if self.dual_model_enabled and not self._is_primary_to_move(gs):
                        resign_candidate_indices_opponent.append(idx)
                        resign_tensors_opponent.append(tensor)
                    else:
                        resign_candidate_indices_primary.append(idx)
                        resign_tensors_primary.append(tensor)

                def _run_resign_eval(indices, tensors, model_ref, mcts_ref):
                    if not indices:
                        return
                    boards_np = np.stack(tensors, axis=0)
                    board_tensors = torch.from_numpy(boards_np).to(
                        self.device,
                        memory_format=torch.channels_last,
                        non_blocking=True,
                    )
                    with torch.inference_mode():
                        if mcts_ref.use_amp:
                            with torch.autocast(device_type='cuda', dtype=mcts_ref.amp_dtype):
                                _, resign_values = model_ref(
                                    board_tensors,
                                    apply_log_softmax=False,
                                )
                        else:
                            _, resign_values = model_ref(
                                board_tensors,
                                apply_log_softmax=False,
                            )

                        if resign_values.dim() == 2 and resign_values.shape[1] == 3:
                            wdl_probs = torch.softmax(resign_values, dim=1)
                            resign_values = (wdl_probs[:, 0] - wdl_probs[:, 2])
                        resign_values_np = resign_values.float().cpu().numpy()

                    for i, gs_idx in enumerate(indices):
                        gs = game_states[gs_idx]
                        value_est = float(resign_values_np[i])
                        if value_est <= self.resign_threshold:
                            gs['resign_confirmations'] += 1
                        else:
                            gs['resign_confirmations'] = 0

                        if gs['resign_confirmations'] >= self.resign_required_confirmations:
                            # Side to move resigns: opposite side wins.
                            gs['resigned_outcome'] = -1.0 if gs['board'].turn == chess.WHITE else 1.0
                            gs['done'] = True
                            gs['root'] = None
                            gs['root_primary'] = None
                            gs['root_opponent'] = None
                            self._mark_game_completed(gs)

                _run_resign_eval(resign_candidate_indices_primary, resign_tensors_primary, self.model, self.mcts)
                if self.dual_model_enabled:
                    _run_resign_eval(
                        resign_candidate_indices_opponent,
                        resign_tensors_opponent,
                        self.opponent_model,
                        self.opponent_mcts,
                    )

                active_indices = [i for i, gs in enumerate(game_states) if not gs['done']]
                if not active_indices:
                    break

            visit_counts_by_index = {}

            def _run_search_for_indices(indices, mcts_ref, root_key, synced_key):
                if not indices:
                    return
                group_states = []
                for i in indices:
                    gs = game_states[i]
                    group_states.append({
                        'board': gs['board'],
                        'root': gs.get(root_key),
                        '_root_synced': bool(gs.get(synced_key, False)),
                        'board_history': gs['board_history'],
                    })
                visit_counts_list_group = mcts_ref.search_many(
                    group_states,
                    num_simulations=self.num_simulations,
                    add_root_noise=True,
                )
                for gs_idx, local_state, visit_counts in zip(indices, group_states, visit_counts_list_group):
                    gs = game_states[gs_idx]
                    gs[root_key] = local_state.get('root')
                    gs[synced_key] = bool(local_state.get('_root_synced', False))
                    visit_counts_by_index[gs_idx] = visit_counts

            if self.dual_model_enabled:
                primary_indices = [i for i in active_indices if self._is_primary_to_move(game_states[i])]
                opponent_indices = [i for i in active_indices if not self._is_primary_to_move(game_states[i])]
                _run_search_for_indices(primary_indices, self.mcts, 'root_primary', '_root_primary_synced')
                _run_search_for_indices(opponent_indices, self.opponent_mcts, 'root_opponent', '_root_opponent_synced')
            else:
                _run_search_for_indices(active_indices, self.mcts, 'root', '_root_synced')

            for idx in active_indices:
                gs = game_states[idx]
                board = gs['board']
                visit_counts = visit_counts_by_index.get(idx, None)
                if visit_counts is None:
                    continue

                if self.temp_threshold > 0 and gs['move_count'] < self.temp_threshold:
                    temperature = self.temperature
                else:
                    temperature = 0.0
                move = self._select_move_from_visits(visit_counts, temperature)

                policy_indices, policy_values = _build_sparse_policy_target_from_visits(visit_counts, board)
                history_count = len(gs['board_history'])
                gs['game_history'].append((
                    history_count,
                    policy_indices,
                    policy_values,
                    board.turn,
                    self._is_primary_to_move(gs),
                ))

                active_root_key, active_synced_key, inactive_root_key, inactive_synced_key = self._active_root_keys_for_game(gs)
                root = gs.get(active_root_key)
                # Update history BEFORE making the move
                # Store cached tensors for both POVs to avoid repeated FEN parse + tensor rebuild.
                gs['board_history'].append(self.mcts._encode_history_entry(board))

                # Reuse selected subtree directly to skip FEN-matching next turn.
                if root is not None and move in root.children:
                    next_root = root.children[move]
                    _ = next_root.board
                    next_root.parent = None
                    gs[active_root_key] = next_root
                    gs[active_synced_key] = True
                else:
                    gs[active_root_key] = None
                    gs[active_synced_key] = False

                if inactive_root_key is not None:
                    gs[inactive_synced_key] = False
                else:
                    gs['_root_synced'] = bool(gs.get('_root_synced', False))

                board.push(move)
                gs['move_count'] += 1

                if board.is_game_over(claim_draw=True) or gs['move_count'] >= max_moves:
                    gs['done'] = True
                    # Free up memory immediately
                    gs['root'] = None
                    gs['root_primary'] = None
                    gs['root_opponent'] = None
                    self._mark_game_completed(gs)

        positions = []
        game_lengths = []
        dropped_positions = 0
        truncated_games = 0
        claimable_draw_ended_games = 0
        completed_length_sum = 0
        truncated_length_sum = 0
        completed_white_wins = 0
        completed_black_wins = 0
        completed_draws = 0
        primary_model_wins = 0
        opponent_model_wins = 0
        model_match_draws = 0
        primary_white_games = 0
        primary_white_wins = 0
        primary_black_games = 0
        primary_black_wins = 0
        history_positions = int(self.config.get('model', {}).get('history_positions', 0))

        for gs in game_states:
            board = gs['board']
            resigned_outcome = gs.get('resigned_outcome', None)
            if resigned_outcome is not None:
                outcome = float(resigned_outcome)
                if outcome > 0:
                    completed_white_wins += 1
                else:
                    completed_black_wins += 1
                primary_color = gs.get('primary_color', chess.WHITE)
                if primary_color == chess.WHITE:
                    primary_white_games += 1
                    if outcome > 0:
                        primary_white_wins += 1
                else:
                    primary_black_games += 1
                    if outcome < 0:
                        primary_black_wins += 1
                if (outcome > 0 and primary_color == chess.WHITE) or (outcome < 0 and primary_color == chess.BLACK):
                    primary_model_wins += 1
                else:
                    opponent_model_wins += 1

                history_len = len(gs['game_history'])
                completed_length_sum += history_len

                for history_count, policy_indices, policy_values, turn, was_primary_turn in gs['game_history']:
                    if self.dual_model_enabled and self.train_primary_positions_only and (not was_primary_turn):
                        continue
                    value = outcome if turn == chess.WHITE else -outcome
                    board_tensor_np = _build_history_tensor_from_encoded(
                        turn=turn,
                        encoded_history=gs['board_history'],
                        history_count=history_count,
                        history_positions=history_positions,
                        empty_history_tensor=_EMPTY_HISTORY_TENSOR,
                    )
                    board_tensor = torch.from_numpy(board_tensor_np)
                    positions.append((
                        board_tensor,
                        policy_indices,
                        policy_values,
                        torch.tensor([value], dtype=torch.float32)
                    ))

                game_lengths.append(history_len)
                continue

            ended_by_claimable_draw = (
                gs['move_count'] < max_moves
                and board.is_game_over(claim_draw=True)
                and not board.is_game_over()
                and board.can_claim_draw()
            )
            if ended_by_claimable_draw:
                claimable_draw_ended_games += 1
            result = board.result(claim_draw=True)
            if result == '*':
                truncated_games += 1
                history_len = len(gs['game_history'])
                dropped_positions += history_len
                truncated_length_sum += history_len
                game_lengths.append(history_len)
                continue
            if result == '1-0':
                outcome = 1.0
                completed_white_wins += 1
            elif result == '0-1':
                outcome = -1.0
                completed_black_wins += 1
            else:
                outcome = 0.0
                completed_draws += 1
            primary_color = gs.get('primary_color', chess.WHITE)
            if primary_color == chess.WHITE:
                primary_white_games += 1
                if outcome > 0:
                    primary_white_wins += 1
            else:
                primary_black_games += 1
                if outcome < 0:
                    primary_black_wins += 1
            if outcome == 0.0:
                model_match_draws += 1
            elif (outcome > 0 and primary_color == chess.WHITE) or (outcome < 0 and primary_color == chess.BLACK):
                primary_model_wins += 1
            else:
                opponent_model_wins += 1
            draw_value_target = self._compute_draw_value_target(gs['move_count'])

            history_len = len(gs['game_history'])
            completed_length_sum += history_len

            for history_count, policy_indices, policy_values, turn, was_primary_turn in gs['game_history']:
                if self.dual_model_enabled and self.train_primary_positions_only and (not was_primary_turn):
                    continue
                if outcome == 0.0:
                    value = draw_value_target
                else:
                    value = outcome if turn == chess.WHITE else -outcome
                board_tensor_np = _build_history_tensor_from_encoded(
                    turn=turn,
                    encoded_history=gs['board_history'],
                    history_count=history_count,
                    history_positions=history_positions,
                    empty_history_tensor=_EMPTY_HISTORY_TENSOR,
                )
                board_tensor = torch.from_numpy(
                    board_tensor_np
                )
                positions.append((
                    board_tensor,
                    policy_indices,
                    policy_values,
                    torch.tensor([value], dtype=torch.float32)
                ))

            game_lengths.append(history_len)

        return positions, game_lengths, {
            'truncated_games': int(truncated_games),
            'dropped_positions': int(dropped_positions),
            'claimable_draw_ended_games': int(claimable_draw_ended_games),
            'completed_length_sum': int(completed_length_sum),
            'truncated_length_sum': int(truncated_length_sum),
            'completed_white_wins': int(completed_white_wins),
            'completed_black_wins': int(completed_black_wins),
            'completed_draws': int(completed_draws),
            'primary_model_wins': int(primary_model_wins),
            'opponent_model_wins': int(opponent_model_wins),
            'model_match_draws': int(model_match_draws),
            'primary_white_games': int(primary_white_games),
            'primary_white_wins': int(primary_white_wins),
            'primary_black_games': int(primary_black_games),
            'primary_black_wins': int(primary_black_wins),
        }

    def _select_move_from_visits(self, visit_counts, temperature):
        return _select_move_from_visits_safe(visit_counts, temperature)


# ============================================================================
# PARALLEL WORKER FOR MULTIPROCESSING
# ============================================================================

def _build_worker_logger(rank, worker_verbose):
    def _wlog(message):
        if worker_verbose:
            print(f"Worker {rank}: {message}")

    return _wlog


def _configure_selfplay_worker_runtime(config, device_id):
    rl_cfg = config.get('reinforcement_learning', {})
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

    if device_id == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    return device


def _build_selfplay_worker_model(config, device):
    from src.model import ChessNet

    worker_config = dict(config)
    worker_config['model'] = {**config.get('model', {}), 'print_summary': False}
    model = ChessNet(worker_config).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def _load_worker_model_state(model, model_state, rank):
    from src.model import normalize_state_dict_keys

    # Accept both raw state_dict and full training checkpoints.
    if isinstance(model_state, dict):
        if 'model_state_dict' in model_state and isinstance(model_state.get('model_state_dict'), dict):
            model_state = model_state['model_state_dict']
        elif 'state_dict' in model_state and isinstance(model_state.get('state_dict'), dict):
            model_state = model_state['state_dict']

    if model_state:
        model_state = {
            k: v for k, v in model_state.items()
            if not (k.endswith('coord_x') or k.endswith('coord_y'))
        }
        model_state = normalize_state_dict_keys(
            model_state,
            target_keys=set(model.state_dict().keys()),
        )

    incompatible = model.load_state_dict(model_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        extra_missing = [
            k for k in incompatible.missing_keys
            if not (k.endswith('coord_x') or k.endswith('coord_y'))
        ]
        if extra_missing or incompatible.unexpected_keys:
            print(
                f"⚠️ Worker {rank}: state_dict mismatch. "
                f"Missing={extra_missing}, Unexpected={incompatible.unexpected_keys}"
            )


def _create_selfplay_engine(model, config, device, num_games, wlog, opponent_model=None):
    rl_cfg = config.get('reinforcement_learning', {})
    use_batch = rl_cfg.get('use_batch_selfplay', False)
    max_batch_games = rl_cfg.get('max_batch_games_per_worker', 256)

    if use_batch:
        configured_max = max_batch_games
        mode_label = "batch"
    else:
        configured_max = 1
        mode_label = "single-via-batch"

    engine = BatchSelfPlayMCTSBatch(model, config, device, configured_max, opponent_model=opponent_model)
    actual_max = min(configured_max, num_games)
    wlog(
        f"Self-play engine: {mode_label} "
        f"(max {configured_max}, actual {actual_max})"
    )
    if getattr(engine, 'dual_model_enabled', False):
        wlog("Dual-color models: ON (white/black can use different checkpoints)")
    return engine


def _play_games_with_engine(rank, engine, config, num_games, result_file_path, wlog):
    rl_cfg = config.get('reinforcement_learning', {})
    progress_file_path = result_file_path.replace('.pkl', '.progress')

    def _write_progress(n):
        try:
            with open(progress_file_path, 'w') as _pf:
                _pf.write(str(n))
        except Exception:
            pass

    if hasattr(engine, '_progress_file'):
        engine._progress_file = progress_file_path

    wlog(
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
    total_dropped_positions = 0
    total_truncated_games = 0
    total_claimable_draw_ended_games = 0
    total_completed_length_sum = 0
    total_truncated_length_sum = 0

    if save_every and save_every > 0:
        with open(result_file_path, 'wb') as f:
            pass

        games_left = num_games
        while games_left > 0:
            chunk_games = min(save_every, games_left)
            if hasattr(engine, '_progress_file'):
                engine._progress_file = progress_file_path
            if hasattr(engine, '_progress_base'):
                engine._progress_base = total_games
            positions, game_lengths = engine.play_games(chunk_games)
            stats = getattr(engine, 'last_selfplay_stats', {}) or {}

            with open(result_file_path, 'ab') as f:
                pickle.dump((positions, game_lengths, stats), f)

            total_positions += len(positions)
            total_games += len(game_lengths)
            total_dropped_positions += int(stats.get('dropped_positions', 0))
            total_truncated_games += int(stats.get('truncated_games', 0))
            total_claimable_draw_ended_games += int(stats.get('claimable_draw_ended_games', 0))
            total_completed_length_sum += int(stats.get('completed_length_sum', 0))
            total_truncated_length_sum += int(stats.get('truncated_length_sum', 0))
            _write_progress(total_games)
            games_left -= chunk_games
    else:
        if hasattr(engine, '_progress_file'):
            engine._progress_file = progress_file_path
        positions, game_lengths = engine.play_games(num_games)
        stats = getattr(engine, 'last_selfplay_stats', {}) or {}
        total_positions = len(positions)
        total_games = len(game_lengths)
        total_dropped_positions = int(stats.get('dropped_positions', 0))
        total_truncated_games = int(stats.get('truncated_games', 0))
        total_claimable_draw_ended_games = int(stats.get('claimable_draw_ended_games', 0))
        total_completed_length_sum = int(stats.get('completed_length_sum', 0))
        total_truncated_length_sum = int(stats.get('truncated_length_sum', 0))
        _write_progress(total_games)

        with open(result_file_path, 'wb') as f:
            pickle.dump((positions, game_lengths, stats), f)

    wlog(f"Generated {total_positions} positions from {total_games} games")
    if total_truncated_games > 0 or total_dropped_positions > 0:
        wlog(
            "Dropped due to truncation: "
            f"{total_dropped_positions} positions across {total_truncated_games} games"
        )
    if total_claimable_draw_ended_games > 0:
        wlog(f"Claimable-draw ended games: {total_claimable_draw_ended_games}")
    if total_games > 0:
        completed_games = max(0, total_games - total_truncated_games)
        if completed_games > 0:
            wlog(
                f"Avg completed game length: {total_completed_length_sum / completed_games:.1f}"
            )
        if total_truncated_games > 0:
            wlog(
                f"Avg truncated game length: {total_truncated_length_sum / total_truncated_games:.1f}"
            )
    wlog(f"Saved to {result_file_path}")
    return total_positions, total_games


def persistent_selfplay_worker(rank, config, device_id, task_queue, result_queue):
    """
    Persistent worker for self-play on Windows.

    The process is spawned once, then receives lightweight play tasks so we avoid
    repeated interpreter startup, imports, model construction and argument pickling.
    """
    rl_cfg = config.get('reinforcement_learning', {})
    worker_verbose = bool(rl_cfg.get('self_play_worker_verbose', False))
    wlog = _build_worker_logger(rank, worker_verbose)

    try:
        device = _configure_selfplay_worker_runtime(config, device_id)
        wlog(f"Persistent worker started on {device}")

        model = _build_selfplay_worker_model(config, device)
        opponent_model = _build_selfplay_worker_model(config, device)
        inference_model = model
        engine = None
        engine_batch_mode = None

        while True:
            task = task_queue.get()
            if task is None or task.get('cmd') == 'stop':
                wlog("Stopping persistent worker")
                break

            result_file_path = task['result_file_path']
            try:
                model_state = torch.load(task['model_state_path'], map_location='cpu')
                _load_worker_model_state(model, model_state, rank)
                model.eval()

                opponent_state_path = task.get('opponent_model_state_path', None)
                if opponent_state_path:
                    opponent_state = torch.load(opponent_state_path, map_location='cpu')
                    _load_worker_model_state(opponent_model, opponent_state, rank)
                    opponent_model.eval()
                    inference_opponent_model = opponent_model
                else:
                    inference_opponent_model = None

                current_batch_mode = bool(rl_cfg.get('use_batch_selfplay', False))
                if (
                    engine is None
                    or engine_batch_mode != current_batch_mode
                    or bool(getattr(engine, 'dual_model_enabled', False)) != bool(inference_opponent_model is not None)
                ):
                    engine = _create_selfplay_engine(
                        inference_model,
                        config,
                        device,
                        int(task['num_games']),
                        wlog,
                        opponent_model=inference_opponent_model,
                    )
                    engine_batch_mode = current_batch_mode

                if hasattr(engine, 'temperature') and task.get('mcts_temperature') is not None:
                    engine.temperature = float(task['mcts_temperature'])

                total_positions, total_games = _play_games_with_engine(
                    rank,
                    engine,
                    config,
                    int(task['num_games']),
                    result_file_path,
                    wlog,
                )
                result_queue.put({
                    'rank': rank,
                    'task_id': task['task_id'],
                    'ok': True,
                    'positions': total_positions,
                    'games': total_games,
                })
            except KeyboardInterrupt:
                try:
                    with open(result_file_path, 'wb') as f:
                        pickle.dump(([], []), f)
                finally:
                    result_queue.put({
                        'rank': rank,
                        'task_id': task.get('task_id'),
                        'ok': False,
                        'interrupt': True,
                    })
                    raise SystemExit(130)
            except Exception as e:
                print(f"❌ Worker {rank} failed: {e}")
                import traceback
                traceback.print_exc()
                with open(result_file_path, 'wb') as f:
                    pickle.dump(([], []), f)
                result_queue.put({
                    'rank': rank,
                    'task_id': task.get('task_id'),
                    'ok': False,
                    'error': str(e),
                })
    except KeyboardInterrupt:
        raise SystemExit(130)


def play_games_mcts_worker(rank, model_state, config, device_id, num_games, result_file_path, opponent_model_state=None):
    """
    🚀 One-shot worker function for parallel MCTS self-play.
    """
    rl_cfg = config.get('reinforcement_learning', {})
    worker_verbose = bool(rl_cfg.get('self_play_worker_verbose', False))
    wlog = _build_worker_logger(rank, worker_verbose)

    try:
        device = _configure_selfplay_worker_runtime(config, device_id)
        wlog(f"Starting on {device}")

        if isinstance(model_state, (str, bytes)):
            model_state = torch.load(model_state, map_location='cpu')

        model = _build_selfplay_worker_model(config, device)
        _load_worker_model_state(model, model_state, rank)

        opponent_model = None
        if opponent_model_state is not None:
            if isinstance(opponent_model_state, (str, bytes)):
                opponent_model_state = torch.load(opponent_model_state, map_location='cpu')
            opponent_model = _build_selfplay_worker_model(config, device)
            _load_worker_model_state(opponent_model, opponent_model_state, rank)

        engine = _create_selfplay_engine(
            model,
            config,
            device,
            num_games,
            wlog,
            opponent_model=opponent_model,
        )
        _play_games_with_engine(rank, engine, config, num_games, result_file_path, wlog)
    except KeyboardInterrupt:
        try:
            with open(result_file_path, 'wb') as f:
                pickle.dump(([], []), f)
        finally:
            raise SystemExit(130)
    except Exception as e:
        print(f"❌ Worker {rank} failed: {e}")
        import traceback
        traceback.print_exc()
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


# For backwards compatibility
BatchSelfPlay = BatchSelfPlayMCTSBatch
MCTS = BatchMCTS
