import chess
import numpy as np
import math
from src.utils.data_helpers import board_to_tensor, move_to_index
import torch
from collections import defaultdict
import threading


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False
        
        # 🚀 Virtual Loss for thread-safe parallel MCTS
        self.virtual_loss = 0
        self.lock = threading.Lock()
    
    def value(self):
        """Average value with virtual loss"""
        if self.visit_count == 0:
            return 0
        return (self.value_sum - self.virtual_loss) / (self.visit_count + self.virtual_loss)
    
    def is_leaf(self):
        return not self.expanded
    
    def add_virtual_loss(self, n=1):
        """Add virtual loss for thread safety"""
        with self.lock:
            self.virtual_loss += n
    
    def remove_virtual_loss(self, n=1):
        """Remove virtual loss after backup"""
        with self.lock:
            self.virtual_loss -= n


class BatchMCTS:
    """
    🚀 Optimized Batch MCTS with:
    - Tree reuse between moves
    - Virtual loss for parallel search
    - Efficient batch evaluation
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.c_puct = config['reinforcement_learning']['mcts_c_puct']
        self.eval_batch_size = config['reinforcement_learning'].get('mcts_batch_size', 32)
        
        # 🚀 Tree reuse
        self.root = None
        self.reuse_tree = config['reinforcement_learning'].get('mcts_reuse_tree', True)
    
    def search(self, board, num_simulations, temperature=1.0):
        """
        Run MCTS with tree reuse and batch evaluation
        """
        # 🚀 Tree reuse: if root exists and matches board, reuse it
        if self.reuse_tree and self.root is not None:
            # Try to find current position in existing tree
            for move, child in self.root.children.items():
                if child.board.fen() == board.fen():
                    # Found it! Reuse this subtree
                    self.root = child
                    self.root.parent = None  # Make it new root
                    break
            else:
                # Position not found, create new tree
                self.root = MCTSNode(board)
        else:
            self.root = MCTSNode(board)
        
        # Add Dirichlet noise to root
        add_noise = self.config['reinforcement_learning'].get('mcts_dirichlet_weight', 0) > 0
        
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
                while not node.is_leaf() and not node.board.is_game_over():
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
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_child = None
        
        for move, child in node.children.items():
            score = self._ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _ucb_score(self, parent, child):
        """Upper Confidence Bound with virtual loss"""
        q_value = child.value()  # Already includes virtual loss
        u_value = (self.c_puct * child.prior * 
                   math.sqrt(parent.visit_count + parent.virtual_loss) / 
                   (1 + child.visit_count + child.virtual_loss))
        return q_value + u_value
    
    def _batch_expand_and_evaluate(self, nodes, add_noise=False):
        """Batch expansion and evaluation"""
        terminal_values = []
        non_terminal_indices = []
        non_terminal_nodes = []
        
        for i, node in enumerate(nodes):
            if node.board.is_game_over():
                result = node.board.result()
                if result == '1-0':
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif result == '0-1':
                    value = -1.0 if node.board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0
                terminal_values.append((i, value))
            else:
                non_terminal_indices.append(i)
                non_terminal_nodes.append(node)
        
        all_values = [None] * len(nodes)
        
        if non_terminal_nodes:
            # Stack tensors
            board_tensors = torch.stack([
                torch.FloatTensor(board_to_tensor(node.board))
                for node in non_terminal_nodes
            ]).to(self.device)
            
            # Single GPU call
            with torch.no_grad():
                policy_logits_batch, values_batch = self.model(board_tensors)
            
            # Process results
            for idx, node in enumerate(non_terminal_nodes):
                policy_logits = policy_logits_batch[idx].cpu().numpy()
                value = values_batch[idx].cpu().item()
                
                legal_moves = list(node.board.legal_moves)
                policy = np.zeros(4096)
                
                for move in legal_moves:
                    move_idx = move_to_index(move)
                    policy[move_idx] = np.exp(policy_logits[move_idx])
                
                policy = policy / (policy.sum() + 1e-8)
                
                # Dirichlet noise
                if add_noise and len(legal_moves) > 0:
                    alpha = self.config['reinforcement_learning']['mcts_dirichlet_alpha']
                    weight = self.config['reinforcement_learning']['mcts_dirichlet_weight']
                    noise = np.random.dirichlet([alpha] * len(legal_moves))
                    
                    for i, move in enumerate(legal_moves):
                        move_idx = move_to_index(move)
                        policy[move_idx] = (1 - weight) * policy[move_idx] + weight * noise[i]
                
                # Create children
                for move in legal_moves:
                    move_idx = move_to_index(move)
                    child_board = node.board.copy()
                    child_board.push(move)
                    node.children[move] = MCTSNode(
                        child_board,
                        parent=node,
                        move=move,
                        prior=policy[move_idx]
                    )
                
                node.expanded = True
                all_values[non_terminal_indices[idx]] = value
        
        for idx, value in terminal_values:
            all_values[idx] = value
        
        return all_values
    
    def _backpropagate(self, search_path, value):
        """Backpropagate value"""
        for node in reversed(search_path):
            node.value_sum += value if node.board.turn == chess.WHITE else -value
            node.visit_count += 1
            value = -value
    
    def reset_tree(self):
        """Reset tree (call after game ends)"""
        self.root = None


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