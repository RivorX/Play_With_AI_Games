import chess
import numpy as np
import math
from src.data import board_to_tensor, move_to_index
import torch
from collections import defaultdict


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # Prior probability from network
        
        self.children = {}  # Dict of move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False
    
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def is_leaf(self):
        """Check if node is a leaf (not expanded)"""
        return not self.expanded


class BatchMCTS:
    """
    🚀 Batch MCTS - evaluates multiple positions in parallel
    Key optimization: batch neural network calls instead of sequential
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.c_puct = config['reinforcement_learning']['mcts_c_puct']
        
        # Batch evaluation cache
        self.eval_batch_size = config['reinforcement_learning'].get('mcts_batch_size', 32)
    
    def search(self, board, num_simulations):
        """
        Run MCTS from given board position with batch evaluation
        Returns: dict of move -> visit_count
        """
        root = MCTSNode(board)
        
        # Add Dirichlet noise to root for exploration (RL only)
        add_noise = self.config['reinforcement_learning'].get('mcts_dirichlet_weight', 0) > 0
        
        # Batch simulations for better GPU utilization
        for batch_start in range(0, num_simulations, self.eval_batch_size):
            batch_size = min(self.eval_batch_size, num_simulations - batch_start)
            
            # Collect leaf nodes to evaluate in batch
            search_paths = []
            leaf_nodes = []
            
            for _ in range(batch_size):
                node = root
                search_path = [node]
                
                # Selection: traverse tree to leaf
                while not node.is_leaf() and not node.board.is_game_over():
                    node = self._select_child(node)
                    search_path.append(node)
                
                search_paths.append(search_path)
                leaf_nodes.append(node)
            
            # Batch expansion and evaluation
            values = self._batch_expand_and_evaluate(
                leaf_nodes, 
                add_noise=(leaf_nodes[0] == root and add_noise)
            )
            
            # Backpropagation
            for search_path, value in zip(search_paths, values):
                self._backpropagate(search_path, value)
        
        # Return visit counts for each move
        return {move: child.visit_count for move, child in root.children.items()}
    
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
        """Upper Confidence Bound score"""
        # Q(s,a) + U(s,a)
        q_value = child.value()
        u_value = (self.c_puct * child.prior * 
                   math.sqrt(parent.visit_count) / (1 + child.visit_count))
        return q_value + u_value
    
    def _batch_expand_and_evaluate(self, nodes, add_noise=False):
        """
        🚀 KEY OPTIMIZATION: Batch expansion and evaluation
        Evaluates multiple positions in one GPU call
        """
        # Separate terminal and non-terminal nodes
        terminal_values = []
        non_terminal_indices = []
        non_terminal_nodes = []
        
        for i, node in enumerate(nodes):
            if node.board.is_game_over():
                # Terminal node - compute value directly
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
        
        # Batch evaluate non-terminal nodes
        all_values = [None] * len(nodes)
        
        if non_terminal_nodes:
            # Stack all board tensors into a batch
            board_tensors = torch.stack([
                torch.FloatTensor(board_to_tensor(node.board))
                for node in non_terminal_nodes
            ]).to(self.device)
            
            # Single GPU call for entire batch! 🚀
            with torch.no_grad():
                policy_logits_batch, values_batch = self.model(board_tensors)
            
            # Process each node's results
            for idx, node in enumerate(non_terminal_nodes):
                policy_logits = policy_logits_batch[idx].cpu().numpy()
                value = values_batch[idx].cpu().item()
                
                # Expand node
                legal_moves = list(node.board.legal_moves)
                policy = np.zeros(4096)
                
                for move in legal_moves:
                    move_idx = move_to_index(move)
                    policy[move_idx] = np.exp(policy_logits[move_idx])
                
                # Normalize
                policy = policy / (policy.sum() + 1e-8)
                
                # Add Dirichlet noise for exploration (at root)
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
                
                # Store value for this node
                original_idx = non_terminal_indices[idx]
                all_values[original_idx] = value
        
        # Add terminal values
        for idx, value in terminal_values:
            all_values[idx] = value
        
        return all_values
    
    def _backpropagate(self, search_path, value):
        """Backpropagate value up the tree"""
        for node in reversed(search_path):
            node.value_sum += value if node.board.turn == chess.WHITE else -value
            node.visit_count += 1
            value = -value  # Flip value for opponent


def select_move_by_visits(visit_counts, temperature=1.0):
    """
    Select move based on visit counts with temperature
    temperature=0: argmax (greedy)
    temperature=1: sample proportional to visits
    """
    moves = list(visit_counts.keys())
    visits = np.array([visit_counts[m] for m in moves])
    
    if temperature == 0:
        # Greedy
        best_idx = np.argmax(visits)
        return moves[best_idx], visits
    else:
        # Sample with temperature
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx], probs


# Backwards compatibility - alias old class name
MCTS = BatchMCTS