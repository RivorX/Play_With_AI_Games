import chess
import numpy as np
import math
from src.data import board_to_tensor, move_to_index
import torch


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


class MCTS:
    """Monte Carlo Tree Search"""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.c_puct = config['reinforcement_learning']['mcts_c_puct']
    
    def search(self, board, num_simulations):
        """
        Run MCTS from given board position
        Returns: dict of move -> visit_count
        """
        root = MCTSNode(board)
        
        # Add Dirichlet noise to root for exploration (RL only)
        add_noise = self.config['reinforcement_learning'].get('mcts_dirichlet_weight', 0) > 0
        
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree to leaf
            while not node.is_leaf() and not node.board.is_game_over():
                node = self._select_child(node)
                search_path.append(node)
            
            # Expansion and Evaluation
            value = self._expand_and_evaluate(node, add_noise=(node == root and add_noise))
            
            # Backpropagation
            self._backpropagate(search_path, value)
        
        # Return visit counts for each move
        return {move: child.visit_count for move, child in root.children.items()}
    
    def _select_child(self, node):
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        for move, child in node.children.items():
            score = self._ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_child
    
    def _ucb_score(self, parent, child):
        """Upper Confidence Bound score"""
        # Q(s,a) + U(s,a)
        q_value = child.value()
        u_value = (self.c_puct * child.prior * 
                   math.sqrt(parent.visit_count) / (1 + child.visit_count))
        return q_value + u_value
    
    def _expand_and_evaluate(self, node, add_noise=False):
        """Expand node and evaluate with neural network"""
        if node.board.is_game_over():
            # Terminal node
            result = node.board.result()
            if result == '1-0':
                return 1.0 if node.board.turn == chess.WHITE else -1.0
            elif result == '0-1':
                return -1.0 if node.board.turn == chess.WHITE else 1.0
            else:
                return 0.0
        
        # Get network predictions
        board_tensor = torch.FloatTensor(board_to_tensor(node.board)).unsqueeze(0).to(self.device)
        policy_logits, value = self.model.predict(board_tensor)
        
        # Mask illegal moves
        legal_moves = list(node.board.legal_moves)
        policy = np.zeros(4096)
        
        for move in legal_moves:
            idx = move_to_index(move)
            policy[idx] = policy_logits[idx]
        
        # Normalize
        policy = policy / (policy.sum() + 1e-8)
        
        # Add Dirichlet noise for exploration (at root in self-play)
        if add_noise:
            alpha = self.config['reinforcement_learning']['mcts_dirichlet_alpha']
            weight = self.config['reinforcement_learning']['mcts_dirichlet_weight']
            noise = np.random.dirichlet([alpha] * len(legal_moves))
            
            for i, move in enumerate(legal_moves):
                idx = move_to_index(move)
                policy[idx] = (1 - weight) * policy[idx] + weight * noise[i]
        
        # Expand node
        for move in legal_moves:
            idx = move_to_index(move)
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, parent=node, move=move, prior=policy[idx])
        
        node.expanded = True
        return value
    
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