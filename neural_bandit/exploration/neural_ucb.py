"""
Explore-Exploit Controller using Neural Upper Confidence Bound (NeuralUCB)
This module decides when to explore (show random items) vs exploit (show best predictions).
"""
import numpy as np
import torch
from typing import List, Tuple
import random


class NeuralUCB:
    """
    Neural Upper Confidence Bound algorithm for calculating potential rewards.
    Balances exploration (uncertain items) with exploitation (high-value items).
    """
    
    def __init__(
        self,
        n_items: int,
        exploration_rate: float = 0.1,
        ucb_coefficient: float = 2.0
    ):
        """
        Initialize NeuralUCB controller.
        
        Args:
            n_items: Total number of items
            exploration_rate: Percentage of recommendations to be exploration items (0.1 = 10%)
            ucb_coefficient: Confidence bound coefficient for UCB calculation
        """
        self.n_items = n_items
        self.exploration_rate = exploration_rate
        self.ucb_coefficient = ucb_coefficient
        
        # Track item statistics
        self.item_counts = np.zeros(n_items)  # Number of times each item was shown
        self.item_rewards = np.zeros(n_items)  # Cumulative rewards per item
        self.total_interactions = 0
        
    def select_items(
        self,
        candidate_items: List[int],
        q_values: np.ndarray = None,
        top_k: int = 10
    ) -> Tuple[List[int], List[bool]]:
        """
        Select top-k items with exploration-exploitation strategy.
        
        Args:
            candidate_items: Ranked list of candidate items
            q_values: Optional Q-values for each candidate (if None, use ranking order)
            top_k: Number of items to return
            
        Returns:
            Tuple of (selected_items, is_exploration_flags)
        """
        if len(candidate_items) == 0:
            return [], []
        
        # Determine number of exploration items
        n_explore = int(top_k * self.exploration_rate)
        n_exploit = top_k - n_explore
        
        # Exploitation: Take top items
        exploit_items = candidate_items[:min(n_exploit, len(candidate_items))]
        exploit_flags = [False] * len(exploit_items)
        
        # Exploration: Random new items not in candidate list
        all_items = set(range(self.n_items))
        candidate_set = set(candidate_items)
        exploration_pool = list(all_items - candidate_set)
        
        if len(exploration_pool) > 0:
            explore_items = random.sample(
                exploration_pool,
                min(n_explore, len(exploration_pool))
            )
        else:
            explore_items = []
        
        explore_flags = [True] * len(explore_items)
        
        # Combine and shuffle to avoid position bias
        selected_items = exploit_items + explore_items
        is_exploration = exploit_flags + explore_flags
        
        # Shuffle together
        combined = list(zip(selected_items, is_exploration))
        random.shuffle(combined)
        selected_items, is_exploration = zip(*combined) if combined else ([], [])
        
        return list(selected_items), list(is_exploration)
    
    def calculate_ucb_scores(
        self,
        item_ids: List[int],
        predicted_rewards: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Upper Confidence Bound scores for items.
        
        UCB = predicted_reward + coefficient * sqrt(log(total) / count)
        
        Args:
            item_ids: List of item IDs
            predicted_rewards: Predicted rewards (e.g., Q-values)
            
        Returns:
            UCB scores for each item
        """
        ucb_scores = []
        for item_id, pred_reward in zip(item_ids, predicted_rewards):
            count = max(self.item_counts[item_id], 1)  # Avoid division by zero
            
            # UCB formula
            exploration_bonus = self.ucb_coefficient * np.sqrt(
                np.log(max(self.total_interactions, 1)) / count
            )
            
            ucb_score = pred_reward + exploration_bonus
            ucb_scores.append(ucb_score)
        
        return np.array(ucb_scores)
    
    def update_statistics(self, item_id: int, reward: float):
        """
        Update item statistics after observing reward.
        
        Args:
            item_id: Item that was shown
            reward: Observed reward (1 for click, 0 for no click)
        """
        self.item_counts[item_id] += 1
        self.item_rewards[item_id] += reward
        self.total_interactions += 1
    
    def get_item_metrics(self, item_id: int) -> dict:
        """
        Get metrics for a specific item.
        
        Args:
            item_id: Item ID
            
        Returns:
            Dictionary with count, total_reward, and avg_reward
        """
        count = self.item_counts[item_id]
        total_reward = self.item_rewards[item_id]
        avg_reward = total_reward / count if count > 0 else 0.0
        
        return {
            'count': int(count),
            'total_reward': float(total_reward),
            'avg_reward': float(avg_reward)
        }


class ExploreExploitController:
    """
    High-level controller for exploration-exploitation strategy.
    Integrates NeuralUCB with the recommendation pipeline.
    """
    
    def __init__(
        self,
        n_items: int,
        exploration_rate: float = 0.1,
        use_ucb: bool = True
    ):
        """
        Initialize controller.
        
        Args:
            n_items: Total number of items
            exploration_rate: Exploration rate (0.1 = 10%)
            use_ucb: Whether to use UCB for item selection
        """
        self.neural_ucb = NeuralUCB(
            n_items=n_items,
            exploration_rate=exploration_rate
        )
        self.use_ucb = use_ucb
    
    def select_recommendations(
        self,
        candidate_items: List[int],
        q_values: np.ndarray = None,
        top_k: int = 10
    ) -> Tuple[List[int], List[bool]]:
        """
        Select final recommendations with exploration.
        
        Args:
            candidate_items: Ranked candidate items
            q_values: Q-values from DQN
            top_k: Number of recommendations to return
            
        Returns:
            Tuple of (selected_items, is_exploration_flags)
        """
        return self.neural_ucb.select_items(candidate_items, q_values, top_k)
    
    def record_interaction(self, item_id: int, clicked: bool):
        """
        Record user interaction (click or no-click).
        
        Args:
            item_id: Item that was shown
            clicked: Whether user clicked on the item
        """
        reward = 1.0 if clicked else 0.0
        self.neural_ucb.update_statistics(item_id, reward)
