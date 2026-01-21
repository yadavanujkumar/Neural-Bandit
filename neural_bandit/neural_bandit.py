"""
Neural-Bandit: Integrated Recommendation System
Combines Stage 1 (ALS) with Stage 2 (DQN) and Exploration Controller
"""
import numpy as np
from typing import List, Tuple, Optional
from neural_bandit.stage1.als_model import ALSCandidateGenerator
from neural_bandit.stage2.dqn_reranker import DQNReranker
from neural_bandit.exploration.neural_ucb import ExploreExploitController
from neural_bandit.utils.data_generator import generate_context_features


class NeuralBandit:
    """
    Complete Neural-Bandit recommendation system.
    
    Pipeline:
    1. Stage 1 (ALS) generates top-50 candidate items
    2. Stage 2 (DQN) re-ranks based on context
    3. Exploration controller adds exploration items
    """
    
    def __init__(
        self,
        als_model: ALSCandidateGenerator,
        dqn_reranker: DQNReranker,
        exploration_controller: ExploreExploitController,
        enable_dqn: bool = True,
        enable_exploration: bool = True
    ):
        """
        Initialize Neural-Bandit system.
        
        Args:
            als_model: Trained ALS candidate generator
            dqn_reranker: DQN re-ranker
            exploration_controller: Exploration-exploitation controller
            enable_dqn: Whether to use DQN re-ranking (False = pure ALS)
            enable_exploration: Whether to add exploration items
        """
        self.als_model = als_model
        self.dqn_reranker = dqn_reranker
        self.exploration_controller = exploration_controller
        self.enable_dqn = enable_dqn
        self.enable_exploration = enable_exploration
        
        # Metrics tracking
        self.total_recommendations = 0
        self.total_clicks = 0
        self.stage_metrics = {
            'als_only_clicks': 0,
            'als_only_shown': 0,
            'neural_bandit_clicks': 0,
            'neural_bandit_shown': 0
        }
    
    def get_recommendations(
        self,
        user_id: int,
        context: Optional[np.ndarray] = None,
        top_k: int = 10
    ) -> Tuple[List[int], dict]:
        """
        Get top-k recommendations for a user.
        
        Args:
            user_id: User ID
            context: Context features (if None, will be generated)
            top_k: Number of recommendations to return
            
        Returns:
            Tuple of (recommended_items, metadata)
        """
        # Stage 1: Generate candidates using ALS
        candidate_items = self.als_model.generate_candidates(user_id)
        
        metadata = {
            'user_id': user_id,
            'stage1_candidates': len(candidate_items),
            'used_dqn': False,
            'used_exploration': False,
            'exploration_items': []
        }
        
        # If no candidates, return empty
        if len(candidate_items) == 0:
            return [], metadata
        
        # Generate context if not provided
        if context is None:
            context = generate_context_features(batch_size=1)[0]
        
        # Stage 2: Re-rank with DQN (if enabled)
        if self.enable_dqn:
            reranked_items = self.dqn_reranker.rerank(candidate_items, context)
            metadata['used_dqn'] = True
        else:
            reranked_items = candidate_items
        
        # Stage 3: Apply exploration-exploitation (if enabled)
        if self.enable_exploration:
            final_items, is_exploration = self.exploration_controller.select_recommendations(
                reranked_items,
                top_k=top_k
            )
            metadata['used_exploration'] = True
            metadata['exploration_items'] = [
                item for item, is_exp in zip(final_items, is_exploration) if is_exp
            ]
        else:
            final_items = reranked_items[:top_k]
        
        self.total_recommendations += len(final_items)
        
        return final_items, metadata
    
    def record_click(
        self,
        user_id: int,
        item_id: int,
        context: np.ndarray,
        clicked: bool
    ):
        """
        Record user interaction and update models.
        
        Args:
            user_id: User ID
            item_id: Item that was shown
            context: Context at time of recommendation
            clicked: Whether user clicked
        """
        # Update exploration controller
        self.exploration_controller.record_interaction(item_id, clicked)
        
        # Update DQN if enabled
        if self.enable_dqn:
            reward = 1.0 if clicked else 0.0
            # Generate next context (simplified)
            next_context = generate_context_features(batch_size=1)[0]
            self.dqn_reranker.update(context, item_id, reward, next_context, done=True)
        
        # Update metrics
        if clicked:
            self.total_clicks += 1
            if self.enable_dqn:
                self.stage_metrics['neural_bandit_clicks'] += 1
            else:
                self.stage_metrics['als_only_clicks'] += 1
        
        if self.enable_dqn:
            self.stage_metrics['neural_bandit_shown'] += 1
        else:
            self.stage_metrics['als_only_shown'] += 1
    
    def get_ctr(self) -> float:
        """Calculate overall Click-Through Rate."""
        if self.total_recommendations == 0:
            return 0.0
        return self.total_clicks / self.total_recommendations
    
    def get_stage_ctrs(self) -> dict:
        """Calculate CTRs for each stage/configuration."""
        als_ctr = (
            self.stage_metrics['als_only_clicks'] / self.stage_metrics['als_only_shown']
            if self.stage_metrics['als_only_shown'] > 0 else 0.0
        )
        neural_bandit_ctr = (
            self.stage_metrics['neural_bandit_clicks'] / self.stage_metrics['neural_bandit_shown']
            if self.stage_metrics['neural_bandit_shown'] > 0 else 0.0
        )
        
        return {
            'als_only_ctr': als_ctr,
            'neural_bandit_ctr': neural_bandit_ctr
        }
