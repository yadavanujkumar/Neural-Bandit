"""
Stage 1: Stable Candidate Generator using Matrix Factorization (ALS)
This model generates top-50 items for each user based on historical data.
"""
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from typing import List, Dict
import pickle
import os


class ALSCandidateGenerator:
    """
    Matrix Factorization model using ALS algorithm from Implicit library.
    Generates a stable 'safe list' of top-N recommendations per user.
    """
    
    def __init__(
        self,
        factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 15,
        top_n: int = 50
    ):
        """
        Initialize ALS model.
        
        Args:
            factors: Number of latent factors
            regularization: Regularization parameter
            iterations: Number of ALS iterations
            top_n: Number of top items to recommend per user
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        self.top_n = top_n
        self.user_item_matrix = None
        self.is_fitted = False
        
    def fit(self, user_item_matrix: csr_matrix):
        """
        Train the ALS model on user-item interaction matrix.
        
        Args:
            user_item_matrix: Sparse matrix of user-item interactions
        """
        self.user_item_matrix = user_item_matrix
        # Implicit library expects item-user matrix
        self.model.fit(user_item_matrix.T.tocsr())
        self.is_fitted = True
        
    def generate_candidates(self, user_id: int) -> List[int]:
        """
        Generate top-N candidate items for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of item IDs (top-N recommendations)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating recommendations")
        
        # Get recommendations for user
        user_items = self.user_item_matrix[user_id]
        item_ids, scores = self.model.recommend(
            user_id,
            user_items,
            N=self.top_n,
            filter_already_liked_items=True
        )
        
        return item_ids.tolist()
    
    def batch_generate_candidates(self, n_users: int) -> Dict[int, List[int]]:
        """
        Generate top-N candidates for all users (nightly batch job).
        
        Args:
            n_users: Total number of users
            
        Returns:
            Dictionary mapping user_id to list of item_ids
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating recommendations")
        
        candidates = {}
        for user_id in range(n_users):
            try:
                candidates[user_id] = self.generate_candidates(user_id)
            except Exception as e:
                # Handle edge cases (e.g., users with no interactions)
                candidates[user_id] = []
        
        return candidates
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'top_n': self.top_n,
                'is_fitted': self.is_fitted
            }, f)
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.top_n = data['top_n']
            self.is_fitted = data['is_fitted']
