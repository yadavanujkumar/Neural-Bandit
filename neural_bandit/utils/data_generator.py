"""
Utility module for generating synthetic user-item interaction data
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple, List
import random


def generate_synthetic_data(
    n_users: int = 1000,
    n_items: int = 500,
    n_interactions: int = 50000,
    seed: int = 42
) -> Tuple[csr_matrix, pd.DataFrame]:
    """
    Generate synthetic user-item interaction data for recommendation system.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        n_interactions: Number of interactions to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sparse_matrix, interaction_dataframe)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate interactions with power-law distribution
    # Popular items get more interactions
    item_popularity = np.random.power(0.5, n_items)
    item_probabilities = item_popularity / item_popularity.sum()
    
    # Generate user-item pairs
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.choice(n_items, n_interactions, p=item_probabilities)
    
    # Implicit feedback (binary: 1 for interaction)
    data = np.ones(n_interactions)
    
    # Create sparse matrix
    sparse_matrix = csr_matrix(
        (data, (users, items)),
        shape=(n_users, n_items)
    )
    
    # Create dataframe for easy manipulation
    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'interaction': 1
    })
    
    return sparse_matrix, df


def generate_context_features(batch_size: int = 1, seed: int = None) -> np.ndarray:
    """
    Generate context features for current session.
    
    Features:
    - Time of day (0-23, normalized)
    - Device type (0: mobile, 1: desktop, 2: tablet)
    - Last 3 clicks (item IDs)
    
    Args:
        batch_size: Number of context samples to generate
        seed: Random seed
        
    Returns:
        Context feature array of shape (batch_size, 5)
    """
    if seed is not None:
        np.random.seed(seed)
    
    contexts = []
    for _ in range(batch_size):
        time_of_day = np.random.randint(0, 24) / 23.0  # Normalized
        device_type = np.random.randint(0, 3) / 2.0  # Normalized
        last_clicks = np.random.rand(3)  # Simplified: random values for last 3 clicks
        
        context = np.array([time_of_day, device_type] + list(last_clicks))
        contexts.append(context)
    
    return np.array(contexts)


def get_user_item_map(df: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Create mappings between original IDs and matrix indices.
    
    Args:
        df: DataFrame with user_id and item_id columns
        
    Returns:
        Tuple of (user_to_idx, item_to_idx) dictionaries
    """
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(sorted(unique_users))}
    item_to_idx = {item: idx for idx, item in enumerate(sorted(unique_items))}
    
    return user_to_idx, item_to_idx
