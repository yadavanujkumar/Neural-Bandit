"""
Stage 2: Dynamic Re-Ranker using Deep Q-Network (DQN)
This model re-ranks the top-50 items from Stage 1 based on real-time context.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple
import os


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for learning item ranking based on context.
    
    Input: Context features (5D) + Item embedding (10D)
    Output: Q-value for the item in this context
    """
    
    def __init__(self, context_dim: int = 5, item_embedding_dim: int = 10):
        super(DQNNetwork, self).__init__()
        
        input_dim = context_dim + item_embedding_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single Q-value output
        )
        
    def forward(self, context: torch.Tensor, item_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            context: Context features (batch_size, context_dim)
            item_embedding: Item embeddings (batch_size, item_embedding_dim)
            
        Returns:
            Q-values (batch_size, 1)
        """
        x = torch.cat([context, item_embedding], dim=1)
        return self.network(x)


class DQNReranker:
    """
    DQN-based re-ranker that learns to order items based on context.
    """
    
    def __init__(
        self,
        n_items: int,
        context_dim: int = 5,
        item_embedding_dim: int = 10,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        buffer_size: int = 10000
    ):
        """
        Initialize DQN Re-ranker.
        
        Args:
            n_items: Total number of items in catalog
            context_dim: Dimension of context features
            item_embedding_dim: Dimension of item embeddings
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            buffer_size: Size of replay buffer
        """
        self.n_items = n_items
        self.context_dim = context_dim
        self.item_embedding_dim = item_embedding_dim
        self.gamma = gamma
        
        # Initialize networks
        self.q_network = DQNNetwork(context_dim, item_embedding_dim)
        self.target_network = DQNNetwork(context_dim, item_embedding_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Item embeddings (randomly initialized, can be replaced with learned embeddings)
        self.item_embeddings = nn.Embedding(n_items, item_embedding_dim)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        
        # Training metrics
        self.training_losses = []
        
    def rerank(self, candidate_items: List[int], context: np.ndarray) -> List[int]:
        """
        Re-rank candidate items based on current context.
        
        Args:
            candidate_items: List of item IDs from Stage 1
            context: Context features (1D array)
            
        Returns:
            Re-ranked list of item IDs
        """
        if len(candidate_items) == 0:
            return []
        
        self.q_network.eval()
        with torch.no_grad():
            # Prepare inputs
            context_tensor = torch.FloatTensor(context).unsqueeze(0).repeat(len(candidate_items), 1)
            item_ids = torch.LongTensor(candidate_items)
            item_embeddings = self.item_embeddings(item_ids)
            
            # Get Q-values
            q_values = self.q_network(context_tensor, item_embeddings)
            
            # Sort by Q-values (descending)
            sorted_indices = torch.argsort(q_values.squeeze(), descending=True)
            reranked_items = [candidate_items[idx] for idx in sorted_indices.tolist()]
        
        return reranked_items
    
    def update(
        self,
        context: np.ndarray,
        item_id: int,
        reward: float,
        next_context: np.ndarray,
        done: bool
    ):
        """
        Update the DQN based on observed reward.
        
        Args:
            context: Current context
            item_id: Selected item
            reward: Observed reward (1 for click, 0 for no click)
            next_context: Next context
            done: Whether episode is done
        """
        # Store experience in replay buffer
        self.replay_buffer.append((context, item_id, reward, next_context, done))
        
        # Train if buffer has enough samples
        if len(self.replay_buffer) >= 32:
            self._train_step()
    
    def _train_step(self, batch_size: int = 32):
        """Perform one training step using experience replay."""
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        contexts, item_ids, rewards, next_contexts, dones = zip(*batch)
        
        # Convert to tensors
        contexts_tensor = torch.FloatTensor(np.array(contexts))
        item_ids_tensor = torch.LongTensor(item_ids)
        rewards_tensor = torch.FloatTensor(rewards)
        next_contexts_tensor = torch.FloatTensor(np.array(next_contexts))
        dones_tensor = torch.FloatTensor(dones)
        
        # Get item embeddings
        item_embeddings = self.item_embeddings(item_ids_tensor)
        
        # Compute current Q values
        self.q_network.train()
        current_q = self.q_network(contexts_tensor, item_embeddings).squeeze()
        
        # Compute target Q values (simplified: no next state for bandit)
        with torch.no_grad():
            target_q = rewards_tensor
        
        # Compute loss and update
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        
        # Update target network periodically
        if len(self.training_losses) % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        dirpath = os.path.dirname(filepath)
        if dirpath:  # Only create directory if path contains a directory component
            os.makedirs(dirpath, exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'item_embeddings_state_dict': self.item_embeddings.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        checkpoint = torch.load(filepath, weights_only=True)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.item_embeddings.load_state_dict(checkpoint['item_embeddings_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint['training_losses']
