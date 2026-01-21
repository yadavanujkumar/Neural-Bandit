"""
Redis State Store for managing user sessions and model states.
"""
import redis
import json
import numpy as np
from typing import Optional, Dict, Any
import pickle


class RedisStateStore:
    """
    Redis-based state store for Neural-Bandit system.
    Stores user contexts, interaction history, and temporary states.
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        decode_responses: bool = False
    ):
        """
        Initialize Redis connection.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            decode_responses: Whether to decode responses to strings
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=decode_responses
            )
            # Test connection
            self.redis_client.ping()
            self.connected = True
        except (redis.ConnectionError, redis.TimeoutError):
            # Fallback to in-memory storage if Redis is not available
            self.redis_client = None
            self.connected = False
            self.memory_store = {}
    
    def save_user_context(self, user_id: int, context: np.ndarray, ttl: int = 3600):
        """
        Save user's current context.
        
        Args:
            user_id: User ID
            context: Context features
            ttl: Time to live in seconds
        """
        key = f"user_context:{user_id}"
        value = json.dumps(context.tolist())
        
        if self.connected:
            self.redis_client.setex(key, ttl, value)
        else:
            self.memory_store[key] = value
    
    def get_user_context(self, user_id: int) -> Optional[np.ndarray]:
        """
        Retrieve user's context.
        
        Args:
            user_id: User ID
            
        Returns:
            Context array or None if not found
        """
        key = f"user_context:{user_id}"
        
        if self.connected:
            value = self.redis_client.get(key)
        else:
            value = self.memory_store.get(key)
        
        if value:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return np.array(json.loads(value))
        return None
    
    def save_user_interactions(
        self,
        user_id: int,
        interaction: Dict[str, Any],
        max_history: int = 100
    ):
        """
        Save user interaction to history.
        
        Args:
            user_id: User ID
            interaction: Interaction data (item_id, clicked, timestamp, etc.)
            max_history: Maximum number of interactions to keep
        """
        key = f"user_history:{user_id}"
        value = json.dumps(interaction)
        
        if self.connected:
            # Use Redis list (LPUSH for latest first)
            self.redis_client.lpush(key, value)
            self.redis_client.ltrim(key, 0, max_history - 1)
        else:
            if key not in self.memory_store:
                self.memory_store[key] = []
            self.memory_store[key].insert(0, value)
            self.memory_store[key] = self.memory_store[key][:max_history]
    
    def get_user_interactions(
        self,
        user_id: int,
        limit: int = 10
    ) -> list:
        """
        Get user's recent interactions.
        
        Args:
            user_id: User ID
            limit: Number of recent interactions to return
            
        Returns:
            List of interaction dictionaries
        """
        key = f"user_history:{user_id}"
        
        if self.connected:
            values = self.redis_client.lrange(key, 0, limit - 1)
            return [json.loads(v.decode('utf-8') if isinstance(v, bytes) else v) for v in values]
        else:
            values = self.memory_store.get(key, [])
            return [json.loads(v) for v in values[:limit]]
    
    def save_model_state(self, model_name: str, state_dict: dict):
        """
        Save model state (e.g., for checkpointing).
        
        Args:
            model_name: Name/identifier for the model
            state_dict: Model state dictionary
        """
        key = f"model_state:{model_name}"
        value = pickle.dumps(state_dict)
        
        if self.connected:
            self.redis_client.set(key, value)
        else:
            self.memory_store[key] = value
    
    def get_model_state(self, model_name: str) -> Optional[dict]:
        """
        Retrieve model state.
        
        Args:
            model_name: Name/identifier for the model
            
        Returns:
            Model state dictionary or None
        """
        key = f"model_state:{model_name}"
        
        if self.connected:
            value = self.redis_client.get(key)
        else:
            value = self.memory_store.get(key)
        
        if value:
            return pickle.loads(value)
        return None
    
    def increment_metric(self, metric_name: str, amount: int = 1):
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the metric
            amount: Amount to increment by
        """
        key = f"metric:{metric_name}"
        
        if self.connected:
            self.redis_client.incrby(key, amount)
        else:
            current = self.memory_store.get(key, 0)
            self.memory_store[key] = current + amount
    
    def get_metric(self, metric_name: str) -> int:
        """
        Get current value of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Current metric value
        """
        key = f"metric:{metric_name}"
        
        if self.connected:
            value = self.redis_client.get(key)
            return int(value) if value else 0
        else:
            return self.memory_store.get(key, 0)
    
    def clear_all(self):
        """Clear all data (for testing/reset)."""
        if self.connected:
            self.redis_client.flushdb()
        else:
            self.memory_store.clear()
