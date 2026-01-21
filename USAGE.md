# Neural-Bandit Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yadavanujkumar/Neural-Bandit.git
cd Neural-Bandit

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo

Test the system with a command-line demo:

```bash
python demo.py
```

This will:
- Generate synthetic user-item interaction data
- Train the Matrix Factorization (ALS) model
- Initialize the DQN re-ranker
- Run 100 simulated interactions
- Show CTR metrics for both baseline and Neural-Bandit

### 3. Launch the Dashboard

Experience the full A/B testing dashboard:

```bash
streamlit run dashboard.py
```

Then:
1. Click "Initialize & Train Models" in the sidebar
2. Run simulations (1 step, 10 steps, or 100 steps)
3. Watch the learning curve evolve
4. Compare CTR between baseline and Neural-Bandit

## System Architecture

### Stage 1: Matrix Factorization (ALS)

The first stage generates a "safe list" of top-50 candidate items per user using Alternating Least Squares:

```python
from neural_bandit.stage1.als_model import ALSCandidateGenerator

# Initialize and train
als_model = ALSCandidateGenerator(
    factors=50,           # Number of latent factors
    regularization=0.01,  # L2 regularization
    iterations=15,        # ALS iterations
    top_n=50             # Number of candidates to generate
)
als_model.fit(user_item_matrix)

# Generate candidates
candidates = als_model.generate_candidates(user_id)
```

### Stage 2: Deep Q-Network Re-Ranker

The DQN re-ranks Stage 1 candidates based on real-time context:

```python
from neural_bandit.stage2.dqn_reranker import DQNReranker

# Initialize DQN
dqn_reranker = DQNReranker(
    n_items=500,
    context_dim=5,           # Time, device, last 3 clicks
    item_embedding_dim=10,
    learning_rate=0.001
)

# Re-rank candidates with context
context = generate_context_features(batch_size=1)[0]
reranked_items = dqn_reranker.rerank(candidates, context)

# Update based on user feedback
dqn_reranker.update(context, item_id, reward, next_context, done=True)
```

### Exploration Controller: NeuralUCB

Balances exploitation (show best items) with exploration (discover new items):

```python
from neural_bandit.exploration.neural_ucb import ExploreExploitController

# Initialize controller
controller = ExploreExploitController(
    n_items=500,
    exploration_rate=0.1  # 10% exploration
)

# Select items with exploration
selected_items, is_exploration = controller.select_recommendations(
    candidate_items=reranked_items,
    top_k=10
)

# Record user interaction
controller.record_interaction(item_id, clicked=True)
```

### Complete System Integration

The Neural-Bandit class integrates all components:

```python
from neural_bandit.neural_bandit import NeuralBandit

# Create integrated system
neural_bandit = NeuralBandit(
    als_model=als_model,
    dqn_reranker=dqn_reranker,
    exploration_controller=controller,
    enable_dqn=True,
    enable_exploration=True
)

# Get recommendations
recommendations, metadata = neural_bandit.get_recommendations(
    user_id=user_id,
    context=context,
    top_k=10
)

# Record click
neural_bandit.record_click(user_id, item_id, context, clicked=True)
```

## Context Features

The system uses 5 context features:

1. **Time of Day** (normalized 0-1): `hour / 23`
2. **Device Type** (normalized 0-1): Mobile=0, Desktop=0.5, Tablet=1
3. **Last Click 1** (normalized): Most recent item interaction
4. **Last Click 2** (normalized): Second most recent interaction
5. **Last Click 3** (normalized): Third most recent interaction

```python
from neural_bandit.utils.data_generator import generate_context_features

# Generate context
context = generate_context_features(batch_size=1)[0]
# Returns: [time, device, click1, click2, click3]
```

## Data Generation

For testing and development, use synthetic data:

```python
from neural_bandit.utils.data_generator import generate_synthetic_data

# Generate synthetic user-item interactions
sparse_matrix, df = generate_synthetic_data(
    n_users=1000,
    n_items=500,
    n_interactions=50000,
    seed=42
)
```

## Redis State Store (Optional)

For production, use Redis to store session state:

```python
from neural_bandit.utils.redis_store import RedisStateStore

# Initialize Redis store (falls back to in-memory if Redis unavailable)
store = RedisStateStore(host='localhost', port=6379)

# Save user context
store.save_user_context(user_id, context)

# Get user context
context = store.get_user_context(user_id)

# Save interaction history
store.save_user_interactions(user_id, {
    'item_id': 123,
    'clicked': True,
    'timestamp': '2024-01-01T12:00:00'
})

# Get recent interactions
history = store.get_user_interactions(user_id, limit=10)
```

## Model Persistence

Save and load trained models:

```python
# Save ALS model
als_model.save_model('models/als_model.pkl')

# Load ALS model
als_model.load_model('models/als_model.pkl')

# Save DQN model
dqn_reranker.save_model('models/dqn_model.pt')

# Load DQN model
dqn_reranker.load_model('models/dqn_model.pt')
```

## Performance Metrics

Track system performance:

```python
# Overall CTR
ctr = neural_bandit.get_ctr()

# Stage-specific CTRs
stage_ctrs = neural_bandit.get_stage_ctrs()
# Returns: {'als_only_ctr': 0.15, 'neural_bandit_ctr': 0.20}
```

## Dashboard Features

### Real-Time Metrics
- **Baseline CTR**: Pure Matrix Factorization performance
- **Neural-Bandit CTR**: Full system with DQN + Exploration
- **Relative Improvement**: Percentage gain over baseline
- **Total Interactions**: Number of simulated user interactions

### Visualizations
- **CTR Over Time**: Rolling window showing learning progress
- **Learning Threshold**: Marker at 1,000 interactions
- **Side-by-Side Comparison**: Architecture and feature comparison

### Interactive Controls
- **Run 1 Step**: Single interaction simulation
- **Run 10 Steps**: Quick batch of interactions
- **Run 100 Steps**: Full batch with progress bar
- **Reset All**: Clear data and restart

## Expected Results

After 1,000+ interactions, expect to see:

- **CTR Improvement**: 10-30% increase over baseline
- **Context Adaptation**: Better recommendations for different times/devices
- **Exploration Benefits**: Discovery of high-value items missed by pure MF
- **Learning Curve**: Visible improvement in the dashboard graphs

## Tips for Production

1. **Batch Training**: Run ALS training nightly on full historical data
2. **Online Learning**: Update DQN continuously from user interactions
3. **Redis Setup**: Use Redis for session state in distributed systems
4. **Model Checkpoints**: Save models periodically for backup
5. **Monitoring**: Track CTR and other metrics in production
6. **A/B Testing**: Use the dashboard framework for real A/B tests

## Troubleshooting

### Issue: "Port 8501 is not available"
**Solution**: Another Streamlit instance is running. Stop it with:
```bash
pkill -f streamlit
```

### Issue: "Redis connection failed"
**Solution**: The system falls back to in-memory storage automatically. To use Redis:
```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
redis-server
```

### Issue: "IndexError: index out of range in self"
**Solution**: Ensure item IDs are within [0, n_items) range. This was fixed in commit b45c3d3.

### Issue: Slow training on large datasets
**Solution**: Reduce ALS iterations or use GPU acceleration for PyTorch.

## Advanced Usage

### Custom Reward Functions

Override the reward calculation in your interaction loop:

```python
# Simple binary reward
reward = 1.0 if clicked else 0.0

# Time-weighted reward
reward = (1.0 if clicked else 0.0) * (1.0 / (1.0 + time_on_page))

# Position-biased reward
reward = (1.0 if clicked else 0.0) * (1.0 / (1.0 + position))
```

### Custom Context Features

Extend the context feature vector:

```python
import numpy as np

def generate_custom_context(user_id, session_data):
    time_of_day = session_data['hour'] / 23.0
    device_type = session_data['device_code'] / 2.0
    
    # Add custom features
    user_age = get_user_age(user_id) / 100.0
    session_length = min(session_data['duration'], 3600) / 3600.0
    
    return np.array([time_of_day, device_type, user_age, session_length, 0.0])
```

### Multi-Armed Bandit Variants

Experiment with different exploration strategies by modifying `neural_ucb.py`:

```python
# Epsilon-greedy
def epsilon_greedy(items, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(items)
    return items[0]  # Best item

# Thompson Sampling
def thompson_sampling(items, alpha, beta):
    samples = [np.random.beta(alpha[i], beta[i]) for i in items]
    return items[np.argmax(samples)]
```

## Further Reading

- **Matrix Factorization**: [Implicit Library Docs](https://implicit.readthedocs.io/)
- **Deep Q-Networks**: [DQN Paper](https://arxiv.org/abs/1312.5602)
- **Contextual Bandits**: [Introduction to Bandits](https://arxiv.org/abs/1904.07272)
- **NeuralUCB**: [Neural Contextual Bandits](https://arxiv.org/abs/1911.04462)
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io/)

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation in README.md
- Review the demo and test scripts for examples
