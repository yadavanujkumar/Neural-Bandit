# 🎰 Neural-Bandit

A hybrid recommendation engine that combines Traditional Matrix Factorization with Dynamic Deep Q-Network (Reinforcement Learning) to create an intelligent, adaptive recommendation system.

## 🎯 Overview

Neural-Bandit innovates on traditional recommendation systems by introducing a two-stage architecture with intelligent exploration:

1. **Stage 1 - The 'Stable' Candidate Generator**: Uses Matrix Factorization (ALS) to generate top-50 items based on long-term user preferences
2. **Stage 2 - The 'Dynamic' Re-Ranker**: Employs a Deep Q-Network (DQN) to re-rank items based on real-time context (time of day, device type, recent clicks)
3. **The 'Regret' Minimizer**: Implements Neural Upper Confidence Bound (NeuralUCB) algorithm for intelligent exploration-exploitation

## 🚀 Key Features

- **Hybrid Architecture**: Combines the stability of Matrix Factorization with the adaptability of Deep Reinforcement Learning
- **Context-Aware**: Real-time re-ranking based on user session context
- **Intelligent Exploration**: 10% exploration rate with NeuralUCB for discovering new high-value items
- **Live A/B Testing**: Interactive Streamlit dashboard comparing Neural-Bandit vs Pure Matrix Factorization
- **Redis State Management**: Optional Redis integration for production-scale state storage
- **Learning Visualization**: Track how the system learns and improves over 1,000+ interactions

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yadavanujkumar/Neural-Bandit.git
cd Neural-Bandit

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Quick Start

### Run the Interactive Dashboard

```bash
streamlit run dashboard.py
```

This launches an interactive A/B testing dashboard where you can:
- Compare Neural-Bandit vs Traditional Matrix Factorization in real-time
- Visualize learning progress and CTR improvements
- Run simulations to see how the system adapts over time

### Basic Usage Example

```python
from neural_bandit.stage1.als_model import ALSCandidateGenerator
from neural_bandit.stage2.dqn_reranker import DQNReranker
from neural_bandit.exploration.neural_ucb import ExploreExploitController
from neural_bandit.neural_bandit import NeuralBandit
from neural_bandit.utils.data_generator import generate_synthetic_data, generate_context_features

# Generate synthetic data
sparse_matrix, df = generate_synthetic_data(n_users=1000, n_items=500)

# Stage 1: Train ALS model
als_model = ALSCandidateGenerator(factors=50, iterations=15, top_n=50)
als_model.fit(sparse_matrix)

# Stage 2: Initialize DQN reranker
dqn_reranker = DQNReranker(n_items=500, context_dim=5, item_embedding_dim=10)

# Initialize exploration controller
exploration_controller = ExploreExploitController(n_items=500, exploration_rate=0.1)

# Create Neural-Bandit system
neural_bandit = NeuralBandit(
    als_model=als_model,
    dqn_reranker=dqn_reranker,
    exploration_controller=exploration_controller
)

# Get recommendations
user_id = 42
context = generate_context_features(batch_size=1)[0]
recommendations, metadata = neural_bandit.get_recommendations(user_id, context, top_k=10)

# Record user interaction (clicked on item)
neural_bandit.record_click(user_id, recommendations[0], context, clicked=True)
```

## 🏗️ Architecture

### Stage 1: Matrix Factorization (ALS)

- **Library**: Implicit (ALS algorithm)
- **Purpose**: Generate stable, high-quality candidate items based on historical data
- **Output**: Top-50 items per user (the "safe list")
- **Training**: Runs nightly on batch data

### Stage 2: Deep Q-Network (DQN)

- **Framework**: PyTorch
- **Purpose**: Re-rank candidates based on real-time context
- **Input**: 
  - Context features: Time of day, Device type, Last 3 clicks
  - Item embeddings (learned)
- **Output**: Q-values for ranking
- **Learning**: Continuous learning from user interactions

### Exploration Strategy: NeuralUCB

- **Algorithm**: Neural Upper Confidence Bound
- **Exploration Rate**: 10% random items
- **Purpose**: Balance between showing best items (exploitation) and discovering new items (exploration)
- **Innovation**: Mathematically calculates potential reward of risky vs safe items

## 📊 Dashboard Features

The Streamlit dashboard provides:

1. **Real-Time Metrics**:
   - Click-Through Rate (CTR) for both systems
   - Relative improvement percentage
   - Total interaction count

2. **Learning Visualization**:
   - CTR over time (rolling window)
   - Comparison between baseline and Neural-Bandit
   - Learning threshold indicator (1,000 interactions)

3. **Interactive Controls**:
   - Run 1, 10, or 100 simulation steps
   - Reset and restart experiments
   - Track learning progress

4. **System Comparison**:
   - Side-by-side architecture comparison
   - Feature breakdown
   - Performance characteristics

## 🔧 Components

### `/neural_bandit/stage1/als_model.py`
Matrix Factorization candidate generator using ALS algorithm.

### `/neural_bandit/stage2/dqn_reranker.py`
Deep Q-Network for context-aware re-ranking with experience replay.

### `/neural_bandit/exploration/neural_ucb.py`
Neural Upper Confidence Bound for exploration-exploitation balance.

### `/neural_bandit/neural_bandit.py`
Main system integrating all components with unified interface.

### `/neural_bandit/utils/data_generator.py`
Synthetic data generation for testing and simulation.

### `/neural_bandit/utils/redis_store.py`
Redis state store for production deployment (optional, falls back to in-memory).

### `/dashboard.py`
Interactive Streamlit dashboard for A/B testing and visualization.

## 🎯 The Innovation

Neural-Bandit proves that **reinforcement learning can beat traditional collaborative filtering** after sufficient interactions. The dashboard demonstrates:

- **Cold Start**: Initially, both systems perform similarly
- **Learning Phase**: After ~500 interactions, Neural-Bandit begins to improve
- **Mature Performance**: After 1,000+ interactions, Neural-Bandit consistently outperforms pure Matrix Factorization by adapting to context

## 🧪 Tech Stack

- **Python 3.8+**
- **Implicit**: Matrix Factorization (ALS)
- **PyTorch**: Deep Q-Network implementation
- **Redis**: State management (optional)
- **Streamlit**: Interactive dashboard
- **Plotly**: Visualization
- **NumPy/Pandas**: Data processing
- **SciPy**: Sparse matrix operations

## 📈 Performance

Expected improvements after 1,000 interactions:
- **CTR Improvement**: 10-30% over baseline
- **Context Adaptation**: Better recommendations for different times/devices
- **Exploration Benefits**: Discovery of high-value items missed by pure MF

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Advanced exploration strategies
- Multi-armed bandit variants
- Real-world dataset integration
- Production deployment guides
- Performance optimizations

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Implicit library for efficient ALS implementation
- PyTorch team for the deep learning framework
- Streamlit for the amazing dashboard framework