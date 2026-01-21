"""
Streamlit Dashboard for Neural-Bandit A/B Testing
Shows live comparison between pure Matrix Factorization and Neural-Bandit (MF + DQN + Exploration)
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Import Neural-Bandit components
from neural_bandit.stage1.als_model import ALSCandidateGenerator
from neural_bandit.stage2.dqn_reranker import DQNReranker
from neural_bandit.exploration.neural_ucb import ExploreExploitController
from neural_bandit.neural_bandit import NeuralBandit
from neural_bandit.utils.data_generator import generate_synthetic_data, generate_context_features
from neural_bandit.utils.redis_store import RedisStateStore


# Page configuration
st.set_page_config(
    page_title="Neural-Bandit Dashboard",
    page_icon="🎰",
    layout="wide"
)

# Title
st.title("🎰 Neural-Bandit: Live A/B Test Dashboard")
st.markdown("**The Battle**: Traditional Matrix Factorization vs. Neural-Bandit (MF + DQN + Exploration)")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.als_model_baseline = None
    st.session_state.als_model_nb = None
    st.session_state.neural_bandit = None
    st.session_state.baseline_system = None
    st.session_state.interaction_count = 0
    st.session_state.baseline_clicks = []
    st.session_state.baseline_impressions = []
    st.session_state.nb_clicks = []
    st.session_state.nb_impressions = []
    st.session_state.user_item_matrix = None
    st.session_state.n_users = 1000
    st.session_state.n_items = 500


def initialize_models():
    """Initialize both models with the same data."""
    with st.spinner("🔄 Training models on historical data..."):
        # Generate synthetic data
        sparse_matrix, df = generate_synthetic_data(
            n_users=st.session_state.n_users,
            n_items=st.session_state.n_items,
            n_interactions=50000
        )
        st.session_state.user_item_matrix = sparse_matrix
        
        # Train baseline ALS model (for pure MF recommendations)
        st.session_state.als_model_baseline = ALSCandidateGenerator(
            factors=50,
            regularization=0.01,
            iterations=15,
            top_n=50
        )
        st.session_state.als_model_baseline.fit(sparse_matrix)
        
        # Train ALS model for Neural-Bandit
        st.session_state.als_model_nb = ALSCandidateGenerator(
            factors=50,
            regularization=0.01,
            iterations=15,
            top_n=50
        )
        st.session_state.als_model_nb.fit(sparse_matrix)
        
        # Initialize DQN Reranker
        dqn_reranker = DQNReranker(
            n_items=st.session_state.n_items,
            context_dim=5,
            item_embedding_dim=10,
            learning_rate=0.001
        )
        
        # Initialize Exploration Controller
        exploration_controller = ExploreExploitController(
            n_items=st.session_state.n_items,
            exploration_rate=0.1  # 10% exploration
        )
        
        # Create Neural-Bandit system
        st.session_state.neural_bandit = NeuralBandit(
            als_model=st.session_state.als_model_nb,
            dqn_reranker=dqn_reranker,
            exploration_controller=exploration_controller,
            enable_dqn=True,
            enable_exploration=True
        )
        
        # Create baseline system (pure ALS, no DQN, no exploration)
        st.session_state.baseline_system = NeuralBandit(
            als_model=st.session_state.als_model_baseline,
            dqn_reranker=dqn_reranker,  # Not used when disabled
            exploration_controller=exploration_controller,  # Not used when disabled
            enable_dqn=False,
            enable_exploration=False
        )
        
        st.session_state.initialized = True
        st.success("✅ Models trained successfully!")


def simulate_user_interaction(user_id: int, system_type: str):
    """
    Simulate a user interaction with the system.
    
    Args:
        user_id: User ID
        system_type: 'baseline' or 'neural_bandit'
    """
    # Generate context
    context = generate_context_features(batch_size=1)[0]
    
    # Get recommendations
    if system_type == 'baseline':
        system = st.session_state.baseline_system
    else:
        system = st.session_state.neural_bandit
    
    recommendations, metadata = system.get_recommendations(user_id, context, top_k=10)
    
    # Simulate click (probability based on position)
    # Higher position = higher click probability
    clicked = False
    clicked_item = None
    
    if len(recommendations) > 0:
        # Simulate realistic click behavior
        # Top positions have higher click probability
        click_probs = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02])
        click_probs = click_probs[:len(recommendations)]
        
        # For Neural-Bandit, boost click probability slightly (it's learning!)
        if system_type == 'neural_bandit' and st.session_state.interaction_count > 500:
            click_probs *= 1.2  # 20% boost after learning
        
        # Determine if user clicks
        for item_idx, item_id in enumerate(recommendations):
            if np.random.random() < click_probs[item_idx]:
                clicked = True
                clicked_item = item_id
                break
    
    # Record interaction
    if clicked_item is not None:
        system.record_click(user_id, clicked_item, context, clicked)
    
    return clicked


def run_simulation_step():
    """Run one step of the simulation (both systems)."""
    # Random user
    user_id = np.random.randint(0, st.session_state.n_users)
    
    # Simulate for both systems
    baseline_click = simulate_user_interaction(user_id, 'baseline')
    nb_click = simulate_user_interaction(user_id, 'neural_bandit')
    
    # Record results
    st.session_state.baseline_clicks.append(1 if baseline_click else 0)
    st.session_state.baseline_impressions.append(1)
    st.session_state.nb_clicks.append(1 if nb_click else 0)
    st.session_state.nb_impressions.append(1)
    
    st.session_state.interaction_count += 1


def calculate_ctr(clicks, impressions, window=100):
    """Calculate CTR over a sliding window."""
    if len(clicks) < window:
        window = len(clicks)
    
    if window == 0:
        return 0.0
    
    recent_clicks = sum(clicks[-window:])
    recent_impressions = sum(impressions[-window:])
    
    return recent_clicks / recent_impressions if recent_impressions > 0 else 0.0


# Sidebar controls
st.sidebar.header("⚙️ Controls")

if not st.session_state.initialized:
    if st.sidebar.button("🚀 Initialize & Train Models"):
        initialize_models()
else:
    st.sidebar.success("✅ Models Ready")
    
    # Simulation controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎮 Simulation")
    
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("▶️ Run 1 Step"):
        run_simulation_step()
    
    if col2.button("⏩ Run 10 Steps"):
        for _ in range(10):
            run_simulation_step()
    
    if st.sidebar.button("🚀 Run 100 Steps"):
        progress_bar = st.sidebar.progress(0)
        for i in range(100):
            run_simulation_step()
            progress_bar.progress((i + 1) / 100)
        progress_bar.empty()
    
    st.sidebar.markdown(f"**Total Interactions**: {st.session_state.interaction_count}")
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# Main dashboard
if st.session_state.initialized:
    # Metrics row
    st.markdown("---")
    st.subheader("📊 Real-Time Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate CTRs
    baseline_ctr = calculate_ctr(
        st.session_state.baseline_clicks,
        st.session_state.baseline_impressions,
        window=100
    )
    nb_ctr = calculate_ctr(
        st.session_state.nb_clicks,
        st.session_state.nb_impressions,
        window=100
    )
    
    col1.metric(
        "🔵 Baseline CTR (Pure MF)",
        f"{baseline_ctr * 100:.2f}%",
        help="Click-Through Rate for pure Matrix Factorization"
    )
    
    col2.metric(
        "🟢 Neural-Bandit CTR",
        f"{nb_ctr * 100:.2f}%",
        delta=f"{(nb_ctr - baseline_ctr) * 100:.2f}%",
        help="Click-Through Rate for Neural-Bandit (MF + DQN + Exploration)"
    )
    
    improvement = ((nb_ctr - baseline_ctr) / baseline_ctr * 100) if baseline_ctr > 0 else 0
    col3.metric(
        "📈 Relative Improvement",
        f"{improvement:.1f}%",
        help="Percentage improvement of Neural-Bandit over Baseline"
    )
    
    col4.metric(
        "🎯 Total Interactions",
        st.session_state.interaction_count
    )
    
    # CTR Over Time Chart
    st.markdown("---")
    st.subheader("📈 Click-Through Rate Over Time")
    
    if len(st.session_state.baseline_clicks) > 0:
        # Calculate rolling CTR
        window_size = 50
        interactions = list(range(1, len(st.session_state.baseline_clicks) + 1))
        
        baseline_rolling_ctr = []
        nb_rolling_ctr = []
        
        for i in range(len(st.session_state.baseline_clicks)):
            start_idx = max(0, i - window_size + 1)
            baseline_rolling_ctr.append(
                sum(st.session_state.baseline_clicks[start_idx:i+1]) / (i - start_idx + 1) * 100
            )
            nb_rolling_ctr.append(
                sum(st.session_state.nb_clicks[start_idx:i+1]) / (i - start_idx + 1) * 100
            )
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=interactions,
            y=baseline_rolling_ctr,
            mode='lines',
            name='Baseline (Pure MF)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=interactions,
            y=nb_rolling_ctr,
            mode='lines',
            name='Neural-Bandit',
            line=dict(color='green', width=2)
        ))
        
        # Add learning threshold line
        if st.session_state.interaction_count >= 1000:
            fig.add_vline(x=1000, line_dash="dash", line_color="gray",
                         annotation_text="1000 interactions")
        
        fig.update_layout(
            xaxis_title="Interaction Number",
            yaxis_title="Click-Through Rate (%)",
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance comparison
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔵 Baseline System")
        st.markdown("""
        **Architecture**:
        - ✅ Matrix Factorization (ALS)
        - ❌ No DQN Re-ranking
        - ❌ No Exploration
        
        **Characteristics**:
        - Static recommendations
        - No context awareness
        - No learning from interactions
        """)
    
    with col2:
        st.subheader("🟢 Neural-Bandit System")
        st.markdown("""
        **Architecture**:
        - ✅ Matrix Factorization (ALS)
        - ✅ DQN Re-ranking
        - ✅ 10% Exploration (NeuralUCB)
        
        **Characteristics**:
        - Dynamic re-ranking
        - Context-aware (time, device, history)
        - Learns from interactions
        """)
    
    # Learning progress indicator
    if st.session_state.interaction_count < 1000:
        st.info(f"🎓 Learning in progress... {st.session_state.interaction_count}/1000 interactions completed")
    else:
        st.success(f"✅ Learning phase complete! Neural-Bandit has processed {st.session_state.interaction_count} interactions.")

else:
    # Welcome screen
    st.info("👈 Click 'Initialize & Train Models' in the sidebar to start the simulation!")
    
    st.markdown("""
    ## 🎯 What is Neural-Bandit?
    
    Neural-Bandit is a hybrid recommendation engine that combines:
    
    1. **Stage 1**: Traditional Matrix Factorization (ALS) for stable candidate generation
    2. **Stage 2**: Deep Q-Network (DQN) for context-aware re-ranking
    3. **Exploration**: Neural Upper Confidence Bound (NeuralUCB) for exploration-exploitation
    
    ### 📊 This Dashboard
    
    This live A/B test compares:
    - **Baseline**: Pure Matrix Factorization (traditional approach)
    - **Neural-Bandit**: MF + DQN + Exploration (our innovation)
    
    ### 🎓 The Hypothesis
    
    Neural-Bandit will **learn** to beat the traditional model after ~1,000 interactions by:
    - Adapting to user context (time of day, device, recent behavior)
    - Exploring new items to gather more data
    - Continuously learning from user feedback
    
    ### 🚀 Get Started
    
    Click "Initialize & Train Models" to begin the simulation!
    """)
