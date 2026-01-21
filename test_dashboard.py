"""
Validation script for dashboard functionality.
Tests core features without requiring interactive web server.
"""
import numpy as np
from neural_bandit.stage1.als_model import ALSCandidateGenerator
from neural_bandit.stage2.dqn_reranker import DQNReranker
from neural_bandit.exploration.neural_ucb import ExploreExploitController
from neural_bandit.neural_bandit import NeuralBandit
from neural_bandit.utils.data_generator import generate_synthetic_data, generate_context_features


def test_dashboard_initialization():
    """Test the initialization flow used in the dashboard."""
    print("🧪 Testing Dashboard Initialization Flow...")
    
    # Parameters from dashboard
    n_users = 100  # Reduced for faster testing
    n_items = 50
    n_interactions = 5000
    
    # Generate data
    print("  📊 Generating data...")
    sparse_matrix, df = generate_synthetic_data(
        n_users=n_users,
        n_items=n_items,
        n_interactions=n_interactions
    )
    
    # Train ALS (baseline)
    print("  🔵 Training baseline ALS...")
    als_model_baseline = ALSCandidateGenerator(
        factors=50,
        regularization=0.01,
        iterations=10,
        top_n=50
    )
    als_model_baseline.fit(sparse_matrix)
    
    # Train ALS (for Neural-Bandit)
    print("  🟢 Training Neural-Bandit ALS...")
    als_model_nb = ALSCandidateGenerator(
        factors=50,
        regularization=0.01,
        iterations=10,
        top_n=50
    )
    als_model_nb.fit(sparse_matrix)
    
    # Initialize DQN
    print("  🧠 Initializing DQN...")
    dqn_reranker = DQNReranker(
        n_items=n_items,
        context_dim=5,
        item_embedding_dim=10,
        learning_rate=0.001
    )
    
    # Initialize Exploration Controller
    print("  🎯 Initializing exploration...")
    exploration_controller = ExploreExploitController(
        n_items=n_items,
        exploration_rate=0.1
    )
    
    # Create systems
    print("  🎰 Creating systems...")
    baseline_system = NeuralBandit(
        als_model=als_model_baseline,
        dqn_reranker=dqn_reranker,
        exploration_controller=exploration_controller,
        enable_dqn=False,
        enable_exploration=False
    )
    
    neural_bandit = NeuralBandit(
        als_model=als_model_nb,
        dqn_reranker=dqn_reranker,
        exploration_controller=exploration_controller,
        enable_dqn=True,
        enable_exploration=True
    )
    
    print("  ✅ Initialization successful!")
    return baseline_system, neural_bandit, n_users, n_items


def test_dashboard_simulation(baseline_system, neural_bandit, n_users):
    """Test the simulation flow used in the dashboard."""
    print("\n🧪 Testing Dashboard Simulation Flow...")
    
    baseline_clicks = []
    nb_clicks = []
    
    # Run 50 interactions
    for i in range(50):
        user_id = np.random.randint(0, n_users)
        context = generate_context_features(batch_size=1)[0]
        
        # Baseline
        baseline_recs, _ = baseline_system.get_recommendations(user_id, context, top_k=10)
        if len(baseline_recs) > 0:
            clicked = np.random.random() < 0.3
            baseline_clicks.append(1 if clicked else 0)
            if clicked:
                baseline_system.record_click(user_id, baseline_recs[0], context, True)
        
        # Neural-Bandit
        nb_recs, _ = neural_bandit.get_recommendations(user_id, context, top_k=10)
        if len(nb_recs) > 0:
            clicked = np.random.random() < 0.3
            nb_clicks.append(1 if clicked else 0)
            if clicked:
                neural_bandit.record_click(user_id, nb_recs[0], context, True)
    
    # Calculate metrics
    baseline_ctr = sum(baseline_clicks) / len(baseline_clicks) * 100 if baseline_clicks else 0
    nb_ctr = sum(nb_clicks) / len(nb_clicks) * 100 if nb_clicks else 0
    
    print(f"  📊 Baseline CTR: {baseline_ctr:.2f}%")
    print(f"  📊 Neural-Bandit CTR: {nb_ctr:.2f}%")
    print("  ✅ Simulation successful!")
    
    return baseline_clicks, nb_clicks


def test_dashboard_metrics(baseline_clicks, nb_clicks):
    """Test metric calculations used in the dashboard."""
    print("\n🧪 Testing Dashboard Metric Calculations...")
    
    # Rolling CTR calculation (window=10)
    window = 10
    baseline_rolling = []
    nb_rolling = []
    
    for i in range(len(baseline_clicks)):
        start_idx = max(0, i - window + 1)
        baseline_rolling.append(
            sum(baseline_clicks[start_idx:i+1]) / (i - start_idx + 1) * 100
        )
        nb_rolling.append(
            sum(nb_clicks[start_idx:i+1]) / (i - start_idx + 1) * 100
        )
    
    print(f"  📈 Rolling CTRs calculated: {len(baseline_rolling)} points")
    print(f"  📊 Latest baseline CTR: {baseline_rolling[-1]:.2f}%")
    print(f"  📊 Latest NB CTR: {nb_rolling[-1]:.2f}%")
    print("  ✅ Metrics calculation successful!")


def main():
    print("=" * 60)
    print("🎰 Neural-Bandit Dashboard Validation")
    print("=" * 60)
    
    try:
        # Test initialization
        baseline_system, neural_bandit, n_users, n_items = test_dashboard_initialization()
        
        # Test simulation
        baseline_clicks, nb_clicks = test_dashboard_simulation(
            baseline_system, neural_bandit, n_users
        )
        
        # Test metrics
        test_dashboard_metrics(baseline_clicks, nb_clicks)
        
        print("\n" + "=" * 60)
        print("✅ All Dashboard Tests Passed!")
        print("=" * 60)
        print("\n💡 The dashboard is ready to use!")
        print("   Run: streamlit run dashboard.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
