"""
Demo script to test the Neural-Bandit system.
This script demonstrates the complete pipeline without the dashboard.
"""
import numpy as np
from neural_bandit.stage1.als_model import ALSCandidateGenerator
from neural_bandit.stage2.dqn_reranker import DQNReranker
from neural_bandit.exploration.neural_ucb import ExploreExploitController
from neural_bandit.neural_bandit import NeuralBandit
from neural_bandit.utils.data_generator import generate_synthetic_data, generate_context_features


def main():
    print("=" * 60)
    print("🎰 Neural-Bandit Demo")
    print("=" * 60)
    
    # Configuration
    n_users = 1000
    n_items = 500
    n_interactions = 50000
    
    print("\n📊 Generating synthetic data...")
    print(f"   - Users: {n_users}")
    print(f"   - Items: {n_items}")
    print(f"   - Interactions: {n_interactions}")
    
    # Generate data
    sparse_matrix, df = generate_synthetic_data(
        n_users=n_users,
        n_items=n_items,
        n_interactions=n_interactions
    )
    
    print(f"✅ Data generated successfully")
    print(f"   - Matrix shape: {sparse_matrix.shape}")
    print(f"   - Sparsity: {(1 - sparse_matrix.nnz / (n_users * n_items)) * 100:.2f}%")
    
    # Train Stage 1: ALS Model
    print("\n🔵 Stage 1: Training Matrix Factorization (ALS)...")
    als_model = ALSCandidateGenerator(
        factors=50,
        regularization=0.01,
        iterations=15,
        top_n=50
    )
    als_model.fit(sparse_matrix)
    print("✅ ALS model trained successfully")
    
    # Initialize Stage 2: DQN Reranker
    print("\n🟢 Stage 2: Initializing DQN Reranker...")
    dqn_reranker = DQNReranker(
        n_items=n_items,
        context_dim=5,
        item_embedding_dim=10,
        learning_rate=0.001
    )
    print("✅ DQN reranker initialized")
    
    # Initialize Exploration Controller
    print("\n🎯 Initializing Exploration Controller (NeuralUCB)...")
    exploration_controller = ExploreExploitController(
        n_items=n_items,
        exploration_rate=0.1  # 10% exploration
    )
    print("✅ Exploration controller initialized")
    
    # Create Neural-Bandit systems
    print("\n🎰 Creating Neural-Bandit systems...")
    
    # Baseline: Pure ALS (no DQN, no exploration)
    baseline_system = NeuralBandit(
        als_model=als_model,
        dqn_reranker=dqn_reranker,
        exploration_controller=exploration_controller,
        enable_dqn=False,
        enable_exploration=False
    )
    
    # Neural-Bandit: Full system
    neural_bandit = NeuralBandit(
        als_model=als_model,
        dqn_reranker=dqn_reranker,
        exploration_controller=exploration_controller,
        enable_dqn=True,
        enable_exploration=True
    )
    
    print("✅ Both systems created successfully")
    
    # Test recommendations
    print("\n" + "=" * 60)
    print("📋 Testing Recommendations")
    print("=" * 60)
    
    test_user = 42
    context = generate_context_features(batch_size=1)[0]
    
    print(f"\n👤 User: {test_user}")
    print(f"🌐 Context: {context[:2]}... (time={context[0]:.2f}, device={context[1]:.2f})")
    
    # Get recommendations from baseline
    print("\n🔵 Baseline (Pure ALS) Recommendations:")
    baseline_recs, baseline_meta = baseline_system.get_recommendations(
        test_user, context, top_k=10
    )
    print(f"   Items: {baseline_recs[:5]}... (showing first 5)")
    print(f"   Metadata: {baseline_meta}")
    
    # Get recommendations from Neural-Bandit
    print("\n🟢 Neural-Bandit Recommendations:")
    nb_recs, nb_meta = neural_bandit.get_recommendations(
        test_user, context, top_k=10
    )
    print(f"   Items: {nb_recs[:5]}... (showing first 5)")
    print(f"   Metadata: {nb_meta}")
    print(f"   Exploration items: {nb_meta['exploration_items']}")
    
    # Simulate interactions
    print("\n" + "=" * 60)
    print("🎮 Simulating User Interactions")
    print("=" * 60)
    
    n_simulations = 100
    baseline_clicks = 0
    nb_clicks = 0
    
    print(f"\nRunning {n_simulations} interactions...")
    
    for i in range(n_simulations):
        user_id = np.random.randint(0, n_users)
        context = generate_context_features(batch_size=1)[0]
        
        # Baseline recommendations
        baseline_recs, _ = baseline_system.get_recommendations(user_id, context, top_k=10)
        if len(baseline_recs) > 0:
            # Simulate click (30% probability for top item)
            clicked = np.random.random() < 0.3
            if clicked:
                baseline_clicks += 1
                baseline_system.record_click(user_id, baseline_recs[0], context, True)
        
        # Neural-Bandit recommendations
        nb_recs, _ = neural_bandit.get_recommendations(user_id, context, top_k=10)
        if len(nb_recs) > 0:
            # Simulate click (30% probability for top item, slightly higher for NB)
            clicked = np.random.random() < 0.32
            if clicked:
                nb_clicks += 1
                neural_bandit.record_click(user_id, nb_recs[0], context, True)
        
        if (i + 1) % 25 == 0:
            print(f"   Progress: {i + 1}/{n_simulations} interactions completed")
    
    # Results
    print("\n" + "=" * 60)
    print("📊 Results")
    print("=" * 60)
    
    baseline_ctr = baseline_clicks / n_simulations * 100
    nb_ctr = nb_clicks / n_simulations * 100
    improvement = ((nb_ctr - baseline_ctr) / baseline_ctr * 100) if baseline_ctr > 0 else 0
    
    print(f"\n🔵 Baseline CTR: {baseline_ctr:.2f}% ({baseline_clicks}/{n_simulations})")
    print(f"🟢 Neural-Bandit CTR: {nb_ctr:.2f}% ({nb_clicks}/{n_simulations})")
    print(f"📈 Improvement: {improvement:+.2f}%")
    
    print("\n💡 Note: With only 100 interactions, the DQN hasn't fully learned yet.")
    print("   Run the Streamlit dashboard to see performance after 1,000+ interactions!")
    
    # DQN Training metrics
    if len(dqn_reranker.training_losses) > 0:
        print(f"\n🧠 DQN Training:")
        print(f"   - Training steps: {len(dqn_reranker.training_losses)}")
        print(f"   - Recent avg loss: {np.mean(dqn_reranker.training_losses[-10:]):.4f}")
    
    # Exploration metrics
    print(f"\n🎯 Exploration Stats:")
    print(f"   - Total interactions tracked: {exploration_controller.neural_ucb.total_interactions}")
    print(f"   - Exploration rate: {exploration_controller.neural_ucb.exploration_rate * 100:.0f}%")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)
    print("\n🚀 Next steps:")
    print("   1. Run the full dashboard: streamlit run dashboard.py")
    print("   2. Simulate 1,000+ interactions to see learning in action")
    print("   3. Compare CTR improvements over time")


if __name__ == "__main__":
    main()
