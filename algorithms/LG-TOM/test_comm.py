"""
Simple test script to verify the communication architecture works correctly.
This script tests the forward pass without full training.
"""
import sys
sys.path.append('/home/huao/Research/SocialJax')
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import numpy as np

# Import from lgtom_cnn_coins
from lgtom_cnn_coins import (
    ActorCriticComm,
    ProtoLayer,
    aggregate_communication,
    CNN
)


def test_proto_layer():
    """Test the ProtoLayer module"""
    print("\n" + "="*50)
    print("Testing ProtoLayer")
    print("="*50)
    
    # Initialize
    num_protos = 5
    comm_dim = 64
    batch_size = 4
    hidden_dim = 128
    
    proto_layer = ProtoLayer(num_protos=num_protos, comm_dim=comm_dim)
    
    # Create dummy input
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch_size, hidden_dim))
    
    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    variables = proto_layer.init(init_rng, x, train_mode=False)
    
    # Test forward pass (evaluation mode)
    comm_vector, comm_logits = proto_layer.apply(variables, x, train_mode=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Comm vector shape: {comm_vector.shape}")
    print(f"Comm logits shape: {comm_logits.shape}")
    print(f"Expected comm vector shape: ({batch_size}, {comm_dim})")
    print(f"Expected comm logits shape: ({batch_size}, {num_protos})")
    
    assert comm_vector.shape == (batch_size, comm_dim), "Comm vector shape mismatch!"
    assert comm_logits.shape == (batch_size, num_protos), "Comm logits shape mismatch!"
    print("✓ ProtoLayer test passed!")
    
    # Test training mode with RNG
    rng, train_rng = jax.random.split(rng)
    comm_vector_train, comm_logits_train = proto_layer.apply(
        variables, x, train_mode=True, rng=train_rng
    )
    print(f"✓ Training mode test passed!")
    
    return True


def test_actor_critic_comm():
    """Test the full ActorCriticComm module"""
    print("\n" + "="*50)
    print("Testing ActorCriticComm")
    print("="*50)
    
    # Parameters
    action_dim = 4
    comm_dim = 64
    num_protos = 8
    hidden_dim = 128  # Must be embedding_dim (64) + comm_dim (64)
    batch_size = 3
    obs_shape = (11, 11, 25)  # Example observation shape
    
    # Initialize network
    network = ActorCriticComm(
        action_dim=action_dim,
        comm_dim=comm_dim,
        num_protos=num_protos,
        hidden_dim=hidden_dim,
        activation="relu"
    )
    
    # Create dummy inputs
    rng = jax.random.PRNGKey(42)
    obs = jax.random.normal(rng, (batch_size, *obs_shape))
    prev_comm = jnp.zeros((batch_size, comm_dim))
    hidden_state = jnp.zeros((batch_size, hidden_dim))
    
    # Initialize parameters
    rng, init_rng, gumbel_rng = jax.random.split(rng, 3)
    variables = network.init(
        {'params': init_rng, 'gumbel': gumbel_rng},
        obs,
        prev_comm,
        hidden_state,
        train_mode=True
    )
    
    print(f"Network initialized with {len(jax.tree_util.tree_leaves(variables))} parameter arrays")
    
    # Test forward pass
    rng, apply_rng = jax.random.split(rng)
    action_logits, comm_vector, comm_logits, value, new_hidden_state = network.apply(
        variables,
        obs,
        prev_comm,
        hidden_state,
        train_mode=False,
        rngs={'gumbel': apply_rng}
    )
    
    print(f"\nOutput shapes:")
    print(f"  Action logits: {action_logits.shape} (expected: ({batch_size}, {action_dim}))")
    print(f"  Comm vector: {comm_vector.shape} (expected: ({batch_size}, {comm_dim}))")
    print(f"  Comm logits: {comm_logits.shape} (expected: ({batch_size}, {num_protos}))")
    print(f"  Value: {value.shape} (expected: ({batch_size},))")
    print(f"  New hidden state: {new_hidden_state.shape} (expected: ({batch_size}, {hidden_dim}))")
    
    # Verify shapes
    assert action_logits.shape == (batch_size, action_dim)
    assert comm_vector.shape == (batch_size, comm_dim)
    assert comm_logits.shape == (batch_size, num_protos)
    assert value.shape == (batch_size,)
    assert new_hidden_state.shape == (batch_size, hidden_dim)
    
    print("✓ ActorCriticComm test passed!")
    return True


def test_aggregate_communication():
    """Test the communication aggregation function"""
    print("\n" + "="*50)
    print("Testing aggregate_communication")
    print("="*50)
    
    num_envs = 2
    num_agents = 3
    comm_dim = 64
    
    # Create dummy communication vectors
    comm_vectors = jax.random.normal(
        jax.random.PRNGKey(0),
        (num_envs, num_agents, comm_dim)
    )
    
    # Test average aggregation
    aggregated_avg = aggregate_communication(comm_vectors, num_agents, comm_mode='avg')
    print(f"Input shape: {comm_vectors.shape}")
    print(f"Aggregated (avg) shape: {aggregated_avg.shape}")
    print(f"Expected shape: ({num_envs}, {num_agents}, {comm_dim})")
    
    assert aggregated_avg.shape == (num_envs, num_agents, comm_dim)
    
    # Verify no self-communication
    # Each agent should receive average of (num_agents - 1) other agents
    for env_idx in range(num_envs):
        for agent_idx in range(num_agents):
            # Get the communication this agent should receive
            received = aggregated_avg[env_idx, agent_idx]
            
            # Calculate expected (average of all except self)
            other_comms = []
            for j in range(num_agents):
                if j != agent_idx:
                    other_comms.append(comm_vectors[env_idx, j])
            expected = jnp.mean(jnp.stack(other_comms), axis=0)
            
            # Check if they match
            assert jnp.allclose(received, expected, atol=1e-5), \
                f"Aggregation mismatch for env {env_idx}, agent {agent_idx}"
    
    print("✓ Aggregate communication test passed!")
    
    # Test sum aggregation
    aggregated_sum = aggregate_communication(comm_vectors, num_agents, comm_mode='sum')
    print(f"Aggregated (sum) shape: {aggregated_sum.shape}")
    assert aggregated_sum.shape == (num_envs, num_agents, comm_dim)
    print("✓ Sum aggregation test passed!")
    
    return True


def test_multi_step_simulation():
    """Test multi-step forward passes (simulating training loop)"""
    print("\n" + "="*50)
    print("Testing Multi-Step Simulation")
    print("="*50)
    
    # Parameters
    num_steps = 5
    num_envs = 2
    num_agents = 2
    action_dim = 4
    comm_dim = 64
    num_protos = 8
    hidden_dim = 128  # Must be embedding_dim (64) + comm_dim (64)
    obs_shape = (11, 11, 25)
    
    # Initialize network
    network = ActorCriticComm(
        action_dim=action_dim,
        comm_dim=comm_dim,
        num_protos=num_protos,
        hidden_dim=hidden_dim,
        activation="relu"
    )
    
    # Initialize
    rng = jax.random.PRNGKey(123)
    init_obs = jax.random.normal(rng, (1, *obs_shape))
    init_comm = jnp.zeros((1, comm_dim))
    init_hidden = jnp.zeros((1, hidden_dim))
    
    rng, init_rng, gumbel_rng = jax.random.split(rng, 3)
    variables = network.init(
        {'params': init_rng, 'gumbel': gumbel_rng},
        init_obs,
        init_comm,
        init_hidden,
        train_mode=True
    )
    
    # Initialize states
    hidden_states = jnp.zeros((num_envs, num_agents, hidden_dim))
    prev_comm = jnp.zeros((num_envs, num_agents, comm_dim))
    
    print(f"Running {num_steps} simulation steps...")
    
    for step in range(num_steps):
        # Generate observations
        rng, obs_rng = jax.random.split(rng)
        obs = jax.random.normal(obs_rng, (num_envs * num_agents, *obs_shape))
        
        # Reshape states
        hidden_batch = hidden_states.reshape(-1, hidden_dim)
        prev_comm_batch = prev_comm.reshape(-1, comm_dim)
        
        # Forward pass
        rng, apply_rng = jax.random.split(rng)
        action_logits, comm_vectors, comm_logits, values, new_hidden_batch = network.apply(
            variables,
            obs,
            prev_comm_batch,
            hidden_batch,
            train_mode=False,
            rngs={'gumbel': apply_rng}
        )
        
        # Reshape outputs
        comm_vectors_reshaped = comm_vectors.reshape(num_envs, num_agents, -1)
        new_hidden_reshaped = new_hidden_batch.reshape(num_envs, num_agents, -1)
        
        # Aggregate communication
        aggregated_comm = aggregate_communication(
            comm_vectors_reshaped,
            num_agents,
            comm_mode='avg'
        )
        
        # Update states for next step
        hidden_states = new_hidden_reshaped
        prev_comm = aggregated_comm
        
        print(f"  Step {step + 1}: ✓")
    
    print("✓ Multi-step simulation test passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  LG-TOM Communication Architecture Test Suite")
    print("="*70)
    
    tests = [
        ("ProtoLayer", test_proto_layer),
        ("ActorCriticComm", test_actor_critic_comm),
        ("Communication Aggregation", test_aggregate_communication),
        ("Multi-Step Simulation", test_multi_step_simulation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAILED"))
    
    # Print summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    for test_name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"  {symbol} {test_name}: {status}")
    
    all_passed = all(status == "PASSED" for _, status in results)
    print("="*70)
    if all_passed:
        print("  All tests passed! 🎉")
    else:
        print("  Some tests failed. Please review the errors above.")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

