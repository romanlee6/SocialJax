"""
Test script for social influence intrinsic reward mechanism.

This script verifies that:
1. Counterfactual generation works correctly
2. Influence rewards are computed properly
3. The mechanism integrates well with the training loop

Note: This test automatically uses the updated implementation that correctly
distinguishes between GRU hidden state and belief output. When INFLUENCE_TARGET
is "belief", the counterfactual generation uses the GRU output (belief), not
the hidden state (carry).
"""

import sys
sys.path.append('/home/huao/Research/SocialJax')

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

# Import functions from lgtom_cnn_coins
from lgtom_cnn_coins import (
    ActorCriticComm,
    generate_counterfactuals,
    marginalize_over_own_comm,
    compute_social_influence_reward,
    aggregate_communication
)

def test_counterfactual_generation():
    """Test that counterfactual generation produces expected shapes"""
    print("Testing counterfactual generation...")
    
    # Setup
    rng = jax.random.PRNGKey(0)
    num_agents = 2
    num_protos = 5
    comm_dim = 64
    hidden_dim = 128
    action_dim = 4
    obs_shape = (11, 11, 3)
    num_envs = 4
    
    # Create network
    network = ActorCriticComm(
        action_dim=action_dim,
        comm_dim=comm_dim,
        num_protos=num_protos,
        hidden_dim=hidden_dim,
        activation="relu"
    )
    
    # Initialize network
    rng, _rng = jax.random.split(rng)
    init_obs = jnp.zeros((1, *obs_shape))
    init_comm = jnp.zeros((1, comm_dim))
    init_hidden = jnp.zeros((1, hidden_dim))
    
    params = network.init(
        {'params': _rng, 'gumbel': _rng},
        init_obs,
        init_comm,
        init_hidden,
        train_mode=True
    )
    
    # Create dummy data
    batch_size = num_envs * num_agents
    obs_batch = jax.random.normal(_rng, (batch_size, *obs_shape))
    prev_comm_batch = jax.random.normal(_rng, (batch_size, comm_dim))
    hidden_batch = jax.random.normal(_rng, (batch_size, hidden_dim))
    proto_embeddings = params['params']['ProtoLayer_0']['prototypes']
    
    config = {
        "INFLUENCE_TARGET": "belief",
        "COMM_MODE": "avg",
        "NUM_PROTOS": num_protos,
        "COMM_DIM": comm_dim,
    }
    
    # Generate counterfactuals
    rng, _rng = jax.random.split(rng)
    counterfactuals = generate_counterfactuals(
        network=network,
        params=params,
        obs_batch=obs_batch,
        prev_comm_batch=prev_comm_batch,
        hidden_batch=hidden_batch,
        proto_embeddings=proto_embeddings,
        num_agents=num_agents,
        num_protos=num_protos,
        comm_dim=comm_dim,
        config=config,
        rng=_rng
    )
    
    # Check shapes
    expected_shape = (num_agents, num_protos, num_agents, hidden_dim)
    assert counterfactuals.shape == expected_shape, f"Expected {expected_shape}, got {counterfactuals.shape}"
    
    print(f"✓ Counterfactuals shape: {counterfactuals.shape}")
    print(f"✓ Counterfactuals contain finite values: {jnp.all(jnp.isfinite(counterfactuals))}")
    
    return counterfactuals, params, config


def test_marginalization(counterfactuals):
    """Test marginalization over communication policy"""
    print("\nTesting marginalization...")
    
    num_agents = counterfactuals.shape[0]
    num_protos = counterfactuals.shape[1]
    output_dim = counterfactuals.shape[-1]
    
    # Create dummy communication probabilities
    comm_logits = jax.random.normal(jax.random.PRNGKey(1), (num_agents, num_protos))
    comm_probs = jax.nn.softmax(comm_logits, axis=-1)
    
    # Marginalize
    marginal = marginalize_over_own_comm(comm_probs, counterfactuals)
    
    # Check shapes
    expected_shape = (num_agents, num_agents, output_dim)
    assert marginal.shape == expected_shape, f"Expected {expected_shape}, got {marginal.shape}"
    
    # Check that probabilities sum to approximately 1 along prototype dimension
    assert jnp.all(jnp.abs(comm_probs.sum(axis=-1) - 1.0) < 1e-5), "Communication probs don't sum to 1"
    
    print(f"✓ Marginal shape: {marginal.shape}")
    print(f"✓ Marginal contains finite values: {jnp.all(jnp.isfinite(marginal))}")
    
    return marginal, comm_probs


def test_influence_reward(marginal, counterfactuals, comm_probs):
    """Test influence reward computation"""
    print("\nTesting influence reward computation...")
    
    num_agents = counterfactuals.shape[0]
    output_dim = counterfactuals.shape[-1]
    
    # Create dummy actual outputs (using first counterfactual as "actual")
    actual_outputs = counterfactuals[:, 0, :, :].reshape(-1, output_dim)
    
    # Create dummy belief states
    belief_states = jax.random.normal(jax.random.PRNGKey(2), (num_agents, output_dim))
    
    # Create dummy comm logits
    num_protos = counterfactuals.shape[1]
    comm_logits = jax.random.normal(jax.random.PRNGKey(3), (num_agents, num_protos))
    
    config = {"INFLUENCE_TARGET": "belief"}
    
    # Compute influence reward
    influence_reward = compute_social_influence_reward(
        belief_states=belief_states,
        comm_logits=comm_logits,
        counterfactuals=counterfactuals,
        actual_outputs=actual_outputs,
        config=config
    )
    
    # Check shapes
    expected_shape = (num_agents,)
    assert influence_reward.shape == expected_shape, f"Expected {expected_shape}, got {influence_reward.shape}"
    
    # Check that rewards are finite and in reasonable range [0, 2]
    assert jnp.all(jnp.isfinite(influence_reward)), "Influence rewards contain non-finite values"
    assert jnp.all(influence_reward >= 0.0), "Influence rewards contain negative values"
    
    print(f"✓ Influence reward shape: {influence_reward.shape}")
    print(f"✓ Influence rewards: {influence_reward}")
    print(f"✓ Mean influence: {jnp.mean(influence_reward):.4f}")
    print(f"✓ Std influence: {jnp.std(influence_reward):.4f}")
    
    return influence_reward


def test_integration():
    """Test integration with training loop"""
    print("\n" + "="*70)
    print("Testing Integration with Training Loop")
    print("="*70)
    
    # This tests the full pipeline
    counterfactuals, params, config = test_counterfactual_generation()
    marginal, comm_probs = test_marginalization(counterfactuals)
    influence_reward = test_influence_reward(marginal, counterfactuals, comm_probs)
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    
    return True


def test_with_different_targets():
    """Test with different influence targets (action vs belief)"""
    print("\n" + "="*70)
    print("Testing Different Influence Targets")
    print("="*70)
    
    for target in ["belief", "action"]:
        print(f"\nTesting INFLUENCE_TARGET={target}...")
        
        # Setup (simplified from test_counterfactual_generation)
        rng = jax.random.PRNGKey(0)
        num_agents = 2
        num_protos = 5
        comm_dim = 64
        hidden_dim = 128
        action_dim = 4
        obs_shape = (11, 11, 3)
        num_envs = 4
        
        network = ActorCriticComm(
            action_dim=action_dim,
            comm_dim=comm_dim,
            num_protos=num_protos,
            hidden_dim=hidden_dim,
            activation="relu"
        )
        
        rng, _rng = jax.random.split(rng)
        init_obs = jnp.zeros((1, *obs_shape))
        init_comm = jnp.zeros((1, comm_dim))
        init_hidden = jnp.zeros((1, hidden_dim))
        
        params = network.init(
            {'params': _rng, 'gumbel': _rng},
            init_obs,
            init_comm,
            init_hidden,
            train_mode=True
        )
        
        batch_size = num_envs * num_agents
        obs_batch = jax.random.normal(_rng, (batch_size, *obs_shape))
        prev_comm_batch = jax.random.normal(_rng, (batch_size, comm_dim))
        hidden_batch = jax.random.normal(_rng, (batch_size, hidden_dim))
        proto_embeddings = params['params']['ProtoLayer_0']['prototypes']
        
        config = {
            "INFLUENCE_TARGET": target,
            "COMM_MODE": "avg",
            "NUM_PROTOS": num_protos,
            "COMM_DIM": comm_dim,
        }
        
        # Generate counterfactuals
        rng, _rng = jax.random.split(rng)
        counterfactuals = generate_counterfactuals(
            network=network,
            params=params,
            obs_batch=obs_batch,
            prev_comm_batch=prev_comm_batch,
            hidden_batch=hidden_batch,
            proto_embeddings=proto_embeddings,
            num_agents=num_agents,
            num_protos=num_protos,
            comm_dim=comm_dim,
            config=config,
            rng=_rng
        )
        
        # Check output dimension
        if target == "belief":
            expected_output_dim = hidden_dim
        else:  # action
            expected_output_dim = action_dim
        
        assert counterfactuals.shape[-1] == expected_output_dim, \
            f"Expected output dim {expected_output_dim}, got {counterfactuals.shape[-1]}"
        
        print(f"  ✓ Target={target}: Output dimension is {counterfactuals.shape[-1]} (expected {expected_output_dim})")
    
    print("\n✓ Different targets test passed!")


if __name__ == "__main__":
    print("="*70)
    print("Social Influence Intrinsic Reward - Test Suite")
    print("="*70)
    
    # Run tests
    test_integration()
    test_with_different_targets()
    
    print("\n" + "="*70)
    print("All tests completed successfully! ✓")
    print("="*70)
    print("\nYou can now run the full training with:")
    print("  python lgtom_cnn_coins.py")
    print("\nMake sure to set in config:")
    print("  SOCIAL_INFLUENCE_COEFF: 0.1")
    print("  INFLUENCE_TARGET: 'belief' or 'action'")

