""" 
Based on PureJaxRL & jaxmarl Implementation of PPO with LG-TOM Communication

This implementation includes:
1. Social Influence Intrinsic Reward via Counterfactual Reasoning
2. Theory of Mind (ToM) Prediction Model
3. Supervised Learning from Ground Truth or LLM Dataset

THEORY OF MIND (ToM) MODEL:
---------------------------
The ToM model enables agents to predict other agents' belief states based on their 
communication and observations. This implements a key component of human-like social reasoning.

Components:
- ToM Predictor Network: Takes (communication + embedded observation) as input and outputs
  predicted belief states of other agents
- Supervised Learning: ToM can be trained via:
  * Ground truth supervision: Using actual belief states from other agents
  * LLM dataset supervision: Using pre-collected LLM reasoning trajectories (UNDER CONSTRUCTION)
  * Loss function: Configurable cosine similarity or mean squared error
- Integration with Intrinsic Rewards: When both ToM and intrinsic rewards are enabled,
  the ToM predictions are used for counterfactual reasoning in influence calculation

Configuration:
- USE_TOM: Enable/disable ToM prediction model
- SUPERVISED_BELIEF: "none", "ground_truth", or "llm" 
- SUPERVISED_COMM: "none", "ground_truth", or "llm"
- SUPERVISED_LOSS_COEF: Weight for supervised learning loss
- SUPERVISED_LOSS_TYPE: "cosine" (default) or "mse" for supervision
- LLM_DATA_PATH: Path to offline LLM dataset (*** UNDER CONSTRUCTION ***)

SOCIAL INFLUENCE MECHANISM:
---------------------------
The social influence reward measures how much an agent's communication affects other agents'
behaviors or beliefs through counterfactual reasoning.

Key Components:
1. Counterfactual Generation (generate_counterfactuals):
   - For each agent k and each possible message v, we compute:
     "What would other agents j do/believe if agent k sent message v?"
   - This is done by:
     a) Replacing agent k's communication with each prototype message
     b) Running other agents' forward pass with the counterfactual communication
     c) Recording their resulting actions or belief states
   - PARAMETER SHARING MODE: Uses shared policy to predict all agents' responses
   - NON-PARAMETER SHARING MODE: Uses each agent's actual policy for predictions

2. Marginalization (marginalize_over_own_comm):
   - Compute expected influence by marginalizing over agent k's communication policy:
     E_{m ~ π_comm(m|s_k)}[prediction(s_j | m_k=m)]
   - This gives us the expected behavior/belief of agent j given k's comm distribution

3. Influence Reward Computation (compute_social_influence_reward):
   - Measure influence as the difference between:
     * Marginalized counterfactual predictions (what others would do on average)
     * Actual predictions (what others actually do)
   - Higher difference = higher influence = agent's communication matters more
   - BELIEF TARGET: Uses cosine similarity (influence = 1 - similarity)
   - ACTION TARGET: Uses KL divergence (higher KL = more influence)

4. Separate Reward Training:
   - Action policy is trained using external task rewards
   - Communication policy is trained using intrinsic social influence rewards
   - Value function is trained on combined total reward
   - This allows specialization: actions optimize task performance, comm optimizes influence

Configuration Options:
- USE_INTRINSIC_REWARD: Enable/disable intrinsic reward calculation (default: False)
- SOCIAL_INFLUENCE_COEFF: Weight for intrinsic reward (0.0 = disabled, kept for backward compatibility)
- INFLUENCE_TARGET: What to measure influence on
  * "belief": Measure impact on other agents' belief states (GRU output, not hidden state)
  * "action": Measure impact on other agents' action distributions (uses KL divergence)
- USE_SEPARATE_REWARDS: Whether to use separate rewards for action and comm policies (default: True)
- COMM_LOSS_COEF: Weight for communication policy loss (default: 0.1)

Note: When both USE_TOM and USE_INTRINSIC_REWARD are enabled, the system uses ToM predictions
for counterfactual belief states, making the intrinsic reward calculation more efficient and
theoretically grounded.

Implementation Notes:
- Parameter Sharing: Uses agent's own policy to predict others (homogeneous assumption)
- Non-Parameter Sharing: Directly accesses other agents' policies for counterfactual reasoning
- JAX-compatible: All operations use JAX for automatic differentiation and JIT compilation

Based on Theory of Mind (ToM) and counterfactual reasoning from:
- TOM-MAC architecture (mac_tom.py)
- Social Influence in multi-agent communication
"""
import sys
sys.path.append('/home/huao/Research/SocialJax')
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
# from flax.training import checkpoints
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper, SVOLogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def _compute_supervised_loss(predictions, targets, loss_type="cosine"):
    """Compute supervised loss between predictions and targets."""
    predictions = jnp.asarray(predictions)
    targets = jnp.asarray(targets)
    if loss_type == "mse":
        return jnp.mean(jnp.square(predictions - targets), axis=-1)
    elif loss_type == "cosine":
        dot_product = jnp.sum(predictions * targets, axis=-1)
        pred_norm = jnp.linalg.norm(predictions, axis=-1) + 1e-8
        target_norm = jnp.linalg.norm(targets, axis=-1) + 1e-8
        cos_sim = dot_product / (pred_norm * target_norm)
        return 1.0 - cos_sim
    else:
        raise ValueError(f"Unsupported supervised loss type: {loss_type}")


def _count_non_zero_queries(results, threshold=1e-8):
    """Count how many query vectors contain non-zero information."""
    results = jnp.asarray(results)
    non_zero_mask = jnp.any(jnp.abs(results) > threshold, axis=-1)
    return jnp.sum(non_zero_mask).astype(jnp.float32)


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ProtoLayer(nn.Module):
    """Prototype layer for discrete communication using Gumbel-Softmax"""
    num_protos: int
    comm_dim: int
    
    @nn.compact
    def __call__(self, x, train_mode=True, temperature=1.0, rng=None):
        # Generate logits for prototype selection
        logits = nn.Dense(
            self.num_protos,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        
        # Prototype embeddings
        prototypes = self.param(
            'prototypes',
            nn.initializers.uniform(scale=0.5),
            (self.num_protos, self.comm_dim)
        )
        
        # Apply Gumbel-Softmax for differentiable sampling
        if train_mode and rng is not None:
            # Gumbel-Softmax with hard samples during training
            gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape) + 1e-8) + 1e-8)
            gumbel_logits = (logits + gumbel_noise) / temperature
            soft_samples = jax.nn.softmax(gumbel_logits, axis=-1)
            hard_samples = jax.nn.one_hot(jnp.argmax(gumbel_logits, axis=-1), self.num_protos)
            samples = jax.lax.stop_gradient(hard_samples - soft_samples) + soft_samples
        else:
            # Greedy selection during evaluation or if no RNG provided
            samples = jax.nn.one_hot(jnp.argmax(logits, axis=-1), self.num_protos)
        
        # Get communication vector from prototypes
        comm_vector = jnp.dot(samples, prototypes)
        comm_index = jnp.argmax(samples, axis=-1)
        
        return comm_vector, logits, comm_index


class ActorCriticComm(nn.Module):
    """Actor-Critic with Communication based on TomMAC architecture
    
    Includes Theory of Mind (ToM) prediction capability:
    - ToM model takes comm + embedded obs as input
    - Outputs belief prediction of other agents
    - Can be supervised on ground truth beliefs or offline LLM dataset
    """
    action_dim: int
    comm_dim: int = 64
    num_protos: int = 10
    hidden_dim: int = 128  # Must match embedding_dim (64) + comm_dim (64)
    activation: str = "relu"
    use_tom: bool = False  # Enable Theory of Mind prediction
    use_intrinsic_reward: bool = False  # Enable intrinsic reward calculation

    @nn.compact
    def __call__(self, obs, prev_comm, hidden_state, train_mode=True):
        """
        Args:
            obs: observation (batch, height, width, channels)
            prev_comm: previous communication from other agents (batch, comm_dim)
            hidden_state: GRU hidden state (batch, hidden_dim)
            train_mode: whether in training mode
        Returns:
            action_logits, comm_vector, comm_logits, value, new_hidden_state, belief, tom_pred (optional)
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # 1. CNN Embedder
        embedding = CNN(self.activation)(obs)
        
        # 2. Concatenate embedding with received communication
        belief_input = jnp.concatenate([embedding, prev_comm], axis=-1)
        
        # 3. GRU Belief Model
        GRUCell = nn.RNNCellBase
        gru_cell = nn.GRUCell(features=self.hidden_dim)
        new_hidden_state, belief = gru_cell(hidden_state, belief_input)
        
        # 4. Theory of Mind (ToM) Prediction (optional)
        # Predicts other agents' belief states based on their communication and observations
        tom_pred = None
        if self.use_tom:
            # ToM takes the same input as belief GRU (comm + embedding)
            # and predicts what other agents believe
            tom_hidden = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name='tom_predictor'
            )(belief_input)
            # The belief GRU outputs are bounded in [-1, 1] (tanh gating). Using
            # ReLU here prevents the ToM head from matching the sign structure
            # of those targets, which stalls the cosine-similarity loss.

            # tom_hidden = nn.LayerNorm()(tom_hidden)
            tom_activation = nn.tanh
            tom_pred = tom_activation(tom_hidden)
        
        # 5. Communication Policy (using prototype layer)
        proto_layer = ProtoLayer(num_protos=self.num_protos, comm_dim=self.comm_dim)
        comm_hidden = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(belief)
        comm_hidden = activation(comm_hidden)
        
        # Get RNG for Gumbel-Softmax if in training mode
        gumbel_rng = self.make_rng('gumbel') if train_mode else None
        comm_vector, comm_logits, comm_index = proto_layer(comm_hidden, train_mode=train_mode, rng=gumbel_rng)
        
        # 6. Action Policy
        action_hidden = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(belief)
        action_hidden = activation(action_hidden)
        action_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(action_hidden)
        
        # 7. Critic (value function)
        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(belief)
        critic = activation(critic)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)
        
        # Return belief, new_hidden_state, and optional tom_pred
        # belief: the GRU output used for action/comm generation
        # new_hidden_state: the carry state for next timestep
        # tom_pred: ToM prediction of other agents' beliefs (if use_tom=True)
        return (
            action_logits,
            comm_vector,
            comm_logits,
            comm_index,
            jnp.squeeze(value, axis=-1),
            new_hidden_state,
            belief,
            tom_pred,
        )


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class TransitionComm(NamedTuple):
    """Transition with communication"""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    action_reward: jnp.ndarray  # External task reward for action policy
    comm_reward: jnp.ndarray    # Intrinsic social influence reward for comm policy
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    comm_vector: jnp.ndarray
    comm_log_prob: jnp.ndarray
    comm_index: jnp.ndarray
    hidden_state: jnp.ndarray
    belief_state: jnp.ndarray  # Belief state (GRU output) for supervised learning
    tom_prediction: jnp.ndarray  # ToM predictions (if enabled) for supervised learning
    prev_comm: jnp.ndarray  # Received/aggregated communication that was used as input
    info: jnp.ndarray
    agent_positions: jnp.ndarray  # (num_envs * num_agents, 2) - [x, y] positions for semantic key
    closest_ore_types: jnp.ndarray  # (num_envs * num_agents,) - closest ore type indices (0=none, 1=iron, 2=gold)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(params, config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["PARAMETER_SHARING"]:
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    else:
        network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    for o in range(config["GIF_NUM_FRAMES"]):
        print(o)
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
        if config["PARAMETER_SHARING"]: 
            pi, value = network.apply(params, obs_batch)
            action = pi.sample(seed=key_a0)
            env_act = unbatchify(
                action, env.agents, 1, env.num_agents
            )           
        else:
            env_act = {}
            for i in range(env.num_agents):
                pi, value = network[i].apply(params[i], obs_batch)
                action = pi.sample(seed=key_a0)
                env_act[env.agents[i]] = action


        

        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def aggregate_communication(comm_vectors, num_agents, comm_mode='avg'):
    """
    Aggregate communication vectors from all agents.
    
    Args:
        comm_vectors: (num_envs, num_agents, comm_dim)
        num_agents: int
        comm_mode: 'avg' for average, 'sum' for sum
    
    Returns:
        aggregated_comm: (num_envs, num_agents, comm_dim) - received communication for each agent
    """
    # Create mask to exclude self-communication
    mask = 1.0 - jnp.eye(num_agents)  # (num_agents, num_agents)
    
    # comm_vectors shape: (num_envs, num_agents, comm_dim)
    # We want: for each agent, aggregate messages from all other agents
    
    # Expand for broadcasting: (num_envs, num_agents, num_agents, comm_dim)
    comm_expanded = jnp.expand_dims(comm_vectors, axis=1)  # (num_envs, 1, num_agents, comm_dim)
    comm_expanded = jnp.tile(comm_expanded, (1, num_agents, 1, 1))  # (num_envs, num_agents, num_agents, comm_dim)
    
    # Apply mask: (num_envs, num_agents, num_agents, 1)
    mask_expanded = jnp.expand_dims(mask, axis=(0, -1))  # (1, num_agents, num_agents, 1)
    masked_comm = comm_expanded * mask_expanded
    
    # Aggregate across sender dimension
    if comm_mode == 'avg':
        aggregated_comm = jnp.sum(masked_comm, axis=2) / (num_agents - 1)  # (num_envs, num_agents, comm_dim)
    else:  # sum
        aggregated_comm = jnp.sum(masked_comm, axis=2)  # (num_envs, num_agents, comm_dim)
    
    return aggregated_comm


def generate_counterfactuals(network, params, obs_batch, prev_comm_batch, hidden_batch, 
                            proto_embeddings, num_agents, num_protos, comm_dim, config, rng,
                            parameter_sharing=True, tom_predictions=None):
    """
    Generate counterfactual predictions for what other agents would do/believe
    under each possible communication from the current agent.
    
    This implements counterfactual reasoning: "If I send message m, how would others respond?"
    
    For parameter sharing: Uses the shared policy to predict all agents' responses.
    For non-parameter sharing: Uses each agent's actual policy to predict their response.
    
    **ToM Integration:**
    If both USE_TOM and USE_INTRINSIC_REWARD are enabled, uses ToM predictions
    for counterfactual belief states instead of running full forward passes.
    This is more efficient and aligns with the theory that agents reason about
    others' beliefs through their ToM model.
    
    Dimension flow:
    1. Input: (num_envs * num_agents, ...) 
    2. Reshaped: (num_envs, num_agents, ...)
    3. Tiled: (num_agents * num_protos * num_envs, num_agents, ...)
    4. Flattened: (num_agents * num_protos * num_envs * num_agents, ...) for forward pass
    5. Reshaped back: (num_agents, num_protos, num_envs, num_agents, ...)
    6. Averaged: (num_agents, num_protos, num_agents, output_dim)
    
    Args:
        network: The ActorCriticComm network (or list of networks if not parameter sharing)
        params: Network parameters (or list of params if not parameter sharing)
        obs_batch: Observations (num_envs * num_agents, ...)
        prev_comm_batch: Previous communications (num_envs * num_agents, comm_dim)
        hidden_batch: Hidden states (num_envs * num_agents, hidden_dim)
        proto_embeddings: Prototype embeddings (num_protos, comm_dim) or list of embeddings
        num_agents: Number of agents
        num_protos: Number of prototype messages
        comm_dim: Communication dimension
        config: Configuration dict
        rng: Random key
        parameter_sharing: Whether using parameter sharing
        tom_predictions: Optional ToM predictions (num_envs * num_agents, hidden_dim)
        
    Returns:
        counterfactuals: (num_agents, num_protos, num_agents, output_dim)
            where output_dim is:
            - action_dim (for action influence): action probability distribution
            - hidden_dim (for belief influence): GRU output (belief), not hidden state
    """
    num_envs = obs_batch.shape[0] // num_agents
    
    # Check if we should use ToM predictions for counterfactuals
    use_tom_counterfactuals = (
        config.get("USE_TOM", False) and 
        tom_predictions is not None
    )
    
    # Note on the difference between ToM and non-ToM counterfactuals:
    # BOTH cases run full forward passes with permuted communications.
    # The difference is WHICH output we extract:
    # - WITH ToM (use_tom_counterfactuals=True): Extract ToM predictions as counterfactual beliefs
    #   This represents the agent's mental model of how others would think under different comms
    # - WITHOUT ToM (use_tom_counterfactuals=False): Extract actual belief states as counterfactual beliefs  
    #   This represents the ground truth of how others actually would think
    # In both cases, actual_outputs in compute_social_influence_reward() contains current beliefs.
    
    # Reshape to (num_envs, num_agents, ...)
    obs_reshaped = obs_batch.reshape(num_envs, num_agents, *obs_batch.shape[1:])
    prev_comm_reshaped = prev_comm_batch.reshape(num_envs, num_agents, comm_dim)
    hidden_reshaped = hidden_batch.reshape(num_envs, num_agents, -1)
    
    # For each agent k and each prototype v, compute counterfactual predictions
    # We need predictions for all agents in each scenario
    # Total batch size: num_agents (which agent sends) * num_protos (which message) * num_envs * num_agents (predictions for each agent)
    batch_size = num_agents * num_protos * num_envs * num_agents
    
    # Repeat observations for all counterfactual scenarios
    # After tiling: (num_agents * num_protos * num_envs, num_agents, ...)
    obs_repeated = jnp.tile(obs_reshaped, (num_agents * num_protos, 1, 1, 1, 1))
    # Flatten to (num_agents * num_protos * num_envs * num_agents, ...)
    obs_repeated = obs_repeated.reshape(-1, *obs_batch.shape[1:])
    
    # Same for hidden states
    hidden_repeated = jnp.tile(hidden_reshaped, (num_agents * num_protos, 1, 1))
    hidden_repeated = hidden_repeated.reshape(-1, hidden_reshaped.shape[-1])
    
    # Create counterfactual communications
    # For each (agent_k, proto_v), replace agent_k's comm with proto_v
    comm_counterfactual = jnp.tile(prev_comm_reshaped, (num_agents * num_protos, 1, 1))
    
    # Generate indices for replacement
    agent_indices = jnp.arange(num_agents).repeat(num_protos * num_envs)
    proto_indices = jnp.tile(jnp.arange(num_protos).repeat(num_envs), num_agents)
    env_indices = jnp.tile(jnp.arange(num_envs), num_agents * num_protos)
    
    # Replace with prototype embeddings
    if parameter_sharing:
        # Single set of prototypes
        comm_counterfactual = comm_counterfactual.at[
            jnp.arange(num_agents * num_protos * num_envs), 
            agent_indices
        ].set(proto_embeddings[proto_indices])
    else:
        # Each agent has their own prototypes
        # Use vectorized indexing instead of Python loop
        # Stack all prototypes: (num_agents, num_protos, comm_dim) -> index by [agent_idx, proto_idx]
        proto_stack = jnp.stack(proto_embeddings, axis=0)  # (num_agents, num_protos, comm_dim)
        # Get the appropriate prototypes for each scenario
        selected_protos = proto_stack[agent_indices, proto_indices]  # (num_agents*num_protos*num_envs, comm_dim)
        # Set them in the counterfactual communications
        comm_counterfactual = comm_counterfactual.at[
            jnp.arange(num_agents * num_protos * num_envs),
            agent_indices
        ].set(selected_protos)
    
    # Aggregate counterfactual communications
    comm_counterfactual_reshaped = comm_counterfactual.reshape(
        num_agents * num_protos * num_envs, num_agents, comm_dim
    )
    aggregated_comm = jax.vmap(
        lambda c: aggregate_communication(
            jnp.expand_dims(c, 0), num_agents, config.get("COMM_MODE", "avg")
        ).squeeze(0)
    )(comm_counterfactual_reshaped)
    
    # aggregated_comm shape: (num_agents * num_protos * num_envs, num_agents, comm_dim)
    # Flatten to (num_agents * num_protos * num_envs * num_agents, comm_dim)
    aggregated_comm_flat = aggregated_comm.reshape(-1, comm_dim)
    
    # Forward pass through network to get counterfactual predictions
    # The key difference when using ToM: we extract ToM predictions instead of actual beliefs
    rng_split = jax.random.split(rng, batch_size)
    
    if parameter_sharing:
        # Use shared policy for all agents
        action_logits_cf, _, _, _, _, hidden_cf, belief_cf, tom_pred_cf = jax.vmap(
            lambda obs, comm, hid, r: network.apply(
                params,
                jnp.expand_dims(obs, 0),
                jnp.expand_dims(comm, 0),
                jnp.expand_dims(hid, 0),
                train_mode=False,
                rngs={'gumbel': r}
            )
        )(obs_repeated, aggregated_comm_flat, hidden_repeated, rng_split)
        
        # Reshape to remove extra dimensions (from expand_dims in network call)
        action_logits_cf = action_logits_cf.reshape(batch_size, -1)
        hidden_cf = hidden_cf.reshape(batch_size, -1)
        belief_cf = belief_cf.reshape(batch_size, -1)
        if tom_pred_cf is not None:
            tom_pred_cf = tom_pred_cf.reshape(batch_size, -1)
    else:
        # Use each agent's own policy for predictions
        # Process each receiving agent's predictions separately to avoid dynamic indexing
        # batch structure: (sending_agent * num_protos * num_envs * receiving_agent)
        
        all_action_logits = []
        all_hidden = []
        all_belief = []
        all_tom_pred = []
        
        for agent_idx in range(num_agents):
            # Get indices for this receiving agent
            # For each (sending_agent, proto, env) combination, get the prediction for this receiving agent
            agent_indices = jnp.arange(agent_idx, batch_size, num_agents)
            
            # Get data for this agent
            obs_agent = obs_repeated[agent_indices]
            comm_agent = aggregated_comm_flat[agent_indices]
            hidden_agent = hidden_repeated[agent_indices]
            rng_agent = rng_split[agent_indices]
            
            # Apply this agent's policy
            action_logits_i, _,_, _, _, hidden_i, belief_i, tom_pred_i = jax.vmap(
                lambda obs, comm, hid, r: network[agent_idx].apply(
                    params[agent_idx],
                    jnp.expand_dims(obs, 0),
                    jnp.expand_dims(comm, 0),
                    jnp.expand_dims(hid, 0),
                    train_mode=False,
                    rngs={'gumbel': r}
                )
            )(obs_agent, comm_agent, hidden_agent, rng_agent)
            
            # Reshape to remove extra dimensions (from expand_dims in network call)
            # The vmap outputs have shape (batch_per_agent, 1, 1, dim) -> reshape to (batch_per_agent, dim)
            batch_per_agent = action_logits_i.shape[0]
            action_logits_i = action_logits_i.reshape(batch_per_agent, -1)
            hidden_i = hidden_i.reshape(batch_per_agent, -1)
            belief_i = belief_i.reshape(batch_per_agent, -1)
            if tom_pred_i is not None:
                tom_pred_i = tom_pred_i.reshape(batch_per_agent, -1)
            
            all_action_logits.append(action_logits_i)
            all_hidden.append(hidden_i)
            all_belief.append(belief_i)
            all_tom_pred.append(tom_pred_i)
        
        # Interleave results to match the batch structure
        # Stack and reshape to get back to (batch_size, ...) order
        action_logits_cf = jnp.stack(all_action_logits, axis=1)  # (batch_size//num_agents, num_agents, ...)
        action_logits_cf = action_logits_cf.reshape(batch_size, -1)
        
        hidden_cf = jnp.stack(all_hidden, axis=1)
        hidden_cf = hidden_cf.reshape(batch_size, -1)
        
        belief_cf = jnp.stack(all_belief, axis=1)
        belief_cf = belief_cf.reshape(batch_size, -1)
        
        # Only stack ToM predictions if they exist (not None when no_tom is disabled)
        if all_tom_pred[0] is not None:
            tom_pred_cf = jnp.stack(all_tom_pred, axis=1)
            tom_pred_cf = tom_pred_cf.reshape(batch_size, -1)
        else:
            tom_pred_cf = None
    
    # Reshape to (num_agents, num_protos, num_envs, num_agents, ...)
    action_logits_cf = action_logits_cf.reshape(num_agents, num_protos, num_envs, num_agents, -1)
    belief_cf = belief_cf.reshape(num_agents, num_protos, num_envs, num_agents, -1)
    if tom_pred_cf is not None:
        tom_pred_cf = tom_pred_cf.reshape(num_agents, num_protos, num_envs, num_agents, -1)
    
    # Average over environments
    if config.get("INFLUENCE_TARGET", "belief") == "action":
        # Return action probabilities
        action_probs = jax.nn.softmax(action_logits_cf, axis=-1)
        counterfactual_result = action_probs.mean(axis=2)  # (num_agents, num_protos, num_agents, action_dim)
    else:
        # Return belief states: use ToM predictions if enabled, otherwise use actual beliefs
        if use_tom_counterfactuals and tom_pred_cf is not None:
            # Use ToM predictions as counterfactual beliefs
            # This represents each agent's theory of how others' beliefs change with different communications
            counterfactual_result = tom_pred_cf.mean(axis=2)  # (num_agents, num_protos, num_agents, hidden_dim)
        else:
            # Use actual belief states (ground truth) as counterfactual beliefs
            counterfactual_result = belief_cf.mean(axis=2)  # (num_agents, num_protos, num_agents, hidden_dim)
    
    # Stop gradients from flowing through counterfactual reasoning pathways
    return jax.lax.stop_gradient(counterfactual_result)


def marginalize_over_own_comm(comm_probs, counterfactuals, epsilon=1e-8):
    """
    Marginalize counterfactual predictions over agent's own communication distribution.
    
    This computes: E_{m ~ π_comm(m|s_k)}[prediction(s_j | m_k=m)]
    
    Args:
        comm_probs: (num_envs * num_agents, num_protos) - communication probabilities
        counterfactuals: (num_agents, num_protos, num_agents, output_dim)
        
    Returns:
        marginal: (num_agents, num_agents, output_dim) - marginalized predictions
    """
    num_agents = counterfactuals.shape[0]
    num_protos = counterfactuals.shape[1]
    output_dim = counterfactuals.shape[-1]
    
    # Reshape comm_probs to (num_agents, num_protos)
    # Assuming single environment or averaged
    comm_probs_reshaped = comm_probs.reshape(-1, num_agents, num_protos).mean(axis=0)
    
    # Weighted sum: (k, j, d)
    # marginal[k, j] = sum_v comm_probs[k, v] * counterfactuals[k, v, j]
    marginal = jnp.einsum('kv,kvjd->kjd', comm_probs_reshaped, counterfactuals)
    
    # Normalize
    norm = comm_probs_reshaped.sum(axis=1, keepdims=True)[..., None] + epsilon
    marginal = marginal / norm
    
    return marginal


def compute_kl_divergence(p, q, epsilon=1e-8):
    """
    Compute KL divergence KL(p || q) for probability distributions.
    
    Args:
        p: probability distribution (should sum to 1)
        q: probability distribution (should sum to 1)
        epsilon: small constant for numerical stability
        
    Returns:
        kl_div: scalar KL divergence value
    """
    # Clamp probabilities to avoid log(0)
    p = jnp.clip(p, epsilon, 1.0)
    q = jnp.clip(q, epsilon, 1.0)
    return jnp.sum(p * jnp.log(p / q))




def find_closest_ore_in_fov_jax(agent_x, agent_y, agent_direction, grid):
    """
    Find the closest ore in the agent's field of view (JAX-compatible).
    
    Args:
        agent_x: jnp.ndarray or int, agent x position (row)
        agent_y: jnp.ndarray or int, agent y position (col)
        agent_direction: jnp.ndarray or int, agent direction (0=North, 1=East, 2=South, 3=West)
        grid: jnp.ndarray, grid state (H, W)
        
    Returns:
        ore_type_idx: int, 0=none, 1=iron, 2=gold (treats gold_partial as gold)
    """
    # Convert to JAX arrays if needed
    agent_x = jnp.asarray(agent_x, dtype=jnp.int32)
    agent_y = jnp.asarray(agent_y, dtype=jnp.int32)
    agent_direction = jnp.asarray(agent_direction, dtype=jnp.int32)
    # FOV parameters
    forward_range = 9
    backward_range = 1
    left_range = 5
    right_range = 5
    
    # Get grid dimensions
    grid_h, grid_w = grid.shape
    
    # Create coordinate grids
    x_coords = jnp.arange(grid_h)
    y_coords = jnp.arange(grid_w)
    X, Y = jnp.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Compute relative positions
    rel_row = X - agent_x
    rel_col = Y - agent_y
    
    # Transform based on direction using JAX-compatible operations
    # Direction 0=North, 1=East, 2=South, 3=West
    forward_north = -rel_row
    backward_north = rel_row
    left_north = -rel_col
    right_north = rel_col
    
    forward_east = rel_col
    backward_east = -rel_col
    left_east = rel_row
    right_east = -rel_row
    
    forward_south = rel_row
    backward_south = -rel_row
    left_south = rel_col
    right_south = -rel_col
    
    forward_west = -rel_col
    backward_west = rel_col
    left_west = -rel_row
    right_west = rel_row
    
    # Select based on direction
    forward = jnp.where(agent_direction == 0, forward_north,
               jnp.where(agent_direction == 1, forward_east,
               jnp.where(agent_direction == 2, forward_south, forward_west)))
    backward = jnp.where(agent_direction == 0, backward_north,
                jnp.where(agent_direction == 1, backward_east,
                jnp.where(agent_direction == 2, backward_south, backward_west)))
    left = jnp.where(agent_direction == 0, left_north,
            jnp.where(agent_direction == 1, left_east,
            jnp.where(agent_direction == 2, left_south, left_west)))
    right = jnp.where(agent_direction == 0, right_north,
             jnp.where(agent_direction == 1, right_east,
             jnp.where(agent_direction == 2, right_south, right_west)))
    
    # Check if in FOV
    in_forward = (forward >= 0) & (forward <= forward_range)
    in_backward = (backward >= 0) & (backward <= backward_range)
    in_left = (left >= 0) & (left <= left_range)
    in_right = (right >= 0) & (right <= right_range)
    in_fov = (in_forward | in_backward) & (in_left | in_right)
    
    # Compute distances
    distances = jnp.sqrt(rel_row**2 + rel_col**2)
    distances = jnp.where(in_fov, distances, jnp.inf)
    
    # Items.iron_ore = 4, Items.gold_ore = 5, Items.gold_partial = 6
    # Find iron ores (grid value 4)
    iron_ores = (grid == 4) & in_fov
    iron_distances = jnp.where(iron_ores, distances, jnp.inf)
    closest_iron_dist = jnp.min(iron_distances)
    has_iron = jnp.isfinite(closest_iron_dist)
    
    # Find gold ores (grid value 5) and gold_partial (grid value 6) - treat both as gold
    gold_ores = ((grid == 5) | (grid == 6)) & in_fov
    gold_distances = jnp.where(gold_ores, distances, jnp.inf)
    closest_gold_dist = jnp.min(gold_distances)
    has_gold = jnp.isfinite(closest_gold_dist)
    
    # Return closest ore type index: 0=none, 1=iron, 2=gold
    has_ore = has_iron | has_gold
    iron_closer = (closest_iron_dist < closest_gold_dist) & has_iron
    
    ore_type_idx = jnp.where(
        ~has_ore, 0,  # No ore
        jnp.where(iron_closer, 1, 2)  # Iron or gold
    )
    
    return ore_type_idx




def normalize_l2(x):
    """
    Normalize vector(s) using L2 normalization.
    
    Args:
        x: numpy array, 1D or 2D
        
    Returns:
        Normalized array with same shape as input
    """
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def load_offline_llm_dataset(data_path, env_name, config):
    """
    Load offline LLM dataset for supervised communication/belief training.
    
    The dataset uses semantic_key vectors as keys for O(1) lookup:
    - For coop_mining: Key: (agent_id, x, y, ore_type_encoded, num_agents_in_fov, action_idx)
    - Value: {
        'belief_embedding': np.ndarray,
        'communication_embedding': np.ndarray,
        'belief_text': str,
        'communication_text': str
    }
    
    This function creates a JAX-compatible dense embedding table for GPU-accelerated lookups.
    
    Args:
        data_path: str, path to offline dataset (.pkl file)
        env_name: str, name of environment
        config: Configuration dict
        
    Returns:
        dataset: dict with JAX embedding table and lookup function, or None if not available
    """
    import pickle
    import numpy as np
    from pathlib import Path
    
    if not data_path or data_path == "":
        return None
    
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Warning: LLM dataset not found at {data_path}")
        return None
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        dataset = data.get('dataset', {})
        embedding_dim = data.get('embedding_dim', 384)
        
        # For coop_mining: semantic key structure is different
        # Key: (agent_id, agent_x, agent_y, closest_ore_type, num_agents_in_fov, action)
        ore_type_to_idx = data.get('ore_type_to_idx', {'none': 0, 'iron': 1, 'gold': 2})
        action_to_idx = data.get('action_to_idx', {
            'turn_left': 0, 'turn_right': 1, 'left': 2, 'right': 3,
            'up': 4, 'down': 5, 'stay': 6, 'mine': 7
        })
        
        # Get parameters for coop_mining environment
        if env_name == "CoopMining" or "coop_mining" in env_name.lower():
            # Grid size for coop_mining is 27x27
            grid_size_row, grid_size_col = 27, 27
            # Default number of agents is 4, but can be configured
            num_agents = config.get("NUM_AGENTS", 4)
            # Maximum number of agents in FOV (reasonable upper bound)
            max_num_agents_in_fov = config.get("MAX_NUM_AGENTS_IN_FOV", 10)
        else:
            # Fallback for other environments (coins)
            grid_size_row = config.get("GRID_SIZE_ROW", 16)
            grid_size_col = config.get("GRID_SIZE_COL", 11)
            num_agents = 2
            max_num_agents_in_fov = 2
            # Keep old mappings for backward compatibility
            color_to_idx = data.get('color_to_idx', {'red': 0, 'green': 1})
            coin_color_to_idx = data.get('coin_color_to_idx', {'none': 0, 'red': 1, 'green': 2})
        
        # Get target dimensions from config (shorten embeddings from 256 to target dims)
        belief_target_dim = config.get("HIDDEN_DIM", 128)  # Default 128 for belief
        comm_target_dim = config.get("COMM_DIM", 64)  # Default 64 for communication
        
        print(f"Loaded LLM dataset from {data_path}")
        print(f"  Dataset size: {len(dataset)} entries")
        print(f"  Embedding dimension (stored): {embedding_dim}")
        print(f"  Belief target dimension: {belief_target_dim}")
        print(f"  Communication target dimension: {comm_target_dim}")
        print(f"  Grid size: ({grid_size_row}, {grid_size_col})")
        if env_name == "CoopMining" or "coop_mining" in env_name.lower():
            print(f"  Number of agents: {num_agents}")
            print(f"  Max agents in FOV: {max_num_agents_in_fov}")
        
        # Create dense embedding tables for JAX lookups with target dimensions
        # For coop_mining: Shape: (num_agents, grid_size_row, grid_size_col, num_ore_types, max_num_agents_in_fov, num_actions, target_dim)
        # num_ore_types = 3 (none=0, iron=1, gold=2)
        # num_actions = 8 (0-7: turn_left, turn_right, left, right, up, down, stay, mine)
        if env_name == "CoopMining" or "coop_mining" in env_name.lower():
            belief_embedding_table = np.zeros(
                (num_agents, grid_size_row, grid_size_col, 3, max_num_agents_in_fov, 8, belief_target_dim),
                dtype=np.float32
            )
            comm_embedding_table = np.zeros(
                (num_agents, grid_size_row, grid_size_col, 3, max_num_agents_in_fov, 8, comm_target_dim),
                dtype=np.float32
            )
        else:
            # Backward compatibility for coins
            belief_embedding_table = np.zeros(
                (2, grid_size_row, grid_size_col, 3, 7, belief_target_dim),
                dtype=np.float32
            )
            comm_embedding_table = np.zeros(
                (2, grid_size_row, grid_size_col, 3, 7, comm_target_dim),
                dtype=np.float32
            )
        
        # Fill embedding tables from dataset, truncating to target dimensions
        filled_count = 0
        for key_vector, entry in dataset.items():
            if env_name == "CoopMining" or "coop_mining" in env_name.lower():
                # For coop_mining: key_vector is (agent_id, x, y, ore_type, num_agents_in_fov, action)
                if len(key_vector) != 6:
                    continue
                agent_id, x, y, ore_type, num_agents_in_fov, action = key_vector
                
                # Convert ore_type to encoded index if it's a string
                if isinstance(ore_type, str):
                    ore_type_encoded = ore_type_to_idx.get(ore_type.lower(), 0)
                else:
                    ore_type_encoded = int(ore_type)
                
                # Convert action to index if it's a string
                if isinstance(action, str):
                    action_idx = action_to_idx.get(action.lower(), 6)
                else:
                    action_idx = int(action)
                
                # Bounds checking for coop_mining
                if (0 <= agent_id < num_agents and 
                    0 <= x < grid_size_row and 
                    0 <= y < grid_size_col and
                    0 <= ore_type_encoded < 3 and
                    0 <= num_agents_in_fov < max_num_agents_in_fov and
                    0 <= action_idx < 8):
                    
                    # Store belief embedding (truncate from embedding_dim to belief_target_dim)
                    if 'belief_embedding' in entry and entry['belief_embedding'] is not None:
                        belief_emb = np.array(entry['belief_embedding'], dtype=np.float32)
                        if belief_emb.shape[0] >= belief_target_dim:
                            # Truncate by taking first belief_target_dim elements
                            belief_emb_truncated = belief_emb[:belief_target_dim]
                        else:
                            # Pad with zeros if shorter than target
                            belief_emb_truncated = np.pad(belief_emb, (0, belief_target_dim - belief_emb.shape[0]),
                                                          mode='constant')
                        # Normalize after truncation
                        belief_emb_truncated = normalize_l2(belief_emb_truncated)
                        belief_embedding_table[agent_id, x, y, ore_type_encoded, num_agents_in_fov, action_idx] = belief_emb_truncated
                        filled_count += 1
                    
                    # Store communication embedding (truncate from embedding_dim to comm_target_dim)
                    if 'communication_embedding' in entry and entry['communication_embedding'] is not None:
                        comm_emb = np.array(entry['communication_embedding'], dtype=np.float32)
                        if comm_emb.shape[0] >= comm_target_dim:
                            # Truncate by taking first comm_target_dim elements
                            comm_emb_truncated = comm_emb[:comm_target_dim]
                        else:
                            # Pad with zeros if shorter than target
                            comm_emb_truncated = np.pad(comm_emb, (0, comm_target_dim - comm_emb.shape[0]),
                                                        mode='constant')
                        # Normalize after truncation
                        comm_emb_truncated = normalize_l2(comm_emb_truncated)
                        comm_embedding_table[agent_id, x, y, ore_type_encoded, num_agents_in_fov, action_idx] = comm_emb_truncated
            else:
                # Backward compatibility for coins
                color_encoded, x, y, coin_color_encoded, action = key_vector
                
                # Bounds checking
                if (0 <= color_encoded < 2 and 
                    0 <= x < grid_size_row and 
                    0 <= y < grid_size_col and
                    0 <= coin_color_encoded < 3 and
                    0 <= action < 7):
                    
                    # Store belief embedding (truncate from embedding_dim to belief_target_dim)
                    if 'belief_embedding' in entry and entry['belief_embedding'] is not None:
                        belief_emb = np.array(entry['belief_embedding'], dtype=np.float32)
                        if belief_emb.shape[0] >= belief_target_dim:
                            # Truncate by taking first belief_target_dim elements
                            belief_emb_truncated = belief_emb[:belief_target_dim]
                        else:
                            # Pad with zeros if shorter than target
                            belief_emb_truncated = np.pad(belief_emb, (0, belief_target_dim - belief_emb.shape[0]),
                                                          mode='constant')
                        # Normalize after truncation
                        belief_emb_truncated = normalize_l2(belief_emb_truncated)
                        belief_embedding_table[color_encoded, x, y, coin_color_encoded, action] = belief_emb_truncated
                        filled_count += 1
                    
                    # Store communication embedding (truncate from embedding_dim to comm_target_dim)
                    if 'communication_embedding' in entry and entry['communication_embedding'] is not None:
                        comm_emb = np.array(entry['communication_embedding'], dtype=np.float32)
                        if comm_emb.shape[0] >= comm_target_dim:
                            # Truncate by taking first comm_target_dim elements
                            comm_emb_truncated = comm_emb[:comm_target_dim]
                        else:
                            # Pad with zeros if shorter than target
                            comm_emb_truncated = np.pad(comm_emb, (0, comm_target_dim - comm_emb.shape[0]),
                                                        mode='constant')
                        # Normalize after truncation
                        comm_emb_truncated = normalize_l2(comm_emb_truncated)
                        comm_embedding_table[color_encoded, x, y, coin_color_encoded, action] = comm_emb_truncated
        
        print(f"  Filled {filled_count} entries in embedding tables")
        
        # Convert to JAX arrays
        belief_embedding_table_jax = jnp.array(belief_embedding_table, dtype=jnp.float32)
        comm_embedding_table_jax = jnp.array(comm_embedding_table, dtype=jnp.float32)
        
        if env_name == "CoopMining" or "coop_mining" in env_name.lower():
            def construct_semantic_key_vector(agent_id, agent_x, agent_y, closest_ore_type, num_agents_in_fov, action_idx):
                """
                Construct semantic key vector from agent state for coop_mining.
                
                Args:
                    agent_id: int, agent ID (0 to num_agents-1)
                    agent_x: int, agent x position (row)
                    agent_y: int, agent y position (col)
                    closest_ore_type: str or None, closest ore type in FOV ('iron', 'gold', or None/'none')
                    num_agents_in_fov: int, number of other agents in field of view
                    action_idx: int, action index (0-7)
                    
                Returns:
                    Tuple: (agent_id, x, y, ore_type_encoded, num_agents_in_fov, action_idx)
                """
                # Keep agent_id, x, y as integers
                agent_id = int(agent_id)
                x = int(agent_x)
                y = int(agent_y)
                
                # Encode ore type
                if closest_ore_type is None or closest_ore_type == 'none':
                    ore_type_encoded = 0
                else:
                    ore_type_encoded = ore_type_to_idx.get(closest_ore_type.lower(), 0)
                
                # Keep num_agents_in_fov as integer (clip to max)
                num_agents_in_fov_clipped = min(int(num_agents_in_fov), max_num_agents_in_fov - 1)
                
                # Action index
                action = int(action_idx)
                
                return (agent_id, x, y, ore_type_encoded, num_agents_in_fov_clipped, action)
        else:
            # Backward compatibility for coins
            def construct_semantic_key_vector(agent_id, agent_x, agent_y, closest_coin_color, action_idx):
                """
                Construct semantic key vector from agent state.
                
                Args:
                    agent_id: int, agent ID (0=red, 1=green)
                    agent_x: int, agent x position
                    agent_y: int, agent y position
                    closest_coin_color: str or None, closest coin color in FOV ('red', 'green', or None)
                    action_idx: int, action index
                    
                Returns:
                    Tuple: (color_encoded, x, y, coin_color_encoded, action_idx)
                """
                # Encode color: agent 0 is red, agent 1 is green
                color_encoded = 0 if agent_id == 0 else 1
                
                # Keep x, y as integers
                x = int(agent_x)
                y = int(agent_y)
                
                # Encode coin color
                if closest_coin_color is None:
                    coin_color_encoded = 0
                else:
                    coin_color_encoded = coin_color_to_idx.get(closest_coin_color.lower(), 0)
                
                # Action index
                action = int(action_idx)
                
                return (color_encoded, x, y, coin_color_encoded, action)
        
        # Create separate jitted functions for belief and communication lookups
        # (JAX jit doesn't work well with string-based conditionals)
        if env_name == "CoopMining" or "coop_mining" in env_name.lower():
            # For coop_mining
            @jax.jit
            def lookup_belief_embeddings_jax(agent_ids, agent_xs, agent_ys, ore_type_indices, num_agents_in_fov, action_indices):
                """
                JAX-accelerated parallel belief embedding lookup for coop_mining.
                
                Args:
                    agent_ids: (N,) int array, agent IDs (0 to num_agents-1)
                    agent_xs: (N,) int array, agent x positions (row)
                    agent_ys: (N,) int array, agent y positions (col)
                    ore_type_indices: (N,) int array, ore type indices (0=none, 1=iron, 2=gold)
                    num_agents_in_fov: (N,) int array, number of agents in FOV
                    action_indices: (N,) int array, action indices (0-7)
                    
                Returns:
                    (N, embedding_dim) float32 array of embeddings
                """
                # Clamp indices to valid ranges for safe indexing
                agent_ids_clipped = jnp.clip(agent_ids, 0, num_agents - 1)
                x_indices = jnp.clip(agent_xs, 0, grid_size_row - 1)
                y_indices = jnp.clip(agent_ys, 0, grid_size_col - 1)
                ore_indices = jnp.clip(ore_type_indices, 0, 2)
                num_agents_fov_clipped = jnp.clip(num_agents_in_fov, 0, max_num_agents_in_fov - 1)
                action_indices_clipped = jnp.clip(action_indices, 0, 7)
                
                # Vectorized lookup: belief_embedding_table[agent_ids, x_indices, y_indices, ore_indices, num_agents_fov_clipped, action_indices_clipped]
                embeddings = belief_embedding_table_jax[
                    agent_ids_clipped,
                    x_indices,
                    y_indices,
                    ore_indices,
                    num_agents_fov_clipped,
                    action_indices_clipped
                ]  # Shape: (N, belief_target_dim)
                
                return embeddings
            
            @jax.jit
            def lookup_comm_embeddings_jax(agent_ids, agent_xs, agent_ys, ore_type_indices, num_agents_in_fov, action_indices):
                """
                JAX-accelerated parallel communication embedding lookup for coop_mining.
                
                Args:
                    agent_ids: (N,) int array, agent IDs (0 to num_agents-1)
                    agent_xs: (N,) int array, agent x positions (row)
                    agent_ys: (N,) int array, agent y positions (col)
                    ore_type_indices: (N,) int array, ore type indices (0=none, 1=iron, 2=gold)
                    num_agents_in_fov: (N,) int array, number of agents in FOV
                    action_indices: (N,) int array, action indices (0-7)
                    
                Returns:
                    (N, embedding_dim) float32 array of embeddings
                """
                # Clamp indices to valid ranges for safe indexing
                agent_ids_clipped = jnp.clip(agent_ids, 0, num_agents - 1)
                x_indices = jnp.clip(agent_xs, 0, grid_size_row - 1)
                y_indices = jnp.clip(agent_ys, 0, grid_size_col - 1)
                ore_indices = jnp.clip(ore_type_indices, 0, 2)
                num_agents_fov_clipped = jnp.clip(num_agents_in_fov, 0, max_num_agents_in_fov - 1)
                action_indices_clipped = jnp.clip(action_indices, 0, 7)
                
                # Vectorized lookup: comm_embedding_table[agent_ids, x_indices, y_indices, ore_indices, num_agents_fov_clipped, action_indices_clipped]
                embeddings = comm_embedding_table_jax[
                    agent_ids_clipped,
                    x_indices,
                    y_indices,
                    ore_indices,
                    num_agents_fov_clipped,
                    action_indices_clipped
                ]  # Shape: (N, comm_target_dim)
                
                return embeddings
            
            def lookup_embeddings_jax(agent_ids, agent_xs, agent_ys, ore_type_indices, num_agents_in_fov, action_indices, query_type='belief'):
                """
                Wrapper function for JAX-accelerated parallel embedding lookup for coop_mining.
                
                Args:
                    agent_ids: (N,) int array, agent IDs (0 to num_agents-1)
                    agent_xs: (N,) int array, agent x positions (row)
                    agent_ys: (N,) int array, agent y positions (col)
                    ore_type_indices: (N,) int array, ore type indices (0=none, 1=iron, 2=gold)
                    num_agents_in_fov: (N,) int array, number of agents in FOV
                    action_indices: (N,) int array, action indices (0-7)
                    query_type: str, 'belief' or 'communication'
                    
                Returns:
                    (N, target_dim) float32 array of embeddings (belief_target_dim for belief, comm_target_dim for comm)
                """
                if query_type == 'belief':
                    return lookup_belief_embeddings_jax(agent_ids, agent_xs, agent_ys, ore_type_indices, num_agents_in_fov, action_indices)
                else:
                    return lookup_comm_embeddings_jax(agent_ids, agent_xs, agent_ys, ore_type_indices, num_agents_in_fov, action_indices)
            
            # Keep old query function for backward compatibility (non-jitted, single query)
            def query_dataset(agent_id, agent_x, agent_y, closest_ore_type, num_agents_in_fov, action_idx, 
                             query_type='belief'):
                """
                Query dataset for belief or communication embedding (single query, non-jitted) for coop_mining.
                For batch queries, use lookup_embeddings_jax instead.
                
                Args:
                    agent_id: int, agent ID (0 to num_agents-1)
                    agent_x: int, agent x position (row)
                    agent_y: int, agent y position (col)
                    closest_ore_type: str or None, closest ore type in FOV ('iron', 'gold', or None/'none')
                    num_agents_in_fov: int, number of agents in FOV
                    action_idx: int, action index (0-7)
                    query_type: str, 'belief' or 'communication'
                """
                key_vector = construct_semantic_key_vector(
                    agent_id, agent_x, agent_y, closest_ore_type, num_agents_in_fov, action_idx
                )
                
                entry = dataset.get(key_vector, None)
                
                # Get target dimensions from config
                belief_target_dim = config.get("HIDDEN_DIM", 128)
                comm_target_dim = config.get("COMM_DIM", 64)
                
                if entry is None:
                    if query_type == 'belief':
                        return jnp.zeros(belief_target_dim, dtype=jnp.float32)
                    else:
                        return jnp.zeros(comm_target_dim, dtype=jnp.float32)
                
                if query_type == 'belief':
                    embedding = entry.get('belief_embedding', None)
                    target_dim = belief_target_dim
                elif query_type == 'communication':
                    embedding = entry.get('communication_embedding', None)
                    target_dim = comm_target_dim
                else:
                    return jnp.zeros(comm_target_dim, dtype=jnp.float32)
                
                if embedding is None:
                    return jnp.zeros(target_dim, dtype=jnp.float32)
                
                # Truncate embedding to target dimension
                embedding_array = np.array(embedding, dtype=np.float32)
                if embedding_array.shape[0] >= target_dim:
                    embedding_truncated = embedding_array[:target_dim]
                else:
                    # Pad with zeros if shorter than target
                    embedding_truncated = np.pad(embedding_array, (0, target_dim - embedding_array.shape[0]),
                                                 mode='constant')
                
                # Normalize after truncation
                embedding_truncated = normalize_l2(embedding_truncated)
                
                return jnp.array(embedding_truncated, dtype=jnp.float32)
        else:
            # Backward compatibility for coins
            @jax.jit
            def lookup_belief_embeddings_jax(agent_ids, agent_xs, agent_ys, coin_color_indices, action_indices):
                """
                JAX-accelerated parallel belief embedding lookup.
                
                Args:
                    agent_ids: (N,) int array, agent IDs (0=red, 1=green)
                    agent_xs: (N,) int array, agent x positions
                    agent_ys: (N,) int array, agent y positions
                    coin_color_indices: (N,) int array, coin color indices (0=none, 1=red, 2=green)
                    action_indices: (N,) int array, action indices (0-6)
                    
                Returns:
                    (N, embedding_dim) float32 array of embeddings
                """
                # Encode agent IDs to colors: 0 -> 0 (red), 1 -> 1 (green)
                color_indices = agent_ids
                
                # Clamp indices to valid ranges for safe indexing
                color_indices = jnp.clip(color_indices, 0, 1)
                x_indices = jnp.clip(agent_xs, 0, grid_size_row - 1)
                y_indices = jnp.clip(agent_ys, 0, grid_size_col - 1)
                coin_indices = jnp.clip(coin_color_indices, 0, 2)
                action_indices_clipped = jnp.clip(action_indices, 0, 6)
                
                # Vectorized lookup: belief_embedding_table[color_indices, x_indices, y_indices, coin_indices, action_indices_clipped]
                embeddings = belief_embedding_table_jax[
                    color_indices,
                    x_indices,
                    y_indices,
                    coin_indices,
                    action_indices_clipped
                ]  # Shape: (N, belief_target_dim)
                
                return embeddings
            
            @jax.jit
            def lookup_comm_embeddings_jax(agent_ids, agent_xs, agent_ys, coin_color_indices, action_indices):
                """
                JAX-accelerated parallel communication embedding lookup.
                
                Args:
                    agent_ids: (N,) int array, agent IDs (0=red, 1=green)
                    agent_xs: (N,) int array, agent x positions
                    agent_ys: (N,) int array, agent y positions
                    coin_color_indices: (N,) int array, coin color indices (0=none, 1=red, 2=green)
                    action_indices: (N,) int array, action indices (0-6)
                    
                Returns:
                    (N, embedding_dim) float32 array of embeddings
                """
                # Encode agent IDs to colors: 0 -> 0 (red), 1 -> 1 (green)
                color_indices = agent_ids
                
                # Clamp indices to valid ranges for safe indexing
                color_indices = jnp.clip(color_indices, 0, 1)
                x_indices = jnp.clip(agent_xs, 0, grid_size_row - 1)
                y_indices = jnp.clip(agent_ys, 0, grid_size_col - 1)
                coin_indices = jnp.clip(coin_color_indices, 0, 2)
                action_indices_clipped = jnp.clip(action_indices, 0, 6)
                
                # Vectorized lookup: comm_embedding_table[color_indices, x_indices, y_indices, coin_indices, action_indices_clipped]
                embeddings = comm_embedding_table_jax[
                    color_indices,
                    x_indices,
                    y_indices,
                    coin_indices,
                    action_indices_clipped
                ]  # Shape: (N, comm_target_dim)
                
                return embeddings
            
            def lookup_embeddings_jax(agent_ids, agent_xs, agent_ys, coin_color_indices, action_indices, query_type='belief'):
                """
                Wrapper function for JAX-accelerated parallel embedding lookup.
                
                Args:
                    agent_ids: (N,) int array, agent IDs (0=red, 1=green)
                    agent_xs: (N,) int array, agent x positions
                    agent_ys: (N,) int array, agent y positions
                    coin_color_indices: (N,) int array, coin color indices (0=none, 1=red, 2=green)
                    action_indices: (N,) int array, action indices (0-6)
                    query_type: str, 'belief' or 'communication'
                    
                Returns:
                    (N, target_dim) float32 array of embeddings (belief_target_dim for belief, comm_target_dim for comm)
                """
                if query_type == 'belief':
                    return lookup_belief_embeddings_jax(agent_ids, agent_xs, agent_ys, coin_color_indices, action_indices)
                else:
                    return lookup_comm_embeddings_jax(agent_ids, agent_xs, agent_ys, coin_color_indices, action_indices)
            
            # Keep old query function for backward compatibility (non-jitted, single query)
            def query_dataset(agent_id, agent_x, agent_y, closest_coin_color, action_idx, 
                             query_type='belief'):
                """
                Query dataset for belief or communication embedding (single query, non-jitted).
                For batch queries, use lookup_embeddings_jax instead.
                """
                key_vector = construct_semantic_key_vector(
                    agent_id, agent_x, agent_y, closest_coin_color, action_idx
                )
                
                entry = dataset.get(key_vector, None)
                
                # Get target dimensions from config
                belief_target_dim = config.get("HIDDEN_DIM", 128)
                comm_target_dim = config.get("COMM_DIM", 64)
                
                if entry is None:
                    if query_type == 'belief':
                        return jnp.zeros(belief_target_dim, dtype=jnp.float32)
                    else:
                        return jnp.zeros(comm_target_dim, dtype=jnp.float32)
                
                if query_type == 'belief':
                    embedding = entry.get('belief_embedding', None)
                    target_dim = belief_target_dim
                elif query_type == 'communication':
                    embedding = entry.get('communication_embedding', None)
                    target_dim = comm_target_dim
                else:
                    return jnp.zeros(comm_target_dim, dtype=jnp.float32)
                
                if embedding is None:
                    return jnp.zeros(target_dim, dtype=jnp.float32)
                
                # Truncate embedding to target dimension
                embedding_array = np.array(embedding, dtype=np.float32)
                if embedding_array.shape[0] >= target_dim:
                    embedding_truncated = embedding_array[:target_dim]
                else:
                    # Pad with zeros if shorter than target
                    embedding_truncated = np.pad(embedding_array, (0, target_dim - embedding_array.shape[0]),
                                                 mode='constant')
                
                # Normalize after truncation
                embedding_truncated = normalize_l2(embedding_truncated)
                
                return jnp.array(embedding_truncated, dtype=jnp.float32)
        
        result = {
            'dataset': dataset,  # Keep original for reference
            'query': query_dataset,  # Keep for backward compatibility
            'lookup_jax': lookup_embeddings_jax,  # New JAX-accelerated lookup
            'construct_key': construct_semantic_key_vector,
            'embedding_dim': embedding_dim,  # Original stored dimension
            'belief_target_dim': belief_target_dim,  # Truncated dimension for belief
            'comm_target_dim': comm_target_dim,  # Truncated dimension for communication
            'size': len(dataset),
            'grid_size': (grid_size_row, grid_size_col)
        }
        
        # Add coop_mining-specific fields
        if env_name == "CoopMining" or "coop_mining" in env_name.lower():
            result['num_agents'] = num_agents
            result['max_num_agents_in_fov'] = max_num_agents_in_fov
            result['ore_type_to_idx'] = ore_type_to_idx
        
        return result
        
    except Exception as e:
        print(f"Error loading LLM dataset from {data_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_social_influence_reward(belief_states, comm_logits, counterfactuals, 
                                    actual_outputs, config):
    """
    Compute social influence intrinsic reward.
    
    Measures how much an agent's communication changes other agents' behaviors/beliefs.
    Uses different similarity measures based on influence target:
    - For belief: cosine similarity
    - For action: KL divergence (since they are probability distributions)
    
    Args:
        belief_states: (num_envs * num_agents, hidden_dim) - current belief states
        comm_logits: (num_envs * num_agents, num_protos) - communication logits
        counterfactuals: (num_agents, num_protos, num_agents, output_dim)
        actual_outputs: (num_envs * num_agents, output_dim) - actual actions or beliefs
        config: Configuration dict
        
    Returns:
        influence_reward: (num_agents,) - influence reward for each agent
    """
    num_agents = counterfactuals.shape[0]
    influence_target = config.get("INFLUENCE_TARGET", "belief")
    
    # Get communication probabilities
    comm_probs = jax.lax.stop_gradient(jax.nn.softmax(comm_logits, axis=-1))
    
    # Marginalize counterfactuals over own communication
    marginal_predictions = jax.lax.stop_gradient(
        marginalize_over_own_comm(comm_probs, counterfactuals)
    )
    
    # Reshape actual outputs to (num_agents, output_dim)
    actual_outputs_reshaped = jax.lax.stop_gradient(
        actual_outputs.reshape(-1, num_agents, actual_outputs.shape[-1]).mean(axis=0)
    )
    
    # Expand actual outputs to compare with marginal predictions
    # actual_outputs[k, j] should be agent j's actual output
    actual_expanded = jax.lax.stop_gradient(
        jnp.tile(
            jnp.expand_dims(actual_outputs_reshaped, 0), 
            (num_agents, 1, 1)
        )
    )  # (num_agents, num_agents, output_dim)
    
    # Compute influence based on target type
    # if influence_target == "action":
    #     # For action distributions, use KL divergence
    #     # Convert logits to probabilities if needed
    #     marginal_probs = jax.lax.stop_gradient(jax.nn.softmax(marginal_predictions, axis=-1))
    #     actual_probs = jax.lax.stop_gradient(jax.nn.softmax(actual_expanded, axis=-1))
        
    #     # Compute KL divergence: KL(actual || marginal)
    #     # Higher KL = more influence (communication changes the distribution more)
    #     influence = jax.vmap(
    #         lambda pred, actual: jax.vmap(
    #             lambda p, a: compute_kl_divergence(a, p)
    #         )(pred, actual)
    #     )(marginal_probs, actual_probs)
    # else:
    #     # For belief states, use cosine similarity
    #     # 1 - similarity gives influence
    #     sim = jax.vmap(
    #         lambda pred, actual: jax.vmap(
    #             lambda p, a: jnp.dot(p, a) / (jnp.linalg.norm(p) * jnp.linalg.norm(a) + 1e-8)
    #         )(pred, actual)
    #     )(marginal_predictions, actual_expanded)
        
    #     influence = 1.0 - sim
    
    # For action distributions, use KL divergence
    # Convert logits to probabilities if needed
    marginal_probs = jax.lax.stop_gradient(jax.nn.softmax(marginal_predictions, axis=-1))
    actual_probs = jax.lax.stop_gradient(jax.nn.softmax(actual_expanded, axis=-1))
    
    # Compute KL divergence: KL(actual || marginal)
    # Higher KL = more influence (communication changes the distribution more)
    influence = jax.vmap(
        lambda pred, actual: jax.vmap(
            lambda p, a: compute_kl_divergence(a, p)
        )(pred, actual)
    )(marginal_probs, actual_probs)

    # Mask out self-influence (diagonal)
    mask = ~jnp.eye(num_agents, dtype=bool)
    masked_influence = jnp.where(mask, influence, 0.0)
    
    # Average influence over other agents
    influence_reward = masked_influence.sum(axis=1) / (num_agents - 1)
    
    return jax.lax.stop_gradient(influence_reward)


def make_train_comm(config):
    """Training function with communication mechanism"""
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # Note: Communication typically uses parameter sharing, but we support both modes
    if config.get("PARAMETER_SHARING", True):  # Default to True for communication
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    # Load LLM dataset if needed
    llm_dataset = None
    if config.get("SUPERVISED_BELIEF", "none") == "llm" or config.get("SUPERVISED_COMM", "none") == "llm":
        llm_data_path = config.get("LLM_DATA_PATH", "")
        if llm_data_path:
            llm_dataset = load_offline_llm_dataset(llm_data_path, config["ENV_NAME"], config)
            if llm_dataset is None:
                print("Warning: Failed to load LLM dataset. Supervised learning from LLM will be disabled.")
        else:
            print("Warning: LLM_DATA_PATH not set. Supervised learning from LLM will be disabled.")
    config["LLM_DATASET"] = llm_dataset  # Store in config for access during training
    
    env = LogWrapper(env, replace_info=False)
    
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
    
    def train(rng):
        # INIT NETWORK
        if config.get("PARAMETER_SHARING", True):
            network = ActorCriticComm(
                action_dim=env.action_space().n,
                comm_dim=config.get("COMM_DIM", 64),
                num_protos=config.get("NUM_PROTOS", 10),
                hidden_dim=config.get("HIDDEN_DIM", 128),
                activation=config["ACTIVATION"],
                use_tom=config.get("USE_TOM", False),
                use_intrinsic_reward=config.get("USE_INTRINSIC_REWARD", False)
            )
        else:
            network = [ActorCriticComm(
                action_dim=env.action_space().n,
                comm_dim=config.get("COMM_DIM", 64),
                num_protos=config.get("NUM_PROTOS", 10),
                hidden_dim=config.get("HIDDEN_DIM", 128),
                activation=config["ACTIVATION"],
                use_tom=config.get("USE_TOM", False),
                use_intrinsic_reward=config.get("USE_INTRINSIC_REWARD", False)
            ) for _ in range(env.num_agents)]
        
        rng, _rng = jax.random.split(rng)
        init_obs = jnp.zeros((1, *(env.observation_space()[0]).shape))
        init_comm = jnp.zeros((1, config.get("COMM_DIM", 64)))
        init_hidden = jnp.zeros((1, config.get("HIDDEN_DIM", 128)))
        
        if config.get("PARAMETER_SHARING", True):
            network_params = network.init(
                {'params': _rng, 'gumbel': _rng},
                init_obs,
                init_comm,
                init_hidden,
                train_mode=True
            )
        else:
            # Split RNG for each agent to ensure reproducibility
            init_rngs = jax.random.split(_rng, env.num_agents)
            network_params = [network[i].init(
                {'params': init_rngs[i], 'gumbel': init_rngs[i]},
                init_obs,
                init_comm,
                init_hidden,
                train_mode=True
            ) for i in range(env.num_agents)]
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        
        if config.get("PARAMETER_SHARING", True):
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        else:
            train_state = [TrainState.create(
                apply_fn=network[i].apply,
                params=network_params[i],
                tx=tx,
            ) for i in range(env.num_agents)]
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # Initialize hidden states and communication
        hidden_states = jnp.zeros((config["NUM_ENVS"], env.num_agents, config.get("HIDDEN_DIM", 128)))
        prev_comm = jnp.zeros((config["NUM_ENVS"], env.num_agents, config.get("COMM_DIM", 64)))
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, hidden_states, prev_comm, update_step, rng = runner_state
                
                # SELECT ACTION AND GENERATE COMMUNICATION
                rng, _rng = jax.random.split(rng)
                
                if config.get("PARAMETER_SHARING", True):
                    # Reshape observations: (num_envs, num_agents, ...) -> (num_envs * num_agents, ...)
                    obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *(env.observation_space()[0]).shape)
                    hidden_batch = hidden_states.reshape(-1, config.get("HIDDEN_DIM", 128))
                    prev_comm_batch = prev_comm.reshape(-1, config.get("COMM_DIM", 64))
                    
                    # Forward pass through network
                    action_logits, comm_vectors, comm_logits, comm_indices, values, new_hidden_batch, belief_batch, tom_pred_batch = network.apply(
                        train_state.params,
                        obs_batch,
                        prev_comm_batch,
                        hidden_batch,
                        train_mode=True,
                        rngs={'gumbel': _rng}
                    )
                    
                    # Sample actions
                    rng, _rng = jax.random.split(rng)
                    pi = distrax.Categorical(logits=action_logits)
                    actions = pi.sample(seed=_rng)
                    log_probs = pi.log_prob(actions)
                    
                    # Communication log probs
                    comm_pi = distrax.Categorical(logits=comm_logits)
                    comm_log_probs = comm_pi.log_prob(comm_indices)
                    
                    # Reshape back to (num_envs, num_agents, ...)
                    actions_reshaped = actions.reshape(config["NUM_ENVS"], env.num_agents)
                    comm_vectors_reshaped = comm_vectors.reshape(config["NUM_ENVS"], env.num_agents, -1)
                    new_hidden_reshaped = new_hidden_batch.reshape(config["NUM_ENVS"], env.num_agents, -1)
                    
                    # Prepare actions for environment
                    env_act = unbatchify(actions, env.agents, config["NUM_ENVS"], env.num_agents)
                else:
                    # Non-parameter sharing: process each agent separately
                    obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))  # (num_agents, num_envs, ...)
                    env_act = {}
                    log_probs = []
                    values = []
                    comm_vectors_list = []
                    comm_log_probs_list= []
                    comm_indices_list = []
                    new_hidden_list = []
                    action_logits_list = []
                    belief_batch_list = []
                    comm_logits_list = []
                    
                    tom_pred_batch_list = []
                    for i in range(env.num_agents):
                        # Forward pass for agent i
                        action_logits_i, comm_vectors_i, comm_logits_i, comm_index_i, values_i, new_hidden_i, belief_i, tom_pred_i = network[i].apply(
                            train_state[i].params,
                            obs_batch[i],
                            prev_comm[:, i],
                            hidden_states[:, i],
                            train_mode=True,
                            rngs={'gumbel': _rng}
                        )
                        
                        # Sample actions
                        rng, _rng = jax.random.split(rng)
                        pi_i = distrax.Categorical(logits=action_logits_i)
                        action_i = pi_i.sample(seed=_rng)
                        log_probs.append(pi_i.log_prob(action_i))
                        env_act[env.agents[i]] = action_i
                        values.append(values_i)
                        
                        # Communication
                        comm_vectors_list.append(comm_vectors_i)
                        comm_pi_i = distrax.Categorical(logits=comm_logits_i)
                        comm_log_probs_list.append(comm_pi_i.log_prob(comm_index_i))
                        comm_indices_list.append(comm_index_i)
                        new_hidden_list.append(new_hidden_i)
                        
                        # Store for counterfactual computation
                        action_logits_list.append(action_logits_i)
                        belief_batch_list.append(belief_i)
                        comm_logits_list.append(comm_logits_i)
                        tom_pred_batch_list.append(tom_pred_i)
                    
                    # Stack results
                    actions_reshaped = jnp.stack([env_act[env.agents[i]] for i in range(env.num_agents)], axis=1)
                    comm_vectors_reshaped = jnp.stack(comm_vectors_list, axis=1)
                    new_hidden_reshaped = jnp.stack(new_hidden_list, axis=1)
                    obs_batch = obs_batch.reshape(-1, *(env.observation_space()[0]).shape)
                    hidden_batch = hidden_states.reshape(-1, config.get("HIDDEN_DIM", 128))
                    actions = jnp.concatenate([env_act[env.agents[i]] for i in range(env.num_agents)], axis=0)
                    log_probs = jnp.concatenate(log_probs, axis=0)
                    values = jnp.concatenate(values, axis=0)
                    comm_vectors = comm_vectors_reshaped.reshape(-1, config.get("COMM_DIM", 64))
                    comm_log_probs = jnp.concatenate(comm_log_probs_list, axis=0)
                    comm_indices = jnp.concatenate(comm_indices_list, axis=0)
                    
                    # Stack for use in counterfactual computation
                    action_logits_reshaped = action_logits_list  # List for easier indexing
                    # For belief and comm logits, convert to list format as well
                    
                    # Create prev_comm_batch for counterfactual computation
                    prev_comm_batch = prev_comm.reshape(-1, config.get("COMM_DIM", 64))
                
                # Aggregate communication for next step
                aggregated_comm = aggregate_communication(
                    comm_vectors_reshaped,
                    env.num_agents,
                    comm_mode=config.get("COMM_MODE", "avg")
                )
                
                # Compute social influence intrinsic reward (if enabled)
                # In non-PS mode, we need shape (num_agents * NUM_ENVS,) to properly slice per agent
                if config.get("PARAMETER_SHARING", True):
                    influence_reward_batch = jnp.zeros((config["NUM_ACTORS"],))
                else:
                    influence_reward_batch = jnp.zeros((env.num_agents * config["NUM_ENVS"],))
                # Enable intrinsic reward if USE_INTRINSIC_REWARD is True OR if SOCIAL_INFLUENCE_COEFF > 0
                if config.get("USE_INTRINSIC_REWARD", False) or config.get("SOCIAL_INFLUENCE_COEFF", 0.0) > 0.0:
                    if config.get("PARAMETER_SHARING", True):
                        # PARAMETER SHARING CASE:
                        # In parameter sharing, we can directly use the agent's own policy
                        # to compute counterfactual predictions for other agents
                        
                        # Extract prototype embeddings from network parameters
                        proto_embeddings = train_state.params['params']['ProtoLayer_0']['prototypes']
                        
                        # Generate counterfactuals
                        rng, _rng_cf = jax.random.split(rng)
                        counterfactuals = generate_counterfactuals(
                            network=network,
                            params=train_state.params,
                            obs_batch=obs_batch,
                            prev_comm_batch=prev_comm_batch,
                            hidden_batch=hidden_batch,
                            proto_embeddings=proto_embeddings,
                            num_agents=env.num_agents,
                            num_protos=config.get("NUM_PROTOS", 10),
                            comm_dim=config.get("COMM_DIM", 64),
                            config=config,
                            rng=_rng_cf,
                            parameter_sharing=True,
                            tom_predictions=tom_pred_batch
                        )
                        
                        # Determine what to measure influence on
                        if config.get("INFLUENCE_TARGET", "belief") == "action":
                            actual_outputs = action_logits
                        else:
                            # Use belief output from GRU, not hidden state
                            actual_outputs = belief_batch
                        
                        # Compute influence reward
                        influence_reward = compute_social_influence_reward(
                            belief_states=hidden_batch,
                            comm_logits=comm_logits,
                            counterfactuals=counterfactuals,
                            actual_outputs=actual_outputs,
                            config=config
                        )
                        
                        # Expand influence reward to match batch size
                        # Shape: (num_agents,) -> (num_envs, num_agents) -> (num_envs * num_agents,)
                        influence_reward_expanded = jnp.tile(
                            influence_reward, 
                            config["NUM_ENVS"]
                        ).reshape(-1)
                        
                        influence_reward_batch = influence_reward_expanded
                    else:
                        # DECENTRALIZED CASE (NON-PARAMETER SHARING):
                        # Each agent uses other agents' actual policies for counterfactual reasoning
                        
                        # Extract prototype embeddings from each agent's network parameters
                        proto_embeddings = [train_state[i].params['params']['ProtoLayer_0']['prototypes'] 
                                          for i in range(env.num_agents)]
                        
                        # Prepare observations, communications, and hidden states
                        obs_reshaped = obs_batch.reshape(config["NUM_ENVS"], env.num_agents, *obs_batch.shape[1:])
                        prev_comm_reshaped = prev_comm_batch.reshape(config["NUM_ENVS"], env.num_agents, -1)
                        hidden_reshaped = hidden_batch.reshape(config["NUM_ENVS"], env.num_agents, -1)
                        
                        # Flatten back for counterfactual generation
                        obs_flat = obs_reshaped.reshape(-1, *obs_batch.shape[1:])
                        prev_comm_flat = prev_comm_reshaped.reshape(-1, config.get("COMM_DIM", 64))
                        hidden_flat = hidden_reshaped.reshape(-1, hidden_reshaped.shape[-1])
                        
                        # Stack ToM predictions if available
                        tom_pred_flat = None
                        if tom_pred_batch_list and tom_pred_batch_list[0] is not None:
                            tom_pred_stacked = jnp.stack([tom_pred_batch_list[i] for i in range(env.num_agents)], axis=0)
                            tom_pred_flat = tom_pred_stacked.reshape(-1, tom_pred_stacked.shape[-1])
                        
                        # Generate counterfactuals using each agent's own policy
                        rng, _rng_cf = jax.random.split(rng)
                        counterfactuals = generate_counterfactuals(
                            network=network,  # List of networks
                            params=[train_state[i].params for i in range(env.num_agents)],
                            obs_batch=obs_flat,
                            prev_comm_batch=prev_comm_flat,
                            hidden_batch=hidden_flat,
                            proto_embeddings=proto_embeddings,
                            num_agents=env.num_agents,
                            num_protos=config.get("NUM_PROTOS", 10),
                            comm_dim=config.get("COMM_DIM", 64),
                            config=config,
                            rng=_rng_cf,
                            parameter_sharing=False,
                            tom_predictions=tom_pred_flat
                        )
                        
                        # Determine what to measure influence on
                        # Need to get actual outputs from each agent
                        if config.get("INFLUENCE_TARGET", "belief") == "action":
                            # Stack action logits from all agents
                            actual_outputs = jnp.stack([action_logits_reshaped[i] for i in range(env.num_agents)], axis=0)
                            actual_outputs = actual_outputs.reshape(-1, actual_outputs.shape[-1])
                        else:
                            # Stack belief states from all agents
                            actual_outputs = jnp.stack([belief_batch_list[i] for i in range(env.num_agents)], axis=0)
                            actual_outputs = actual_outputs.reshape(-1, actual_outputs.shape[-1])
                        
                        # Stack comm logits from all agents
                        comm_logits_stacked = jnp.stack([comm_logits_list[i] for i in range(env.num_agents)], axis=0)
                        comm_logits_stacked = comm_logits_stacked.reshape(-1, comm_logits_stacked.shape[-1])
                        
                        # Compute influence reward
                        influence_reward = compute_social_influence_reward(
                            belief_states=hidden_flat,
                            comm_logits=comm_logits_stacked,
                            counterfactuals=counterfactuals,
                            actual_outputs=actual_outputs,
                            config=config
                        )
                        
                        # Expand influence reward to match batch size
                        influence_reward_expanded = jnp.tile(
                            influence_reward, 
                            config["NUM_ENVS"]
                        ).reshape(-1)
                        
                        influence_reward_batch = influence_reward_expanded
                
                # Prepare actions for environment
                env_act = [v for v in env_act.values()]
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                # Extract semantic key information (agent positions and closest coin colors)
                # env_state is wrapped by LogWrapper, so access inner state via env_state.env_state
                # env_state.env_state.agent_locs has shape (num_envs, num_agents, 3) where last dim is [x, y, direction]
                actual_env_state = env_state.env_state  # Unwrap LogWrapper state
                agent_locs = actual_env_state.agent_locs  # (num_envs, num_agents, 3)
                grid = actual_env_state.grid  # (num_envs, H, W)
                
                # Extract positions: (num_envs, num_agents, 2) -> (num_envs * num_agents, 2)
                agent_positions = agent_locs[:, :, :2]  # (num_envs, num_agents, 2)
                agent_positions_flat = agent_positions.reshape(-1, 2)  # (num_envs * num_agents, 2)
                
                # Extract directions
                agent_directions = agent_locs[:, :, 2]  # (num_envs, num_agents)
                
                # Find closest ore for each agent using vmap
                def find_ore_for_agent(agent_x, agent_y, agent_dir, grid_single):
                    return find_closest_ore_in_fov_jax(
                        agent_x, agent_y, agent_dir, grid_single
                    )
                
                # Vectorize over environments and agents
                # Reshape for vmap: (num_envs, num_agents, ...)
                closest_ore_types = jax.vmap(
                    jax.vmap(find_ore_for_agent, in_axes=(0, 0, 0, None)), 
                    in_axes=(0, 0, 0, 0)
                )(agent_positions[:, :, 0], agent_positions[:, :, 1], agent_directions, grid)
                # Result: (num_envs, num_agents)
                closest_ore_types_flat = closest_ore_types.reshape(-1)  # (num_envs * num_agents,)
                
                # Store transition
                if config.get("PARAMETER_SHARING", True):
                    # PARAMETER SHARING: rewards are flattened (num_envs * num_agents,)
                    env_reward = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                    
                    # Action policy uses external task reward only
                    action_reward = env_reward
                    
                    # Communication policy uses intrinsic social influence reward
                    comm_reward = config.get("SOCIAL_INFLUENCE_COEFF", 0.0) * influence_reward_batch
                    
                    # Total reward (for value function - combines both)
                    total_reward = action_reward + comm_reward
                    
                    # Store influence reward in info for logging
                    if config.get("SOCIAL_INFLUENCE_COEFF", 0.0) > 0.0:
                        info['social_influence_reward'] = influence_reward_batch.reshape(config["NUM_ENVS"], env.num_agents)
                        info['env_reward_only'] = env_reward.reshape(config["NUM_ENVS"], env.num_agents)
                    
                    info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    
                    # Handle ToM predictions - use zeros if ToM is disabled
                    if tom_pred_batch is not None:
                        tom_pred_for_storage = tom_pred_batch
                    else:
                        tom_pred_for_storage = jnp.zeros_like(belief_batch)
                    
                    transition = TransitionComm(
                        batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                        actions,
                        values,
                        total_reward,
                        action_reward,
                        comm_reward,
                        log_probs,
                        obs_batch,
                        comm_vectors,
                        comm_log_probs,
                        comm_indices,
                        hidden_batch,
                        belief_batch,  # Store belief states for supervised learning
                        tom_pred_for_storage,  # Store ToM predictions (or zeros if disabled)
                        prev_comm_batch,  # Store received/aggregated communication that was used as input
                        info,
                        agent_positions_flat,  # Store agent positions for semantic key
                        closest_ore_types_flat,  # Store closest ore types for semantic key
                    )
                else:
                    # NON-PARAMETER SHARING: rewards are per-agent (num_envs,) for each agent
                    # Store influence reward in info for logging
                    if config.get("SOCIAL_INFLUENCE_COEFF", 0.0) > 0.0:
                        info['social_influence_reward'] = influence_reward_batch.reshape(config["NUM_ENVS"], env.num_agents)
                        info['env_reward_only'] = reward  # Already has shape (num_envs, num_agents)
                    
                    transition = []
                    done_list = [v for v in done.values()]
                    for i in range(env.num_agents):
                        info_i = {key: jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"]), 1), value[:, i]) for key, value in info.items()}
                        
                        # Get rewards for this agent
                        agent_env_reward = reward[:, i]
                        agent_action_reward = agent_env_reward
                        agent_comm_reward = influence_reward_batch[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]] * config.get("SOCIAL_INFLUENCE_COEFF", 0.0)
                        agent_total_reward = agent_action_reward + agent_comm_reward
                        
                        # Get belief and ToM prediction for this agent
                        agent_belief = belief_batch_list[i]
                        if tom_pred_batch_list and tom_pred_batch_list[i] is not None:
                            agent_tom_pred = tom_pred_batch_list[i]
                        else:
                            agent_tom_pred = jnp.zeros_like(agent_belief)
                        
                        # Extract agent positions and ore types for this agent
                        agent_i_positions = agent_positions_flat[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]]
                        agent_i_ore_types = closest_ore_types_flat[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]]
                        
                        transition.append(TransitionComm(
                            done_list[i],
                            env_act[i],
                            values[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],
                            agent_total_reward,
                            agent_action_reward,
                            agent_comm_reward,
                            log_probs[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],
                            obs_batch[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],
                            comm_vectors[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],
                            comm_log_probs[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],
                            comm_indices[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],
                            hidden_batch[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],
                            agent_belief,  # Store belief states for supervised learning
                            agent_tom_pred,  # Store ToM predictions (or zeros if disabled)
                            prev_comm_batch[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],  # Store received/aggregated communication that was used as input
                            info_i,
                            agent_i_positions,  # Store agent positions for semantic key
                            agent_i_ore_types,  # Store closest ore types for semantic key
                        ))
                
                runner_state = (train_state, env_state, obsv, new_hidden_reshaped, aggregated_comm, update_step, rng)
                return runner_state, transition
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, hidden_states, prev_comm, update_step, rng = runner_state
            
            if config.get("PARAMETER_SHARING", True):
                last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *(env.observation_space()[0]).shape)
                last_hidden_batch = hidden_states.reshape(-1, config.get("HIDDEN_DIM", 128))
                last_comm_batch = prev_comm.reshape(-1, config.get("COMM_DIM", 64))
                
                _, _, _, _, last_val, _, _, _ = network.apply(
                    train_state.params,
                    last_obs_batch,
                    last_comm_batch,
                    last_hidden_batch,
                    train_mode=False,
                    rngs={'gumbel': rng}
                )
            else:
                last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
                last_val = []
                for i in range(env.num_agents):
                    _, _, _, _, last_val_i, _, _, _ = network[i].apply(
                        train_state[i].params,
                        last_obs_batch[i],
                        prev_comm[:, i],
                        hidden_states[:, i],
                        train_mode=False,
                        rngs={'gumbel': rng}
                    )
                    last_val.append(last_val_i)
                last_val = jnp.stack(last_val, axis=0)
            
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value
            
            def _calculate_separate_advantages(traj_batch, last_val):
                """Calculate separate advantages for action and communication policies"""
                # Advantages for action policy (using action_reward)
                def _get_action_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, action_reward = (
                        transition.done,
                        transition.value,
                        transition.action_reward,
                    )
                    delta = action_reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, action_advantages = jax.lax.scan(
                    _get_action_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                
                # Advantages for communication policy (using comm_reward)
                def _get_comm_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, comm_reward = (
                        transition.done,
                        transition.value,
                        transition.comm_reward,
                    )
                    delta = comm_reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, comm_advantages = jax.lax.scan(
                    _get_comm_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                
                # Value target uses total reward
                def _get_value_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, value_advantages = jax.lax.scan(
                    _get_value_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                
                targets = value_advantages + traj_batch.value
                return action_advantages, comm_advantages, targets
            
            # Use separate advantages if we want to train action and comm separately
            use_separate_rewards = config.get("USE_SEPARATE_REWARDS", True)
            
            if config.get("PARAMETER_SHARING", True):
                if use_separate_rewards:
                    action_advantages, comm_advantages, targets = _calculate_separate_advantages(traj_batch, last_val)
                else:
                    advantages, targets = _calculate_gae(traj_batch, last_val)
                    action_advantages = advantages
                    comm_advantages = advantages
            else:
                action_advantages_list = []
                comm_advantages_list = []
                targets = []
                for i in range(env.num_agents):
                    if use_separate_rewards:
                        action_adv_i, comm_adv_i, targets_i = _calculate_separate_advantages(traj_batch[i], last_val[i])
                        action_advantages_list.append(action_adv_i)
                        comm_advantages_list.append(comm_adv_i)
                    else:
                        advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i])
                        action_advantages_list.append(advantages_i)
                        comm_advantages_list.append(advantages_i)
                    targets.append(targets_i)
                action_advantages = jnp.stack(action_advantages_list, axis=0)
                comm_advantages = jnp.stack(comm_advantages_list, axis=0)
                targets = jnp.stack(targets, axis=0)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    traj_batch, action_adv, comm_adv, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, action_gae, comm_gae, targets, network_used, rng):
                        # RERUN NETWORK
                        # Use stored prev_comm (received/aggregated communication) directly
                        batch_size = traj_batch.obs.shape[0]
                        comm_dim = config.get("COMM_DIM", 64)
                        hidden_dim = config.get("HIDDEN_DIM", 128)
                        
                        action_logits, _, comm_logits, _, values, _, belief_recomputed, tom_pred_recomputed = network_used.apply(
                            params,
                            traj_batch.obs,
                            traj_batch.prev_comm,  # Use stored received/aggregated communication
                            traj_batch.hidden_state,
                            train_mode=True,
                            rngs={'gumbel': rng}
                        )
                        
                        # Action policy
                        pi = distrax.Categorical(logits=action_logits)
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # Communication policy
                        comm_pi = distrax.Categorical(logits=comm_logits)
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            values - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(values - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        
                        # CALCULATE ACTION POLICY LOSS (using action_gae based on external rewards)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        action_gae_normalized = (action_gae - action_gae.mean()) / (action_gae.std() + 1e-8)
                        loss_actor1 = ratio * action_gae_normalized
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * action_gae_normalized
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        # 2. Calculate log_prob of the actions we TOOK (traj_batch.comm_action)
                        # This corresponds to 'comm_log_probs' in your torch code, but calculated properly
                        new_comm_log_prob = comm_pi.log_prob(traj_batch.comm_index)
                        
                        # 3. Calculate Ratio (pi_new / pi_old)
                        # exp(log(a) - log(b)) = a / b
                        comm_ratio = jnp.exp(new_comm_log_prob - traj_batch.comm_log_prob)
                        
                        # 4. Normalize Advantages (Crucial for stability in PPO)
                        comm_gae = (comm_gae - comm_gae.mean()) / (comm_gae.std() + 1e-8)
                        
                        # 5. Calculate PPO Clipped Loss
                        # Loss 1: Unclipped
                        loss_comm1 = comm_ratio * comm_gae
                        
                        # Loss 2: Clipped
                        loss_comm2 = jnp.clip(
                            comm_ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]
                        ) * comm_gae
                        
                        # Maximize advantage (minimize negative min)
                        loss_comm = -jnp.minimum(loss_comm1, loss_comm2).mean()
                        
                        comm_entropy = comm_pi.entropy().mean()

                        # SUPERVISED LEARNING LOSS
                        # Uses configurable cosine similarity or MSE loss for both belief and communication supervision
                        supervised_loss = 0.0
                        supervised_loss_type = config.get("SUPERVISED_LOSS_TYPE", "cosine")
                        llm_belief_non_zero = jnp.array(0.0, dtype=jnp.float32)
                        llm_belief_total = jnp.array(0.0, dtype=jnp.float32)
                        llm_comm_non_zero = jnp.array(0.0, dtype=jnp.float32)
                        llm_comm_total = jnp.array(0.0, dtype=jnp.float32)
                        
                        # Supervised belief loss (requires USE_TOM = True)
                        if config.get("USE_TOM", False):
                            supervised_belief = config.get("SUPERVISED_BELIEF", "none")
                            
                            # Supervised belief loss (configurable similarity/MSE)
                            if supervised_belief == "ground_truth" and tom_pred_recomputed is not None:
                                # Reshape to (num_envs, num_agents, hidden_dim)
                                num_envs = batch_size // env.num_agents
                                belief_reshaped = traj_batch.belief_state.reshape(num_envs, env.num_agents, hidden_dim)
                                tom_pred_reshaped = tom_pred_recomputed.reshape(num_envs, env.num_agents, hidden_dim)
                                
                                # For each agent, predict other agents' beliefs
                                # Ground truth: belief_reshaped[:, j] for agent j
                                # ToM prediction from agent i: tom_pred_reshaped[:, i]
                                # We want agent i's ToM to predict all other agents' beliefs
                                
                                # Expand to compare each agent's ToM pred with each other agent's belief
                                # Use cross-agent supervision: agent i predicts agent j's belief (i != j)
                                tom_expanded = jnp.expand_dims(tom_pred_reshaped, 2)  # (num_envs, num_agents, 1, hidden_dim)
                                belief_expanded = jnp.expand_dims(belief_reshaped, 1)  # (num_envs, 1, num_agents, hidden_dim)
                                
                                # Compute supervised loss according to configured metric
                                belief_loss = _compute_supervised_loss(
                                    tom_expanded, belief_expanded, supervised_loss_type
                                )  # (num_envs, num_agents, num_agents)
                                
                                # Mask out self-prediction (diagonal)
                                mask = 1.0 - jnp.eye(env.num_agents)  # (num_agents, num_agents)
                                mask = jnp.expand_dims(mask, 0)  # (1, num_agents, num_agents)
                                
                                masked_belief_loss = belief_loss * mask
                                supervised_loss += jnp.mean(masked_belief_loss) * config.get("SUPERVISED_LOSS_COEF", 0.1)
                            
                            elif supervised_belief == "llm":
                                # LLM dataset supervision with JAX-accelerated parallel lookups
                                llm_dataset = config.get("LLM_DATASET", None)
                                if llm_dataset is not None and tom_pred_recomputed is not None:
                                    # Reshape to (num_envs, num_agents, hidden_dim)
                                    num_envs = batch_size // env.num_agents
                                    tom_pred_reshaped = tom_pred_recomputed.reshape(num_envs, env.num_agents, hidden_dim)
                                    
                                    # Get stored semantic key information
                                    agent_positions = traj_batch.agent_positions  # (batch_size, 2)
                                    closest_ore_types = traj_batch.closest_ore_types  # (batch_size,)
                                    actions = traj_batch.action  # (batch_size,)
                                    
                                    # Reshape to (num_envs, num_agents, ...)
                                    agent_positions_reshaped = agent_positions.reshape(num_envs, env.num_agents, 2)
                                    closest_ore_types_reshaped = closest_ore_types.reshape(num_envs, env.num_agents)
                                    actions_reshaped = actions.reshape(num_envs, env.num_agents)
                                    
                                    # Use JAX-accelerated parallel lookup for all agents and environments at once
                                    # Flatten to (num_envs * num_agents,) for batch lookup
                                    # agent_ids: for each environment, we have agents 0, 1, ..., num_agents-1
                                    # So we need: [0, 1, ..., num_agents-1, 0, 1, ..., num_agents-1, ...] (repeated num_envs times)
                                    agent_ids_flat = jnp.tile(jnp.arange(env.num_agents), num_envs)  # (num_envs * num_agents,)
                                    agent_xs_flat = agent_positions_reshaped[:, :, 0].flatten()  # (num_envs * num_agents,)
                                    agent_ys_flat = agent_positions_reshaped[:, :, 1].flatten()  # (num_envs * num_agents,)
                                    ore_type_indices_flat = closest_ore_types_reshaped.flatten()  # (num_envs * num_agents,)
                                    action_indices_flat = actions_reshaped.flatten()  # (num_envs * num_agents,)
                                    
                                    # Parallel lookup: all queries processed in one GPU call
                                    target_beliefs_flat = llm_dataset['lookup_jax'](
                                        agent_ids=agent_ids_flat,
                                        agent_xs=agent_xs_flat.astype(jnp.int32),
                                        agent_ys=agent_ys_flat.astype(jnp.int32),
                                        ore_type_indices=ore_type_indices_flat.astype(jnp.int32),
                                        num_agents_in_fov=jnp.zeros_like(ore_type_indices_flat).astype(jnp.int32),  # TODO: add num_agents_in_fov tracking
                                        action_indices=action_indices_flat.astype(jnp.int32),
                                        query_type='belief'
                                    )  # (num_envs * num_agents, embedding_dim)
                                    
                                    # Reshape back to (num_envs, num_agents, embedding_dim)
                                    target_beliefs = target_beliefs_flat.reshape(num_envs, env.num_agents, -1)
                                    
                                    # For ToM: agent i predicts agent j's belief (i != j)
                                    # Expand to compare each agent's ToM pred with each other agent's target belief
                                    tom_expanded = jnp.expand_dims(tom_pred_reshaped, 2)  # (num_envs, num_agents, 1, hidden_dim)
                                    target_expanded = jnp.expand_dims(target_beliefs, 1)  # (num_envs, 1, num_agents, embedding_dim)
                                    
                                    # Compute supervised loss according to configured metric
                                    belief_loss = _compute_supervised_loss(
                                        tom_expanded, target_expanded, supervised_loss_type
                                    )  # (num_envs, num_agents, num_agents)
                                    
                                    # Mask out self-prediction (diagonal)
                                    mask = 1.0 - jnp.eye(env.num_agents)  # (num_agents, num_agents)
                                    mask = jnp.expand_dims(mask, 0)  # (1, num_agents, num_agents)
                                    
                                    masked_belief_loss = belief_loss * mask
                                    supervised_loss += jnp.mean(masked_belief_loss) * config.get("SUPERVISED_LOSS_COEF", 0.1)
                                    
                                    # Track number of LLM queries that produced non-zero responses
                                    belief_total_queries = jnp.array(target_beliefs_flat.shape[0], dtype=jnp.float32)
                                    belief_non_zero_queries = _count_non_zero_queries(target_beliefs_flat)
                                    llm_belief_non_zero = llm_belief_non_zero + belief_non_zero_queries
                                    llm_belief_total = llm_belief_total + belief_total_queries
                                else:
                                    if llm_dataset is None:
                                        print("Warning: LLM dataset not available for belief supervision")
                        
                        # Supervised communication loss (requires USE_TOM = False)
                        # Each agent aligns their own communication vector with LLM ground truth
                        supervised_comm = config.get("SUPERVISED_COMM", "none")
                        if supervised_comm == "llm":
                            # LLM dataset supervision for communication with JAX-accelerated parallel lookups
                            llm_dataset = config.get("LLM_DATASET", None)
                            if llm_dataset is not None:
                                num_envs = batch_size // env.num_agents
                                # Reshape communication vectors: (batch_size, comm_dim) -> (num_envs, num_agents, comm_dim)
                                comm_reshaped = traj_batch.comm_vector.reshape(num_envs, env.num_agents, comm_dim)
                                
                                # Get stored semantic key information
                                agent_positions = traj_batch.agent_positions  # (batch_size, 2)
                                closest_ore_types = traj_batch.closest_ore_types  # (batch_size,)
                                actions = traj_batch.action  # (batch_size,)
                                
                                # Reshape to (num_envs, num_agents, ...)
                                agent_positions_reshaped = agent_positions.reshape(num_envs, env.num_agents, 2)
                                closest_ore_types_reshaped = closest_ore_types.reshape(num_envs, env.num_agents)
                                actions_reshaped = actions.reshape(num_envs, env.num_agents)
                                
                                # Use JAX-accelerated parallel lookup for all agents and environments at once
                                # Flatten to (num_envs * num_agents,) for batch lookup
                                agent_ids_flat = jnp.tile(jnp.arange(env.num_agents), num_envs)  # (num_envs * num_agents,)
                                agent_xs_flat = agent_positions_reshaped[:, :, 0].flatten()  # (num_envs * num_agents,)
                                agent_ys_flat = agent_positions_reshaped[:, :, 1].flatten()  # (num_envs * num_agents,)
                                ore_type_indices_flat = closest_ore_types_reshaped.flatten()  # (num_envs * num_agents,)
                                action_indices_flat = actions_reshaped.flatten()  # (num_envs * num_agents,)
                                
                                # Parallel lookup: all queries processed in one GPU call
                                target_comms_flat = llm_dataset['lookup_jax'](
                                    agent_ids=agent_ids_flat,
                                    agent_xs=agent_xs_flat.astype(jnp.int32),
                                    agent_ys=agent_ys_flat.astype(jnp.int32),
                                    ore_type_indices=ore_type_indices_flat.astype(jnp.int32),
                                    num_agents_in_fov=jnp.zeros_like(ore_type_indices_flat).astype(jnp.int32),  # TODO: add num_agents_in_fov tracking
                                    action_indices=action_indices_flat.astype(jnp.int32),
                                    query_type='communication'
                                )  # (num_envs * num_agents, embedding_dim)
                                
                                # Reshape back to (num_envs, num_agents, embedding_dim)
                                target_comms = target_comms_flat.reshape(num_envs, env.num_agents, -1)
                                
                                # Compute supervised loss according to configured metric
                                comm_loss = _compute_supervised_loss(
                                    comm_reshaped, target_comms, supervised_loss_type
                                )  # (num_envs, num_agents)
                                supervised_loss += jnp.mean(comm_loss) * config.get("SUPERVISED_LOSS_COEF", 0.1)
                                
                                # Track LLM communication queries returning non-zero responses
                                comm_total_queries = jnp.array(target_comms_flat.shape[0], dtype=jnp.float32)
                                comm_non_zero_queries = _count_non_zero_queries(target_comms_flat)
                                llm_comm_non_zero = llm_comm_non_zero + comm_non_zero_queries
                                llm_comm_total = llm_comm_total + comm_total_queries
                            else:
                                print("Warning: LLM dataset not available for communication supervision")
                        
                        total_loss = (
                            loss_actor
                            + config.get("COMM_LOSS_COEF", 0.1) * loss_comm  # Separate coefficient for comm loss
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            - config["ENT_COEF"] * comm_entropy
                            + supervised_loss  # Add supervised learning loss
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            loss_comm,
                            entropy,
                            comm_entropy,
                            supervised_loss,
                            llm_belief_non_zero,
                            llm_belief_total,
                            llm_comm_non_zero,
                            llm_comm_total,
                        )
                    
                    rng, _rng = jax.random.split(update_state[-1])
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, loss_components), grads = grad_fn(
                        train_state.params, traj_batch, action_adv, comm_adv, targets, network_used, _rng
                    )
                    # Extract individual loss components from the aux output
                    # loss_components = (value_loss, loss_actor, loss_comm, entropy, comm_entropy, supervised_loss)
                    train_state = train_state.apply_gradients(grads=grads)
                    # Return loss components for logging
                    if loss_components is not None and len(loss_components) >= 10:
                        (
                            _,
                            _,
                            loss_comm_val,
                            _,
                            _,
                            supervised_loss_val,
                            belief_non_zero,
                            belief_total,
                            comm_non_zero,
                            comm_total,
                        ) = loss_components
                        return train_state, (
                            total_loss,
                            loss_comm_val,
                            supervised_loss_val,
                            belief_non_zero,
                            belief_total,
                            comm_non_zero,
                            comm_total,
                        )
                    else:
                        return train_state, (total_loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                
                train_state, traj_batch, action_adv, comm_adv, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, action_adv, comm_adv, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                
                if config.get("PARAMETER_SHARING", True):
                    train_state, loss_info = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                    )
                else:
                    train_state, loss_info = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )
                
                # Extract and average loss components across minibatches
                if isinstance(loss_info, tuple):
                    total_loss = loss_info[0].mean()
                    loss_comm_avg = loss_info[1].mean() if len(loss_info) > 1 else 0.0
                    supervised_loss_avg = loss_info[2].mean() if len(loss_info) > 2 else 0.0
                    belief_non_zero_avg = loss_info[3].mean() if len(loss_info) > 3 else 0.0
                    belief_total_avg = loss_info[4].mean() if len(loss_info) > 4 else 0.0
                    comm_non_zero_avg = loss_info[5].mean() if len(loss_info) > 5 else 0.0
                    comm_total_avg = loss_info[6].mean() if len(loss_info) > 6 else 0.0
                else:
                    total_loss = loss_info.mean()
                    loss_comm_avg = 0.0
                    supervised_loss_avg = 0.0
                    belief_non_zero_avg = 0.0
                    belief_total_avg = 0.0
                    comm_non_zero_avg = 0.0
                    comm_total_avg = 0.0
                
                update_state = (train_state, traj_batch, action_adv, comm_adv, targets, rng)
                return update_state, (
                    total_loss,
                    loss_comm_avg,
                    supervised_loss_avg,
                    belief_non_zero_avg,
                    belief_total_avg,
                    comm_non_zero_avg,
                    comm_total_avg,
                )
            
            if config.get("PARAMETER_SHARING", True):
                update_state = (train_state, traj_batch, action_advantages, comm_advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                metric = traj_batch.info
                # Extract loss components averaged across epochs
                if isinstance(loss_info, tuple) and len(loss_info) >= 3:
                    metric['loss'] = loss_info[0].mean()
                    metric['comm_loss'] = loss_info[1].mean()
                    metric['supervised_loss'] = loss_info[2].mean()
                    if len(loss_info) >= 5:
                        belief_non_zero = loss_info[3].mean()
                        belief_total = loss_info[4].mean()
                        metric['llm_supervision/belief_non_zero_queries'] = belief_non_zero
                        metric['llm_supervision/belief_total_queries'] = belief_total
                        metric['llm_supervision/belief_non_zero_rate'] = jnp.where(
                            belief_total > 0, belief_non_zero / (belief_total + 1e-8), 0.0
                        )
                    if len(loss_info) >= 7:
                        comm_non_zero = loss_info[5].mean()
                        comm_total = loss_info[6].mean()
                        metric['llm_supervision/comm_non_zero_queries'] = comm_non_zero
                        metric['llm_supervision/comm_total_queries'] = comm_total
                        metric['llm_supervision/comm_non_zero_rate'] = jnp.where(
                            comm_total > 0, comm_non_zero / (comm_total + 1e-8), 0.0
                        )
                else:
                    metric['loss'] = loss_info[0].mean() if isinstance(loss_info, tuple) else loss_info.mean()
                rng = update_state[-1]
            else:
                update_state_dict = []
                metric = []
                # Split RNG for each agent to ensure reproducibility
                agent_rngs = jax.random.split(rng, env.num_agents)
                for i in range(env.num_agents):
                    update_state = (train_state[i], traj_batch[i], action_advantages[i], comm_advantages[i], targets[i], agent_rngs[i])
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]
                    metric_i = traj_batch[i].info
                    if isinstance(loss_info, tuple) and len(loss_info) >= 3:
                        metric_i['loss'] = loss_info[0].mean()
                        metric_i['comm_loss'] = loss_info[1].mean()
                        metric_i['supervised_loss'] = loss_info[2].mean()
                        if len(loss_info) >= 5:
                            belief_non_zero = loss_info[3].mean()
                            belief_total = loss_info[4].mean()
                            metric_i['llm_supervision/belief_non_zero_queries'] = belief_non_zero
                            metric_i['llm_supervision/belief_total_queries'] = belief_total
                            metric_i['llm_supervision/belief_non_zero_rate'] = jnp.where(
                                belief_total > 0, belief_non_zero / (belief_total + 1e-8), 0.0
                            )
                        if len(loss_info) >= 7:
                            comm_non_zero = loss_info[5].mean()
                            comm_total = loss_info[6].mean()
                            metric_i['llm_supervision/comm_non_zero_queries'] = comm_non_zero
                            metric_i['llm_supervision/comm_total_queries'] = comm_total
                            metric_i['llm_supervision/comm_non_zero_rate'] = jnp.where(
                                comm_total > 0, comm_non_zero / (comm_total + 1e-8), 0.0
                            )
                    else:
                        metric_i['loss'] = loss_info[0].mean() if isinstance(loss_info, tuple) else loss_info.mean()
                    metric.append(metric_i)
                # Combine RNGs from all agents deterministically
                rng = update_state_dict[-1][-1]  # Use the last agent's RNG as the final RNG
            
            def callback(metric):
                # Convert all JAX arrays to Python scalars for wandb
                metric_python = {}
                for key, value in metric.items():
                    try:
                        # Convert JAX arrays to Python scalars
                        if hasattr(value, 'item'):
                            metric_python[key] = value.item()
                        elif isinstance(value, (jnp.ndarray, np.ndarray)):
                            metric_python[key] = float(value)
                        else:
                            metric_python[key] = value
                    except (ValueError, TypeError):
                        # Skip if conversion fails
                        continue
                wandb.log(metric_python)
            
            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            if config.get("PARAMETER_SHARING", True):
                metric["update_step"] = update_step
                metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            else:
                for i in range(env.num_agents):
                    metric[i]["update_step"] = update_step
                    metric[i]["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                metric = metric[0]
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            # Coin-specific metric (skip for coop_mining)
            if "eat_own_coins" in metric:
                metric["eat_own_coins"] = metric["eat_own_coins"] * config["ENV_KWARGS"]["num_inner_steps"]
            
            # Coop_mining-specific metrics: gold and iron ore mined
            if "mining_gold" in metric:
                metric["ore_mined/gold"] = metric["mining_gold"]
            if "mining_iron" in metric:
                metric["ore_mined/iron"] = metric["mining_iron"]
            
            # Log social influence rewards separately if enabled
            if config.get("SOCIAL_INFLUENCE_COEFF", 0.0) > 0.0:
                try:
                    if "social_influence_reward" in metric:
                        # Convert to Python scalar for WandB
                        intrinsic_reward_val = metric["social_influence_reward"]
                        metric["intrinsic_reward/social_influence"] = intrinsic_reward_val
                        metric["intrinsic_reward"] = intrinsic_reward_val  # Also log as intrinsic_reward
                    if "env_reward_only" in metric:
                        # Convert to Python scalar for WandB
                        metric["extrinsic_reward/environment"] = metric["env_reward_only"]
                    # Log the coefficient for reference
                    metric["intrinsic_reward/influence_coeff"] = config.get("SOCIAL_INFLUENCE_COEFF", 0.0)
                    metric["intrinsic_reward/influence_target"] = 1.0 if config.get("INFLUENCE_TARGET", "belief") == "belief" else 0.0
                except (KeyError, TypeError) as e:
                    # If metrics don't exist yet or can't be converted, skip logging them
                    pass
            
            # Log supervised_loss and comm_loss if available
            try:
                if "supervised_loss" in metric:
                    metric["supervised_loss"] = float(metric["supervised_loss"])
                if "comm_loss" in metric:
                    metric["comm_loss"] = float(metric["comm_loss"])
            except (KeyError, TypeError) as e:
                # If metrics don't exist yet or can't be converted, skip logging them
                pass
            
            # Log all coefficient values for hyperparameter sweep analysis
            try:
                metric["hyperparameters/supervised_loss_coef"] = float(config.get("SUPERVISED_LOSS_COEF", 1.0))
                metric["hyperparameters/comm_loss_coef"] = float(config.get("COMM_LOSS_COEF", 1.0))
                metric["hyperparameters/social_influence_coeff"] = float(config.get("SOCIAL_INFLUENCE_COEFF", 0.0))
            except (KeyError, TypeError) as e:
                # If config values don't exist, skip logging them
                pass
            
            jax.debug.callback(callback, metric)
            
            runner_state = (train_state, env_state, last_obs, hidden_states, prev_comm, update_step, rng)
            return runner_state, metric
        
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, hidden_states, prev_comm, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}
    
    return train


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["PARAMETER_SHARING"]:
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Load LLM dataset if needed
    llm_dataset = None
    if config.get("SUPERVISED_BELIEF", "none") == "llm" or config.get("SUPERVISED_COMM", "none") == "llm":
        llm_data_path = config.get("LLM_DATA_PATH", "")
        if llm_data_path:
            llm_dataset = load_offline_llm_dataset(llm_data_path, config["ENV_NAME"], config)
            if llm_dataset is None:
                print("Warning: Failed to load LLM dataset. Supervised learning from LLM will be disabled.")
        else:
            print("Warning: LLM_DATA_PATH not set. Supervised learning from LLM will be disabled.")
    config["LLM_DATASET"] = llm_dataset  # Store in config for access during training

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=0.,
        end_value=1.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    rew_shaping_anneal_org = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        if config["PARAMETER_SHARING"]:
            network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        else:
            network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
        
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        if config["PARAMETER_SHARING"]:
            network_params = network.init(_rng, init_x)
        else:
            # Split RNG for each agent to ensure reproducibility
            init_rngs = jax.random.split(_rng, env.num_agents)
            network_params = [network[i].init(init_rngs[i], init_x) for i in range(env.num_agents)]
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        if config["PARAMETER_SHARING"]:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        else:
            train_state = [TrainState.create(
                apply_fn=network[i].apply,
                params=network_params[i],
                tx=tx,
            ) for i in range(env.num_agents)]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)


                
                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)
                
                if config["PARAMETER_SHARING"]:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                    print("input_obs_shape", obs_batch.shape)
                    pi, value = network.apply(train_state.params, obs_batch)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    env_act = unbatchify(
                        action, env.agents, config["NUM_ENVS"], env.num_agents
                    )
                else:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                    env_act = {}
                    log_prob = []
                    value = []
                    for i in range(env.num_agents):
                        print("input_obs_shape", obs_batch[i].shape)
                        pi, value_i = network[i].apply(train_state[i].params, obs_batch[i])
                        action = pi.sample(seed=_rng)
                        log_prob.append(pi.log_prob(action))
                        env_act[env.agents[i]] = action
                        value.append(value_i)



                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                # shaped_reward = compute_grouped_rewards(reward)
                # reward = jax.tree_util.tree_map(lambda x,y: x*rew_shaping_anneal_org(current_timestep)+y*rew_shaping_anneal(current_timestep), reward, shaped_reward)

                
                if config["PARAMETER_SHARING"]:
                    info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    transition = Transition(
                        batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                        action,
                        value,
                        batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                        log_prob,
                        obs_batch,
                        info,
                        )
                else:
                    transition = []
                    done = [v for v in done.values()]
                    for i in range(env.num_agents):
                        info_i = {key: jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"]),1), value[:,i]) for key, value in info.items()}
                        transition.append(Transition(
                            done[i],
                            env_act[i],
                            value[i],
                            reward[:,i],
                            log_prob[i],
                            obs_batch[i],
                            info_i,
                        ))
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            if config["PARAMETER_SHARING"]:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                _, last_val = network.apply(train_state.params, last_obs_batch)
            else:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                last_val = []
                for i in range(env.num_agents):
                    _, last_val_i = network[i].apply(train_state[i].params, last_obs_batch[i])
                    last_val.append(last_val_i)
                last_val = jnp.stack(last_val, axis=0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # reward_mean = jnp.mean(reward, axis=0)
                    # # reward_std = jnp.std(reward, axis=0) + 1e-8
                    # reward = (reward - reward_mean)# / reward_std
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value
            if config["PARAMETER_SHARING"]:
                advantages, targets = _calculate_gae(traj_batch, last_val)
            else:
                advantages = []
                targets = []
                for i in range(env.num_agents):
                    advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i])
                    advantages.append(advantages_i)
                    targets.append(targets_i)
                advantages = jnp.stack(advantages, axis=0)
                targets = jnp.stack(targets, axis=0)
            # UPDATE NETWORK
            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used):
                        # RERUN NETWORK
                        pi, value = network_used.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)


                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets, network_used
                        )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                # if config["PARAMETER_SHARING"]:
                    
                # else:
                #     batch = jax.tree_util.tree_map(
                #         lambda x: x.reshape((batch_size,) + x.shape[2:]),  # 保持第一个维度为batch_size，自动计算第二个维度
                #         batch
                #     )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                if config["PARAMETER_SHARING"]:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                    )
                else:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            if config["PARAMETER_SHARING"]:
                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
            else:
                update_state_dict = []
                metric = []
                # Split RNG for each agent to ensure reproducibility
                agent_rngs = jax.random.split(rng, env.num_agents)
                for i in range(env.num_agents):
                    update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], agent_rngs[i])
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]
                    metric_i = traj_batch[i].info
                    metric_i['loss'] = loss_info[0]
                    metric.append(metric_i)
                # Combine RNGs from all agents deterministically
                rng = update_state_dict[-1][-1]  # Use the last agent's RNG as the final RNG
                
            def callback(metric):
                # Convert all JAX arrays to Python scalars for wandb
                metric_python = {}
                for key, value in metric.items():
                    try:
                        # Convert JAX arrays to Python scalars
                        if hasattr(value, 'item'):
                            metric_python[key] = value.item()
                        elif isinstance(value, (jnp.ndarray, np.ndarray)):
                            metric_python[key] = float(value)
                        else:
                            metric_python[key] = value
                    except (ValueError, TypeError):
                        # Skip if conversion fails
                        continue
                wandb.log(metric_python)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            if config["PARAMETER_SHARING"]:
                metric["update_step"] = update_step
                metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                # jax.debug.callback(callback, metric)
            else:
                for i in range(env.num_agents):
                    metric[i]["update_step"] = update_step
                    metric[i]["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                metric = metric[0]
                # jax.debug.callback(callback, metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            # Coin-specific metric (skip for coop_mining)
            if "eat_own_coins" in metric:
                metric["eat_own_coins"] = metric["eat_own_coins"] * config["ENV_KWARGS"]["num_inner_steps"]
            
            # Coop_mining-specific metrics: gold and iron ore mined
            if "mining_gold" in metric:
                metric["ore_mined/gold"] = metric["mining_gold"]
            if "mining_iron" in metric:
                metric["ore_mined/iron"] = metric["mining_iron"]
            
            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def single_run(config):
    config = OmegaConf.to_container(config)
    
    use_comm = config.get("USE_COMM", False)
    param_sharing = config.get("PARAMETER_SHARING", False)
    
    # # Build tags
    # if use_comm:
    #     tags = ["LGTOM", "COMM"]
    #     name_suffix = "lgtom_comm"
    # else:
    #     tags = ["IPPO", "FF"]
    #     name_suffix = "ippo_cnn"
    
    # # Add parameter sharing to tags and name
    # if param_sharing:  # Comm always uses parameter sharing
    #     tags.append("PS")
    #     name_suffix += "_ps"
    # else:
    #     tags.append("IND")
    #     name_suffix += "_ind"
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["LGTOM", "COMM", "ToM",'Intrinsic','Ground_truth'],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'social_coop_mining'
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    
    # Select training function based on USE_COMM flag
    if use_comm:
        train_jit = jax.jit(make_train_comm(config))
    else:
        train_jit = jax.jit(make_train(config))
    
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}'
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    
    if use_comm:
        # Communication model with parameter sharing support
        if config.get("PARAMETER_SHARING", True):
            save_path = f"./checkpoints/lgtom/{filename}_ps.pkl"
            save_params(train_state, save_path)
            params = load_params(save_path)
        else:
            # Communication with independent agents (rare case)
            params = []
            for i in range(config['ENV_KWARGS']['num_agents']):
                save_path = f"./checkpoints/lgtom/{filename}_ind_{i}.pkl"
                save_params(train_state[i], save_path)
                params.append(load_params(save_path))
        
        evaluate_comm(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)
    else:
        # Standard IPPO (no communication)
        if config["PARAMETER_SHARING"]:
            save_path = f"./checkpoints/individual/{filename}_ps.pkl"
            save_params(train_state, save_path)
            params = load_params(save_path)
            evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)
        else:
            params = []
            for i in range(config['ENV_KWARGS']['num_agents']):
                save_path = f"./checkpoints/individual/{filename}_ind_{i}.pkl"
                save_params(train_state[i], save_path)
                params.append(load_params(save_path))
            evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)
    # state_seq = get_rollout(train_state.params, config)
    # viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    # viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)

def evaluate_comm(params, env, save_path, config):
    """Evaluation function for communication-based model"""
    rng = jax.random.PRNGKey(0)
    
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False
    
    # Initialize network(s)
    if config.get("PARAMETER_SHARING", True):
        network = ActorCriticComm(
            action_dim=env.action_space().n,
            comm_dim=config.get("COMM_DIM", 64),
            num_protos=config.get("NUM_PROTOS", 10),
            hidden_dim=config.get("HIDDEN_DIM", 128),
            activation=config.get("ACTIVATION", "relu"),
            use_tom=config.get("USE_TOM", False),
            use_intrinsic_reward=config.get("USE_INTRINSIC_REWARD", False)
        )
    else:
        network = [ActorCriticComm(
            action_dim=env.action_space().n,
            comm_dim=config.get("COMM_DIM", 64),
            num_protos=config.get("NUM_PROTOS", 10),
            hidden_dim=config.get("HIDDEN_DIM", 128),
            activation=config.get("ACTIVATION", "relu"),
            use_tom=config.get("USE_TOM", False),
            use_intrinsic_reward=config.get("USE_INTRINSIC_REWARD", False)
        ) for _ in range(env.num_agents)]
    
    # Initialize hidden states and communication
    hidden_states = jnp.zeros((env.num_agents, config.get("HIDDEN_DIM", 128)))
    prev_comm = jnp.zeros((env.num_agents, config.get("COMM_DIM", 64)))
    
    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/coop_mining_comm"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # Stack observations
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        
        # Forward pass
        rng, _rng = jax.random.split(rng)
        
        if config.get("PARAMETER_SHARING", True):
            action_logits, comm_vectors, _, _, _, new_hidden_states, _, _ = network.apply(
                params,
                obs_batch,
                prev_comm,
                hidden_states,
                train_mode=False,
                rngs={'gumbel': _rng}
            )
            
            # Sample actions
            rng, _rng = jax.random.split(rng)
            pi = distrax.Categorical(logits=action_logits)
            actions = pi.sample(seed=_rng)
        else:
            # Non-parameter sharing: process each agent separately
            actions_list = []
            comm_vectors_list = []
            new_hidden_list = []
            
            for i in range(env.num_agents):
                action_logits_i, comm_vectors_i, _, _, _, new_hidden_i, _, _ = network[i].apply(
                    params[i],
                    jnp.expand_dims(obs_batch[i], axis=0),
                    jnp.expand_dims(prev_comm[i], axis=0),
                    jnp.expand_dims(hidden_states[i], axis=0),
                    train_mode=False,
                    rngs={'gumbel': _rng}
                )
                
                rng, _rng = jax.random.split(rng)
                pi_i = distrax.Categorical(logits=action_logits_i)
                action_i = pi_i.sample(seed=_rng)
                
                actions_list.append(action_i.squeeze())
                comm_vectors_list.append(comm_vectors_i.squeeze())
                new_hidden_list.append(new_hidden_i.squeeze())
            
            actions = jnp.stack(actions_list)
            comm_vectors = jnp.stack(comm_vectors_list)
            new_hidden_states = jnp.stack(new_hidden_list)
        
        # Aggregate communication for next step
        comm_expanded = jnp.expand_dims(comm_vectors, axis=0)  # (1, num_agents, comm_dim)
        aggregated_comm = aggregate_communication(
            comm_expanded,
            env.num_agents,
            comm_mode=config.get("COMM_MODE", "avg")
        ).squeeze(0)  # (num_agents, comm_dim)
        
        # Update states
        hidden_states = new_hidden_states
        prev_comm = aggregated_comm
        
        # Convert actions to dict format
        env_act = {env.agents[i]: actions[i].item() for i in range(env.num_agents)}
        
        # Execute actions
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v for v in env_act.values()])
        done = done["__all__"]
        
        # Render
        img = env.render(state)
        pics.append(img)
        
        print('###################')
        print(f'Actions: {env_act}')
        print(f'Reward: {reward}')
        print(f'Comm vectors (first 5 dims): {comm_vectors[:, :5]}')
        print("###################")
    
    # Save GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    n_agents = len(env.agents)
    param_str = "ps" if config.get("PARAMETER_SHARING", True) else "ind"
    gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_frames-{o_t + 1}_comm_{param_str}.gif"
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )

    # Log the GIF to WandB
    print("Logging GIF to WandB")
    wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode with Comm", format="gif")})


def evaluate(params, env, save_path, config):
    rng = jax.random.PRNGKey(0)
    
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False
    
    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/coop_mining"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # 获取所有智能体的观察
        # print(o_t)
        # 使用模型选择动作
        if config["PARAMETER_SHARING"]:
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
            network = ActorCritic(action_dim=env.action_space().n, activation="relu")  # 使用与训练时相同的参数
            pi, _ = network.apply(params, obs_batch)
            rng, _rng = jax.random.split(rng)
            actions = pi.sample(seed=_rng)
            # 转换动作格式
            env_act = {k: v.squeeze() for k, v in unbatchify(
                actions, env.agents, 1, env.num_agents
            ).items()}
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            network = [ActorCritic(action_dim=env.action_space().n, activation="relu") for _ in range(env.num_agents)]
            for i in range(env.num_agents):
                obs = jnp.expand_dims(obs_batch[i],axis=0)
                pi, _ = network[i].apply(params[i], obs)
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action

        
        # 执行动作
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]
        
        # 记录结果
        # episode_reward += sum(reward.values())
        
        # 渲染
        img = env.render(state)
        pics.append(img)
        
        print('###################')
        print(f'Actions: {env_act}')
        print(f'Reward: {reward}')
        # print(f'State: {state.agent_locs}')
        # print(f'State: {state.claimed_indicator_time_matrix}')
        print("###################")
    
    # 保存GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    n_agents = len(env.agents)
    gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_frames-{o_t + 1}.gif"
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )

    # Log the GIF to WandB
    print("Logging GIF to WandB")
    wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})
        
        # print(f"Episode {episode} total reward: {episode_reward}")

def tune(default_config):
    """
    Hyperparameter sweep with wandb for coefficient ablation study.
    
    Sweeps over three hyperparameters:
    - COMM_LOSS_COEF: [0.1, 1.0] - Communication loss coefficient
    - SOCIAL_INFLUENCE_COEFF: [0.1, 1.0] - Intrinsic reward coefficient
    - SUPERVISED_LOSS_COEF: [0.1, 1.0] - Supervised learning loss coefficient
    
    Fixed parameters:
    - USE_COMM: True
    - PARAMETER_SHARING: False (non-parameter-sharing)
    - USE_SEPARATE_REWARDS: False (joint_reward)
    - INFLUENCE_TARGET: "belief"
    - SEED: 110
    - ENV_KWARGS.shared_rewards: False
    - USE_TOM: True
    - SUPERVISED_BELIEF: "llm"
    - USE_INTRINSIC_REWARD: True
    
    Total runs: 2 × 2 × 2 = 8 configurations
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "lgtom_coefficient_sweep",
        "method": "grid",  # Try all combinations
        "program": "lgtom_cnn_coop_mining.py",  # The script to run
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            # Sweep parameters: 2 × 2 × 2 = 8 combinations
            "COMM_LOSS_COEF": {"values": [0.1, 1.0]},  # Communication loss coefficient
            "SOCIAL_INFLUENCE_COEFF": {"values": [0.1, 1.0]},  # Intrinsic reward coefficient
            "SUPERVISED_LOSS_COEF": {"values": [0.1, 1.0]},  # Supervised learning loss coefficient
            
            # Fixed parameters
            "USE_COMM": {"values": [True]},  # Always use communication
            "PARAMETER_SHARING": {"values": [False]},  # Non-parameter-sharing
            "USE_SEPARATE_REWARDS": {"values": [False]},  # Joint reward
            "INFLUENCE_TARGET": {"values": ["belief"]},  # Belief-based influence
            "SEED": {"values": [110]},  # Fixed seed
            "ENV_KWARGS.shared_rewards": {"values": [False]},  # Individual rewards
            "USE_TOM": {"values": [True]},  # Enable ToM
            "SUPERVISED_BELIEF": {"values": ["llm"]},  # Supervised belief with LLM
            "USE_INTRINSIC_REWARD": {"values": [True]},  # Enable intrinsic reward
        },
    }

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        
        # Overwrite config with sweep parameters
        for k, v in dict(wandb.config).items():
            if "." in k:
                # Handle nested keys like ENV_KWARGS.shared_rewards
                parent, child = k.split(".", 1)
                if parent not in config:
                    config[parent] = {}
                config[parent][child] = v
            else:
                config[k] = v
        
        # Ensure fixed settings are set
        config["USE_COMM"] = True
        config["PARAMETER_SHARING"] = False
        config["USE_SEPARATE_REWARDS"] = False
        config["INFLUENCE_TARGET"] = "belief"
        config["USE_TOM"] = True
        config["SUPERVISED_BELIEF"] = "llm"
        config["SUPERVISED_COMM"] = "none"
        config["USE_INTRINSIC_REWARD"] = True
        
        # Ensure LLM dataset path is set (required for LLM supervision)
        if not config.get("LLM_DATA_PATH", ""):
            print("Warning: LLM_DATA_PATH not set. LLM supervision will be disabled.")
        
        # Build descriptive run name
        comm_coef = config.get("COMM_LOSS_COEF", 0.1)
        intrinsic_coef = config.get("SOCIAL_INFLUENCE_COEFF", 0.1)
        supervised_coef = config.get("SUPERVISED_LOSS_COEF", 0.1)
        run_name = f"comm{comm_coef}_intr{intrinsic_coef}_sup{supervised_coef}_s{config['SEED']}"
        wandb.run.name = run_name
        
        tags = ["LGTOM", "COMM", "NON_PS", "JOINT_REWARD", "LLM_BELIEF", "INTRINSIC", "TOM_SUPERVISED", "COEFF_SWEEP"]
        wandb.run.tags = tags
        
        # Log configuration to wandb config
        wandb.config.update({
            "USE_TOM": config.get("USE_TOM", False),
            "SUPERVISED_BELIEF": config.get("SUPERVISED_BELIEF", "none"),
            "SUPERVISED_COMM": config.get("SUPERVISED_COMM", "none"),
            "USE_INTRINSIC_REWARD": config.get("USE_INTRINSIC_REWARD", False),
            "SOCIAL_INFLUENCE_COEFF": config.get("SOCIAL_INFLUENCE_COEFF", 0.0),
            "COMM_LOSS_COEF": config.get("COMM_LOSS_COEF", 0.1),
            "SUPERVISED_LOSS_COEF": config.get("SUPERVISED_LOSS_COEF", 0.1),
        })
        
        print("="*70)
        print(f"Running coefficient sweep experiment: {run_name}")
        print(f"  COMM_LOSS_COEF: {comm_coef}")
        print(f"  SOCIAL_INFLUENCE_COEFF: {intrinsic_coef}")
        print(f"  SUPERVISED_LOSS_COEF: {supervised_coef}")
        print(f"  USE_TOM: {config.get('USE_TOM', False)}")
        print(f"  SUPERVISED_BELIEF: {config.get('SUPERVISED_BELIEF', 'none')}")
        print(f"  USE_INTRINSIC_REWARD: {config.get('USE_INTRINSIC_REWARD', False)}")
        print(f"  PARAMETER_SHARING: {config.get('PARAMETER_SHARING', False)}")
        print(f"  USE_SEPARATE_REWARDS: {config.get('USE_SEPARATE_REWARDS', False)}")
        print(f"  INFLUENCE_TARGET: {config.get('INFLUENCE_TARGET', 'belief')}")
        print(f"  LLM_DATA_PATH: {config.get('LLM_DATA_PATH', 'Not set')}")
        print(f"  SEED: {config['SEED']}")
        print(f"  Total Timesteps: {config['TOTAL_TIMESTEPS']:.0e}")
        print(f"  Tags: {tags}")
        print("="*70)

        # Run training
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        
        # Select appropriate training function (always use communication in this sweep)
        train_fn = make_train_comm(config)
        
        train_vjit = jax.jit(jax.vmap(train_fn))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0])
        
        print(f"Training completed for {run_name}")
        
        # Optional: Save checkpoint and evaluate
        # Uncomment if you want to save models during sweep
        # filename = f"{config['ENV_NAME']}_comm{comm_coef}_intr{intrinsic_coef}_sup{supervised_coef}_seed{config['SEED']}"
        # if config.get("PARAMETER_SHARING", True):
        #     save_path = f"./checkpoints/sweep/{filename}.pkl"
        #     save_params(train_state, save_path)
        # else:
        #     for i in range(config['ENV_KWARGS']['num_agents']):
        #         save_path = f"./checkpoints/sweep/{filename}_{i}.pkl"
        #         save_params(train_state[i], save_path)

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    
    total_runs = 8  # 2 × 2 × 2 = 8 configurations
    
    print("\n" + "="*70)
    print("Starting WandB Sweep: Coefficient Ablation Study")
    print(f"Sweep ID: {sweep_id}")
    print(f"Total Configurations: {total_runs}")
    print(f"\nSweep Parameters:")
    print(f"  - COMM_LOSS_COEF: [0.1, 1.0]")
    print(f"  - SOCIAL_INFLUENCE_COEFF: [0.1, 1.0]")
    print(f"  - SUPERVISED_LOSS_COEF: [0.1, 1.0]")
    print(f"\nFixed Settings:")
    print(f"  - USE_COMM: True (communication enabled)")
    print(f"  - PARAMETER_SHARING: False (non-parameter-sharing)")
    print(f"  - USE_SEPARATE_REWARDS: False (joint_reward)")
    print(f"  - INFLUENCE_TARGET: 'belief' (belief-based influence)")
    print(f"  - SEED: 110")
    print(f"  - ENV_KWARGS.shared_rewards: False (individual rewards)")
    print(f"  - USE_TOM: True")
    print(f"  - SUPERVISED_BELIEF: 'llm' (supervised belief with LLM)")
    print(f"  - USE_INTRINSIC_REWARD: True")
    print(f"\nTimesteps per run: {default_config['TOTAL_TIMESTEPS']:.0e}")
    print(f"Total runs: {total_runs}")
    print("="*70 + "\n")
    
    # Run sweep agent for all configurations
    wandb.agent(sweep_id, wrapped_make_train, count=total_runs)


@hydra.main(version_base=None, config_path="config", config_name="lgtom_cnn_coop_mining")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)
if __name__ == "__main__":
    main()
