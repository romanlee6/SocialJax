""" 
Based on PureJaxRL & jaxmarl Implementation of PPO with AutoEncoder Communication

This implementation includes:
1. AutoEncoder Loss: Reconstructs observation embeddings from communication vectors
   - Uses MSE loss between original embeddings and reconstructed embeddings from comm_vector
   - Encourages communication vectors to encode meaningful information about observations
2. Individual Reward and Individual Policy setup (for fair comparison with LG-TOM)
3. Same model structure as LG-TOM (JAX, CNN, GRU, Prototype Communication) but without ToM and intrinsic rewards

AUTOENCODER MECHANISM:
----------------------
The autoencoder loss encourages agents to learn communication vectors that can reconstruct
their observation embeddings. This creates an information bottleneck that forces agents to
encode only the most relevant information about their observations.

Key Components:
1. Encoder: Observation -> CNN -> Embedding -> GRU -> Belief -> Communication Vector
2. Decoder: Communication Vector -> Decoder Head -> Reconstructed Embedding
3. Reconstruction Loss: MSE(original_embedding, reconstructed_embedding)

Configuration Options:
- AUTOENCODER_LOSS_COEF: Weight for autoencoder reconstruction loss (default: 0.1)
- COMM_LOSS_COEF: Weight for communication policy loss (default: 0.1)
- PARAMETER_SHARING: Whether to share parameters across agents (default: False for individual policies)
- USE_COMM: Always True for AutoEncoder (communication is required)

Implementation Notes:
- Based on LG-TOM architecture for fair comparison
- Uses same setup: individual rewards, individual policies
- No ToM (Theory of Mind) predictions
- No intrinsic rewards (social influence rewards)
- Communication vectors are trained to reconstruct observation embeddings

Reference:
- Based on mac_tom.py decoder implementation
- LG-TOM structure (lgtom_cnn_coins.py)
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
        
        return comm_vector, logits


class ActorCriticComm(nn.Module):
    """Actor-Critic with Communication based on AutoEncoder architecture
    
    Uses autoencoder loss to reconstruct observation embeddings from communication vectors.
    Based on LG-TOM structure but with no ToM and no intrinsic rewards.
    """
    action_dim: int
    comm_dim: int = 64
    num_protos: int = 10
    hidden_dim: int = 128  # Must match embedding_dim (64) + comm_dim (64)
    activation: str = "relu"
    embedding_dim: int = 64  # CNN output dimension

    @nn.compact
    def __call__(self, obs, prev_comm, hidden_state, train_mode=True):
        """
        Args:
            obs: observation (batch, height, width, channels)
            prev_comm: previous communication from other agents (batch, comm_dim)
            hidden_state: GRU hidden state (batch, hidden_dim)
            train_mode: whether in training mode
        Returns:
            action_logits, comm_vector, comm_logits, value, new_hidden_state, belief, reconstructed_embedding
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
        
        # 4. Communication Policy (using prototype layer)
        proto_layer = ProtoLayer(num_protos=self.num_protos, comm_dim=self.comm_dim)
        comm_hidden = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(belief)
        comm_hidden = activation(comm_hidden)
        
        # Get RNG for Gumbel-Softmax if in training mode
        gumbel_rng = self.make_rng('gumbel') if train_mode else None
        comm_vector, comm_logits = proto_layer(comm_hidden, train_mode=train_mode, rng=gumbel_rng)
        
        # 5. Decoder Head: Reconstruct embedding from communication vector
        # Maps from comm_dim (64) back to embedding_dim (64)
        reconstructed_embedding = nn.Dense(
            self.embedding_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name='decoder_head'
        )(comm_vector)
        reconstructed_embedding = activation(reconstructed_embedding)
        
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
        
        # Return belief, new_hidden_state, and reconstructed embedding for autoencoder loss
        # belief: the GRU output used for action/comm generation
        # new_hidden_state: the carry state for next timestep
        # reconstructed_embedding: reconstructed embedding from comm_vector for autoencoder loss
        return action_logits, comm_vector, comm_logits, jnp.squeeze(value, axis=-1), new_hidden_state, belief, reconstructed_embedding


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
    comm_reward: jnp.ndarray    # Communication reward (same as action reward for autoencoder)
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    comm_vector: jnp.ndarray
    comm_log_prob: jnp.ndarray
    hidden_state: jnp.ndarray
    belief_state: jnp.ndarray  # Belief state (GRU output)
    embedding: jnp.ndarray  # Original embedding for autoencoder loss
    reconstructed_embedding: jnp.ndarray  # Reconstructed embedding from comm_vector
    prev_comm: jnp.ndarray  # Received/aggregated communication that was used as input
    info: jnp.ndarray


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
        action_logits_cf, _, _, _, hidden_cf, belief_cf, tom_pred_cf = jax.vmap(
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
            action_logits_i, _, _, _, hidden_i, belief_i, tom_pred_i = jax.vmap(
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
        return action_probs.mean(axis=2)  # (num_agents, num_protos, num_agents, action_dim)
    else:
        # Return belief states: use ToM predictions if enabled, otherwise use actual beliefs
        if use_tom_counterfactuals and tom_pred_cf is not None:
            # Use ToM predictions as counterfactual beliefs
            # This represents each agent's theory of how others' beliefs change with different communications
            return tom_pred_cf.mean(axis=2)  # (num_agents, num_protos, num_agents, hidden_dim)
        else:
            # Use actual belief states (ground truth) as counterfactual beliefs
            return belief_cf.mean(axis=2)  # (num_agents, num_protos, num_agents, hidden_dim)


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


def compute_supervised_belief_loss(tom_predictions, ground_truth_beliefs, config):
    """
    Compute supervised loss for ToM belief predictions using cosine similarity.
    
    Args:
        tom_predictions: (num_envs * num_agents, hidden_dim) - ToM predicted beliefs
        ground_truth_beliefs: (num_envs * num_agents, hidden_dim) - actual belief states of other agents
        config: Configuration dict
        
    Returns:
        loss: scalar supervised loss value (1 - cosine_similarity)
    """
    # Compute cosine similarity
    dot_product = jnp.sum(tom_predictions * ground_truth_beliefs, axis=-1)
    tom_norm = jnp.linalg.norm(tom_predictions, axis=-1) + 1e-8
    belief_norm = jnp.linalg.norm(ground_truth_beliefs, axis=-1) + 1e-8
    cos_sim = dot_product / (tom_norm * belief_norm)
    
    # Loss = 1 - cosine_similarity (minimize to increase similarity)
    loss = jnp.mean(1.0 - cos_sim)
    return loss


def compute_supervised_comm_loss(comm_vectors, target_comm_vectors, config):
    """
    Compute supervised loss for communication based on cosine similarity.
    
    Args:
        comm_vectors: (num_envs * num_agents, comm_dim) - predicted communication vectors
        target_comm_vectors: (num_envs * num_agents, comm_dim) - target communication vectors
        config: Configuration dict
        
    Returns:
        loss: scalar supervised loss value (1 - cosine_similarity)
    """
    # Compute cosine similarity between predicted and target communication vectors
    dot_product = jnp.sum(comm_vectors * target_comm_vectors, axis=-1)
    comm_norm = jnp.linalg.norm(comm_vectors, axis=-1) + 1e-8
    target_norm = jnp.linalg.norm(target_comm_vectors, axis=-1) + 1e-8
    cos_sim = dot_product / (comm_norm * target_norm)
    
    # Loss = 1 - cosine_similarity (minimize to increase similarity)
    loss = jnp.mean(1.0 - cos_sim)
    return loss


def load_offline_llm_dataset(data_path, env_name, config):
    """
    Load offline LLM dataset for supervised communication/belief training.
    
    *** UNDER CONSTRUCTION ***
    This function is a placeholder for loading pre-collected LLM interaction data.
    
    The dataset should contain:
    - State observations
    - Communication embeddings from LLM reasoning
    - Belief state predictions
    - Action distributions
    
    Expected format:
    {
        'observations': array of shape (N, obs_dim),
        'communications': array of shape (N, comm_dim),
        'beliefs': array of shape (N, hidden_dim),
        'actions': array of shape (N, action_dim),
        'state_keys': list of state tuples for indexing
    }
    
    TODO:
    - Define exact data format specification
    - Implement data loading from pickle/numpy files
    - Add data preprocessing and normalization
    - Implement state matching/similarity functions
    - Add caching for efficiency
    
    Args:
        data_path: str, path to offline dataset
        env_name: str, name of environment
        config: Configuration dict
        
    Returns:
        dataset: dict containing offline data, or None if not available
    """
    # PLACEHOLDER IMPLEMENTATION
    print("="*70)
    print("WARNING: Offline LLM dataset loading is UNDER CONSTRUCTION")
    print("This feature is not yet fully implemented.")
    print("To use supervised learning from LLM data:")
    print("  1. Collect LLM trajectories using llms/coins_llm_simulation.py")
    print("  2. Process and format the data into required structure")
    print("  3. Implement data loading logic in this function")
    print("  4. Set SUPERVISED_COMM or SUPERVISED_BELIEF to 'llm' in config")
    print("="*70)
    
    # Return None to indicate dataset not available
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
    comm_probs = jax.nn.softmax(comm_logits, axis=-1)
    
    # Marginalize counterfactuals over own communication
    marginal_predictions = marginalize_over_own_comm(comm_probs, counterfactuals)
    
    # Reshape actual outputs to (num_agents, output_dim)
    actual_outputs_reshaped = actual_outputs.reshape(-1, num_agents, actual_outputs.shape[-1]).mean(axis=0)
    
    # Expand actual outputs to compare with marginal predictions
    # actual_outputs[k, j] should be agent j's actual output
    actual_expanded = jnp.tile(
        jnp.expand_dims(actual_outputs_reshaped, 0), 
        (num_agents, 1, 1)
    )  # (num_agents, num_agents, output_dim)
    
    # Compute influence based on target type
    if influence_target == "action":
        # For action distributions, use KL divergence
        # Convert logits to probabilities if needed
        marginal_probs = jax.nn.softmax(marginal_predictions, axis=-1)
        actual_probs = jax.nn.softmax(actual_expanded, axis=-1)
        
        # Compute KL divergence: KL(actual || marginal)
        # Higher KL = more influence (communication changes the distribution more)
        influence = jax.vmap(
            lambda pred, actual: jax.vmap(
                lambda p, a: compute_kl_divergence(a, p)
            )(pred, actual)
        )(marginal_probs, actual_probs)
    else:
        # For belief states, use cosine similarity
        # 1 - similarity gives influence
        sim = jax.vmap(
            lambda pred, actual: jax.vmap(
                lambda p, a: jnp.dot(p, a) / (jnp.linalg.norm(p) * jnp.linalg.norm(a) + 1e-8)
            )(pred, actual)
        )(marginal_predictions, actual_expanded)
        
        influence = 1.0 - sim
    
    # Mask out self-influence (diagonal)
    mask = ~jnp.eye(num_agents, dtype=bool)
    masked_influence = jnp.where(mask, influence, 0.0)
    
    # Average influence over other agents
    influence_reward = masked_influence.sum(axis=1) / (num_agents - 1)
    
    return influence_reward


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
                embedding_dim=64  # CNN output dimension
            )
        else:
            network = [ActorCriticComm(
                action_dim=env.action_space().n,
                comm_dim=config.get("COMM_DIM", 64),
                num_protos=config.get("NUM_PROTOS", 10),
                hidden_dim=config.get("HIDDEN_DIM", 128),
                activation=config["ACTIVATION"],
                embedding_dim=64  # CNN output dimension
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
            network_params = [network[i].init(
                {'params': _rng, 'gumbel': _rng},
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
                    # Also compute embedding for autoencoder loss
                    embedding_batch = CNN(config["ACTIVATION"]).apply(
                        {'params': train_state.params['params']['CNN_0']},
                        obs_batch
                    )
                    action_logits, comm_vectors, comm_logits, values, new_hidden_batch, belief_batch, reconstructed_embedding = network.apply(
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
                    comm_log_probs = comm_pi.entropy()
                    
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
                    comm_log_probs_list = []
                    new_hidden_list = []
                    action_logits_list = []
                    belief_batch_list = []
                    comm_logits_list = []
                    
                    embedding_batch_list = []
                    reconstructed_embedding_list = []
                    for i in range(env.num_agents):
                        # Forward pass for agent i
                        # Also compute embedding for autoencoder loss
                        embedding_i = CNN(config["ACTIVATION"]).apply(
                            {'params': train_state[i].params['params']['CNN_0']},
                            obs_batch[i]
                        )
                        action_logits_i, comm_vectors_i, comm_logits_i, values_i, new_hidden_i, belief_i, reconstructed_embedding_i = network[i].apply(
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
                        comm_log_probs_list.append(comm_pi_i.entropy())
                        new_hidden_list.append(new_hidden_i)
                        
                        # Store for autoencoder loss
                        action_logits_list.append(action_logits_i)
                        belief_batch_list.append(belief_i)
                        comm_logits_list.append(comm_logits_i)
                        embedding_batch_list.append(embedding_i)
                        reconstructed_embedding_list.append(reconstructed_embedding_i)
                    
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
                    
                    # Stack for use in training
                    action_logits_reshaped = action_logits_list  # List for easier indexing
                    # For belief and comm logits, convert to list format as well
                    
                    # Create prev_comm_batch for transition storage
                    prev_comm_batch = prev_comm.reshape(-1, config.get("COMM_DIM", 64))
                
                # Aggregate communication for next step
                aggregated_comm = aggregate_communication(
                    comm_vectors_reshaped,
                    env.num_agents,
                    comm_mode=config.get("COMM_MODE", "avg")
                )
                
                # Prepare actions for environment
                env_act = [v for v in env_act.values()]
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                # Store transition
                if config.get("PARAMETER_SHARING", True):
                    # PARAMETER SHARING: rewards are flattened (num_envs * num_agents,)
                    env_reward = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                    
                    # Action policy uses external task reward only
                    action_reward = env_reward
                    
                    # Communication policy uses same reward as action policy for autoencoder
                    comm_reward = action_reward
                    
                    # Total reward (for value function - same as action reward)
                    total_reward = action_reward
                    
                    info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    
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
                        hidden_batch,
                        belief_batch,  # Store belief states
                        embedding_batch,  # Store original embeddings for autoencoder loss
                        reconstructed_embedding,  # Store reconstructed embeddings
                        prev_comm_batch,  # Store received/aggregated communication that was used as input
                        info,
                    )
                else:
                    # NON-PARAMETER SHARING: rewards are per-agent (num_envs,) for each agent
                    transition = []
                    done_list = [v for v in done.values()]
                    for i in range(env.num_agents):
                        info_i = {key: jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"]), 1), value[:, i]) for key, value in info.items()}
                        
                        # Get rewards for this agent
                        agent_env_reward = reward[:, i]
                        agent_action_reward = agent_env_reward
                        agent_comm_reward = agent_action_reward  # Same as action reward for autoencoder
                        agent_total_reward = agent_action_reward
                        
                        # Get belief and embeddings for this agent
                        agent_belief = belief_batch_list[i]
                        agent_embedding = embedding_batch_list[i]
                        agent_reconstructed_embedding = reconstructed_embedding_list[i]
                        
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
                            hidden_batch[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],
                            agent_belief,  # Store belief states
                            agent_embedding,  # Store original embeddings for autoencoder loss
                            agent_reconstructed_embedding,  # Store reconstructed embeddings
                            prev_comm_batch[i*config["NUM_ENVS"]:(i+1)*config["NUM_ENVS"]],  # Store received/aggregated communication that was used as input
                            info_i,
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
                
                _, _, _, last_val, _, _, _ = network.apply(
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
                    _, _, _, last_val_i, _, _, _ = network[i].apply(
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
                        
                        # Also compute embedding for autoencoder loss
                        embedding_batch = CNN(config["ACTIVATION"]).apply(
                            {'params': params['params']['CNN_0']},
                            traj_batch.obs
                        )
                        action_logits, _, comm_logits, values, _, belief_recomputed, reconstructed_embedding = network_used.apply(
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
                        # Get the actual selected comm (from stored comm_log_prob, we can't directly get it)
                        # We'll use the entropy-based loss for comm policy
                        
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
                        
                        # CALCULATE COMMUNICATION POLICY LOSS
                        # For communication, we use a policy gradient approach based on the comm advantages
                        # Since comm uses Gumbel-Softmax, we optimize the logits directly
                        comm_gae_normalized = (comm_gae - comm_gae.mean()) / (comm_gae.std() + 1e-8)
                        comm_entropy = comm_pi.entropy().mean()
                        # Loss for comm: negative advantage-weighted entropy (encourages high-reward comms)
                        loss_comm = -comm_gae_normalized.mean() * comm_entropy
                        
                        # CALCULATE AUTOENCODER RECONSTRUCTION LOSS
                        # MSE loss between original embedding and reconstructed embedding from comm_vector
                        autoencoder_loss = jnp.mean(jnp.square(embedding_batch - reconstructed_embedding))
                        
                        total_loss = (
                            loss_actor
                            + config.get("COMM_LOSS_COEF", 0.1) * loss_comm  # Separate coefficient for comm loss
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config.get("AUTOENCODER_LOSS_COEF", 0.1) * autoencoder_loss  # Autoencoder reconstruction loss
                        )
                        return total_loss, (value_loss, loss_actor, loss_comm, entropy, comm_entropy, autoencoder_loss)
                    
                    rng, _rng = jax.random.split(update_state[-1])
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, action_adv, comm_adv, targets, network_used, _rng
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss
                
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
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                    )
                else:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )
                
                update_state = (train_state, traj_batch, action_adv, comm_adv, targets, rng)
                return update_state, total_loss
            
            if config.get("PARAMETER_SHARING", True):
                update_state = (train_state, traj_batch, action_advantages, comm_advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
            else:
                update_state_dict = []
                metric = []
                for i in range(env.num_agents):
                    update_state = (train_state[i], traj_batch[i], action_advantages[i], comm_advantages[i], targets[i], rng)
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]
                    metric_i = traj_batch[i].info
                    metric_i['loss'] = loss_info[0]
                    metric.append(metric_i)
                    rng = update_state[-1]
            
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
            metric["eat_own_coins"] = metric["eat_own_coins"] * config["ENV_KWARGS"]["num_inner_steps"]
            
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
            network_params = [network[i].init(_rng, init_x) for i in range(env.num_agents)]
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
                for i in range(env.num_agents):
                    update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]
                    metric_i = traj_batch[i].info
                    metric_i['loss'] = loss_info[0]
                    metric.append(metric_i)
                    rng = update_state[-1]
                
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
            metric["eat_own_coins"] = metric["eat_own_coins"] * config["ENV_KWARGS"]["num_inner_steps"]
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
    
    # Build tags
    if use_comm:
        tags = ["AUTOENCODER", "COMM"]
        name_suffix = "autoencoder_comm"
    else:
        tags = ["IPPO", "FF"]
        name_suffix = "ippo_cnn"
    
    # Add parameter sharing to tags and name
    if param_sharing:
        tags.append("PS")
        name_suffix += "_ps"
    else:
        tags.append("IND")
        name_suffix += "_ind"
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
        name=f'{name_suffix}_coins'
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
            save_path = f"./checkpoints/autoencoder/{filename}_ps.pkl"
            save_params(train_state, save_path)
            params = load_params(save_path)
        else:
            # Communication with independent agents
            params = []
            for i in range(config['ENV_KWARGS']['num_agents']):
                save_path = f"./checkpoints/autoencoder/{filename}_ind_{i}.pkl"
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
            embedding_dim=64  # CNN output dimension
        )
    else:
        network = [ActorCriticComm(
            action_dim=env.action_space().n,
            comm_dim=config.get("COMM_DIM", 64),
            num_protos=config.get("NUM_PROTOS", 10),
            hidden_dim=config.get("HIDDEN_DIM", 128),
            activation=config.get("ACTIVATION", "relu"),
            embedding_dim=64  # CNN output dimension
        ) for _ in range(env.num_agents)]
    
    # Initialize hidden states and communication
    hidden_states = jnp.zeros((env.num_agents, config.get("HIDDEN_DIM", 128)))
    prev_comm = jnp.zeros((env.num_agents, config.get("COMM_DIM", 64)))
    
    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/coins_comm"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # Stack observations
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        
        # Forward pass
        rng, _rng = jax.random.split(rng)
        
        if config.get("PARAMETER_SHARING", True):
            action_logits, comm_vectors, _, _, new_hidden_states, _, _ = network.apply(
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
                action_logits_i, comm_vectors_i, _, _, new_hidden_i, _, _ = network[i].apply(
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
    root_dir = f"evaluation/coins"
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
    Hyperparameter sweep with wandb for AutoEncoder experiments.
    
    Simple sweep over seeds for AutoEncoder baseline.
    Can optionally sweep over AUTOENCODER_LOSS_COEF if needed.
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "autoencoder_coins_sweep",
        "method": "grid",  # Try all combinations
        "program": "autoencoder_cnn_coins.py",  # The script to run
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            # Multiple seeds for reproducibility
            "SEED": {"values": [42, 68, 123, 456]},  # 4 different random seeds
            
            # Optional: sweep over autoencoder loss coefficient
            # "AUTOENCODER_LOSS_COEF": {"values": [0.01, 0.1, 1.0]},
            
            # Fixed parameters
            "PARAMETER_SHARING": {"values": [False]},  # Individual policies (matching LG-TOM)
            "USE_COMM": {"values": [True]},  # Always use communication
            "ENV_KWARGS.shared_rewards": {"values": [False]},  # Individual rewards
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
        
        # Ensure fixed settings for AutoEncoder
        config["USE_COMM"] = True
        config["PARAMETER_SHARING"] = config.get("PARAMETER_SHARING", False)
        
        # Build descriptive run name
        param_str = "ps" if config.get("PARAMETER_SHARING", False) else "ind"
        ae_coef = config.get("AUTOENCODER_LOSS_COEF", 0.1)
        run_name = f"autoencoder_{param_str}_ae{ae_coef}_s{config['SEED']}"
        wandb.run.name = run_name
        
        # Update tags
        tags = ["AUTOENCODER", "COMM"]
        if config.get("PARAMETER_SHARING", False):
            tags.append("PS")
        else:
            tags.append("IND")
        tags.append(f"AE_COEF_{ae_coef}")
        
        wandb.run.tags = tags
        
        print("="*70)
        print(f"Running experiment: {run_name}")
        print(f"  PARAMETER_SHARING: {config.get('PARAMETER_SHARING', False)}")
        print(f"  AUTOENCODER_LOSS_COEF: {ae_coef}")
        print(f"  SEED: {config['SEED']}")
        print(f"  Total Timesteps: {config['TOTAL_TIMESTEPS']:.0e}")
        print(f"  Tags: {tags}")
        print("="*70)

        # Run training
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        
        # Always use communication training function for AutoEncoder
        train_fn = make_train_comm(config)
        
        train_vjit = jax.jit(jax.vmap(train_fn))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0])
        
        print(f"Training completed for {run_name}")

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    
    print("\n" + "="*70)
    print("Starting WandB Sweep: AutoEncoder Baseline")
    print(f"Sweep ID: {sweep_id}")
    print(f"Total Combinations:")
    print(f"  - SEED: 4 values")
    print(f"Timesteps per run: {default_config['TOTAL_TIMESTEPS']:.0e}")
    print(f"\nFixed Settings:")
    print(f"  - PARAMETER_SHARING: False (individual policies, matching LG-TOM)")
    print(f"  - USE_COMM: True")
    print(f"  - Individual rewards (not shared)")
    print(f"  - AUTOENCODER_LOSS_COEF: {default_config.get('AUTOENCODER_LOSS_COEF', 0.1)}")
    print("="*70 + "\n")
    
    # Run sweep agent
    num_seeds = len(sweep_config["parameters"]["SEED"]["values"])
    wandb.agent(sweep_id, wrapped_make_train, count=num_seeds)


@hydra.main(version_base=None, config_path="config", config_name="autoencoder_cnn_coins")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)
if __name__ == "__main__":
    main()
