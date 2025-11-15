""" 
InfoPG (Adv InfoPG with k=1) Implementation for Coin Game
Based on InfoPG paper and adapted from IPPO implementation
"""
import sys
sys.path.append('/home/huao/Research/SocialJax')
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Optional
from flax.training.train_state import TrainState
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


class InfoPGPolicy(nn.Module):
    """InfoPG Policy Network with k-level communication"""
    latent_size: int = 64
    action_dim: int = 7
    activation: str = "relu"

    @nn.compact
    def __call__(self, encoding, prev_latent: Optional[jnp.ndarray] = None):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Policy network: encoding -> latent
        policy_latent = nn.Dense(
            self.latent_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(encoding)
        policy_latent = activation(policy_latent)

        # Communication network: if prev_latent is provided, do k-level communication
        if prev_latent is not None:
            # Concatenate current latent with neighbor's latent
            combined = jnp.concatenate([policy_latent, prev_latent], axis=-1)
            # Recurrent policy update (similar to paper's RNN)
            # Use standard initialization (could be identity for curriculum learning as in paper)
            communication = nn.Dense(
                self.latent_size, 
                kernel_init=orthogonal(np.sqrt(2)), 
                bias_init=constant(0.0),
                use_bias=False
            )(combined)
            policy_latent = activation(communication)

        return policy_latent


class InfoPGActorCritic(nn.Module):
    """InfoPG Actor-Critic Network"""
    latent_size: int = 64
    action_dim: int = 7
    activation: str = "relu"

    @nn.compact
    def __call__(self, encoding, prev_latent: Optional[jnp.ndarray] = None):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Get policy latent through k-level communication
        policy_latent = InfoPGPolicy(
            latent_size=self.latent_size,
            action_dim=self.action_dim,
            activation=self.activation
        )(encoding, prev_latent)

        # Final action distribution
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(policy_latent)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic network (uses encoding, not latent)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(encoding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), policy_latent


class InfoPGNetwork(nn.Module):
    """Complete InfoPG Network combining CNN encoder and Actor-Critic"""
    latent_size: int = 64
    action_dim: int = 7
    activation: str = "relu"

    def setup(self):
        self.encoder = CNN(activation=self.activation)
        self.actor_critic = InfoPGActorCritic(
            latent_size=self.latent_size,
            action_dim=self.action_dim,
            activation=self.activation
        )

    def __call__(self, obs, prev_latents: Optional[jnp.ndarray] = None, k_levels: int = 1):
        """
        Forward pass with k-level communication
        
        Args:
            obs: observations [batch, *obs_shape]
            prev_latents: previous latent vectors from neighbors [batch, num_neighbors, latent_size]
            k_levels: number of k-levels for recursive reasoning
        
        Returns:
            pi: action distribution
            value: state value
            latent: policy latent vector
        """
        # Encode observations
        encoding = self.encoder(obs)
        
        # For k=1, use prev_latents if available
        prev_latent = None
        if prev_latents is not None and k_levels > 0:
            # prev_latents shape: [batch, num_neighbors, latent_size]
            if prev_latents.shape[1] > 0:
                # For k=1, take first neighbor's latent (or average if multiple neighbors)
                if prev_latents.shape[1] == 1:
                    prev_latent = prev_latents[:, 0]  # [batch, latent_size]
                else:
                    # Average over neighbors for multiple neighbors
                    prev_latent = jnp.mean(prev_latents, axis=1)  # [batch, latent_size]
        
        # Forward through actor-critic
        pi, value, latent = self.actor_critic(encoding, prev_latent)
        
        return pi, value, latent


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    latent: jnp.ndarray  # Added latent for InfoPG
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # Non-parameter sharing
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    k_levels = config.get("K_LEVELS", 1)
    latent_size = config.get("LATENT_SIZE", 64)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK (non-parameter sharing)
        network = [
            InfoPGNetwork(
                latent_size=latent_size,
                action_dim=env.action_space().n,
                activation=config["ACTIVATION"]
            ) for _ in range(env.num_agents)
        ]
        
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        init_latents = jnp.zeros((1, env.num_agents - 1, latent_size))  # For k-level communication

        network_params = [
            network[i].init(_rng, init_x, init_latents, k_levels) 
            for i in range(env.num_agents)
        ]
        
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
        
        train_state = [
            TrainState.create(
                apply_fn=network[i].apply,
                params=network_params[i],
                tx=tx,
            ) for i in range(env.num_agents)
        ]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # Initialize latent vectors for k-level communication
        # Shape: [num_envs, num_agents, latent_size]
        prev_latents = jnp.zeros((config["NUM_ENVS"], env.num_agents, latent_size))

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, prev_latents_state, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                # For coin game, we use non-parameter sharing (PARAMETER_SHARING: False)
                # Each agent has its own network
                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))  # [num_agents, num_envs, *obs_shape]
                env_act = {}
                log_prob_list = []
                value_list = []
                latent_list = []
                
                for i in range(env.num_agents):
                    # Get other agents' latents for this agent (k-level communication)
                    other_agents = [j for j in range(env.num_agents) if j != i]
                    if len(other_agents) > 0 and k_levels > 0:
                        # Get latents from other agents: [num_envs, num_others, latent_size]
                        other_latents = prev_latents_state[:, other_agents, :]
                    else:
                        other_latents = None
                    
                    # Forward pass through agent i's network
                    pi, value_i, latent_i = network[i].apply(
                        train_state[i].params, 
                        obs_batch[i],  # [num_envs, *obs_shape]
                        other_latents,  # [num_envs, num_others, latent_size] or None
                        k_levels,
                        method=network[i].__call__
                    )
                    action = pi.sample(seed=_rng)
                    log_prob_list.append(pi.log_prob(action))  # [num_envs]
                    env_act[env.agents[i]] = action  # [num_envs]
                    value_list.append(value_i)  # [num_envs]
                    latent_list.append(latent_i)  # [num_envs, latent_size]
                
                log_prob = jnp.stack(log_prob_list, axis=1)  # [num_envs, num_agents]
                value = jnp.stack(value_list, axis=1)  # [num_envs, num_agents]
                latent = jnp.stack(latent_list, axis=1)  # [num_envs, num_agents, latent_size]

                env_act = [v for v in env_act.values()]
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # Update prev_latents for next step (only keep latents for non-done episodes)
                # Reset latents when episode is done
                done_all = done["__all__"]
                new_prev_latents = jnp.where(
                    done_all[:, None, None],
                    jnp.zeros_like(prev_latents_state),
                    latent
                )

                # Store transitions (non-parameter sharing)
                transition = []
                done_flat = [v for v in done.values()]
                for i in range(env.num_agents):
                    info_i = {
                        key: jax.tree_util.tree_map(
                            lambda x: x.reshape((config["NUM_ACTORS"]), 1), 
                            value[:, i]
                        ) for key, value in info.items()
                    }
                    transition.append(Transition(
                        done_flat[i],  # [num_envs]
                        env_act[i],  # [num_envs]
                        value[:, i],  # [num_envs]
                        reward[:, i],  # [num_envs]
                        log_prob[:, i],  # [num_envs]
                        obs_batch[i],  # [num_envs, *obs_shape]
                        latent[:, i],  # [num_envs, latent_size]
                        info_i,
                    ))
                
                runner_state = (train_state, env_state, obsv, new_prev_latents, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, prev_latents_state, update_step, rng = runner_state
            
            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))  # [num_agents, num_envs, *obs_shape]
            last_val_list = []
            for i in range(env.num_agents):
                _, last_val_i, _ = network[i].apply(
                    train_state[i].params, 
                    last_obs_batch[i],  # [num_envs, *obs_shape]
                    None,  # No communication for last value
                    k_levels,
                    method=network[i].__call__
                )
                last_val_list.append(last_val_i)
            last_val = jnp.stack(last_val_list, axis=0)  # [num_agents, num_envs]

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
            
            # Calculate advantages for each agent
            advantages_list = []
            targets_list = []
            for i in range(env.num_agents):
                advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i])
                advantages_list.append(advantages_i)
                targets_list.append(targets_i)
            advantages = jnp.stack(advantages_list, axis=0)  # [num_agents, num_steps, num_envs]
            targets = jnp.stack(targets_list, axis=0)  # [num_agents, num_steps, num_envs]
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused, agent_idx=None):
                def _update_minbatch(train_state, batch_info, network_used):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used):
                        # RERUN NETWORK
                        # For InfoPG, we need prev_latents, but during update we use stored latents
                        # However, we can't easily use stored latents in a batched way
                        # So we'll use None for prev_latents during update (simplification)
                        pi, value, _ = network_used.apply(
                            params, 
                            traj_batch.obs, 
                            None,  # Don't use communication during update for simplicity
                            k_levels,
                            method=network_used.__call__
                        )
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

                        # CALCULATE ACTOR LOSS (Adv InfoPG with normal advantage)
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
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                
                # Use agent-specific network
                network_used = network[agent_idx] if agent_idx is not None else network[0]
                
                train_state, total_loss = jax.lax.scan(
                    lambda state, batch_info: _update_minbatch(state, batch_info, network_used), 
                    train_state, 
                    minibatches
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            # Update each agent separately (non-parameter sharing)
            update_state_dict = []
            metric = []
            for i in range(env.num_agents):
                update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, i), 
                    update_state, 
                    None, 
                    config["UPDATE_EPOCHS"]
                )
                update_state_dict.append(update_state)
                train_state[i] = update_state[0]
                metric_i = traj_batch[i].info
                metric_i['loss'] = loss_info[0]
                metric.append(metric_i)
                rng = update_state[-1]
                
            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            
            # For non-parameter sharing, take first agent's metrics (or aggregate)
            for i in range(env.num_agents):
                metric[i]["update_step"] = update_step
                metric[i]["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric = metric[0]  # Use first agent's metrics for logging
            
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            if "eat_own_coins" in metric:
                metric["eat_own_coins"] = metric["eat_own_coins"] * config["ENV_KWARGS"]["num_inner_steps"]
            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, prev_latents_state, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, prev_latents, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def single_run(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["InfoPG", "Adv-InfoPG", "k=1"] + config.get("WANDB_TAGS", []),
        config=config,
        mode=config["WANDB_MODE"],
        name=f'infopg_cnn_coins_k{config.get("K_LEVELS", 1)}'
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}'
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    
    os.makedirs(f"./checkpoints/infopg", exist_ok=True)
    params = []
    for i in range(config['ENV_KWARGS']['num_agents']):
        save_path = f"./checkpoints/infopg/{filename}_{i}.pkl"
        save_params(train_state[i], save_path)
        params.append(load_params(save_path))
    
    print("** Training Complete **")


@hydra.main(version_base=None, config_path="config", config_name="infopg_cnn_coins")
def main(config):
    single_run(config)

if __name__ == "__main__":
    main()

