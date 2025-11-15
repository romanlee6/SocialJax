"""
Unified WandB Sweep Script to Compare LG-TOM, InfoPG, and AutoEncoder
on Coin Game with consistent hyperparameters.

Usage:
    python compare_algorithms_sweep.py

This will create a WandB sweep with 5 experiments:
1. LG-TOM: no_ToM + no_Intrinsic
2. LG-TOM: No_ToM + Intrinsic (intrinsic_coef=0.1, COMM_LOSS_COEF=1)
3. LG-TOM: ToM + Intrinsic (intrinsic_coef=0.1, COMM_LOSS_COEF=1, supervised_loss_coeff=1)
4. InfoPG: k=1
5. AutoEncoder: AUTOENCODER_LOSS_COEF=1

Common Settings:
- Individual rewards (shared_rewards=False)
- No parameter sharing (PARAMETER_SHARING=False)
- Seed: 110
- Total timesteps: 1e8
- NUM_ENVS: 512 (unified across all methods)

LG-TOM Variants:
1. no_ToM + no_Intrinsic
2. No_ToM + Intrinsic (intrinsic_coef=0.1, COMM_LOSS_COEF=1)
3. ToM + Intrinsic (intrinsic_coef=0.1, COMM_LOSS_COEF=1, supervised_loss_coeff=1)

AutoEncoder:
- reconstructed_embedding_loss_coef = 1 (AUTOENCODER_LOSS_COEF=1)

InfoPG:
- Standard InfoPG with k=1
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import wandb
import copy
from omegaconf import OmegaConf
import hydra
from pathlib import Path

# Import configs
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydraInstance
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import a module from a file path, handling special characters in directory names"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import training functions from each algorithm using file paths
# Handle hyphenated directory name "LG-TOM"
base_path = Path(__file__).parent
lgtom_module = import_module_from_path(
    "lgtom_cnn_coins",
    base_path / "algorithms" / "LG-TOM" / "lgtom_cnn_coins.py"
)
infopg_module = import_module_from_path(
    "infopg_cnn_coins",
    base_path / "algorithms" / "InfoPG" / "infopg_cnn_coins.py"
)
autoencoder_module = import_module_from_path(
    "autoencoder_cnn_coins",
    base_path / "algorithms" / "AutoEncoder" / "autoencoder_cnn_coins.py"
)

make_train_lgtom = lgtom_module.make_train_comm
make_train_infopg = infopg_module.make_train
make_train_autoencoder = autoencoder_module.make_train_comm

def load_config(algorithm_name):
    """Load default config for each algorithm"""
    # Clear any existing hydra instance
    if GlobalHydraInstance().instance is not None:
        GlobalHydraInstance().instance.clear()
    
    config_name_map = {
        "lgtom": "lgtom_cnn_coins",
        "infopg": "infopg_cnn_coins",
        "autoencoder": "autoencoder_cnn_coins"
    }
    
    # Handle hyphenated directory name for LG-TOM
    algorithm_dir_map = {
        "lgtom": "LG-TOM",
        "infopg": "InfoPG",
        "autoencoder": "AutoEncoder"
    }
    
    config_path = f"algorithms/{algorithm_dir_map[algorithm_name]}/config"
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name_map[algorithm_name])
        return OmegaConf.to_container(cfg, resolve=True)

def create_base_config():
    """Create base configuration with common settings"""
    return {
        "SEED": 110,
        "TOTAL_TIMESTEPS": 1e8,
        "REWARD": "individual",
        "PARAMETER_SHARING": False,
        "ENV_KWARGS": {
            "shared_rewards": False,
            "num_agents": 2,
            "num_inner_steps": 1000,
            "cnn": True,
            "jit": True,
        },
        # Common training hyperparameters
        "LR": 0.0005,
        "NUM_ENVS": 512,
        "NUM_STEPS": 1000,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 500,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "coin_game",
        "REW_SHAPING_HORIZON": 2.5e6,
        "SHAPING_BEGIN": 1e6,
        "ANNEAL_LR": True,
        "NUM_SEEDS": 1,
        "GIF_NUM_FRAMES": 250,
        # WandB settings
        "ENTITY": "",
        "PROJECT": "socialjax",
        "WANDB_MODE": "online",
    }

def configure_lgtom(config, variant):
    """Configure LG-TOM for specific variant"""
    # Common LG-TOM settings
    config["USE_COMM"] = True
    config["COMM_DIM"] = 64
    config["NUM_PROTOS"] = 10
    config["HIDDEN_DIM"] = 128
    config["COMM_MODE"] = "avg"
    config["INFLUENCE_TARGET"] = "belief"
    config["USE_SEPARATE_REWARDS"] = True
    config["COMM_LOSS_COEF"] = 1
    
    if variant == "no_tom_no_intrinsic":
        config["USE_TOM"] = False
        config["USE_INTRINSIC_REWARD"] = False
        config["SOCIAL_INFLUENCE_COEFF"] = 0.0
        config["SUPERVISED_BELIEF"] = "none"
        config["SUPERVISED_LOSS_COEF"] = 0.0
    elif variant == "no_tom_intrinsic":
        config["USE_TOM"] = False
        config["USE_INTRINSIC_REWARD"] = True
        config["SOCIAL_INFLUENCE_COEFF"] = 0.1
        config["SUPERVISED_BELIEF"] = "none"
        config["SUPERVISED_LOSS_COEF"] = 0.0
    elif variant == "tom_intrinsic":
        config["USE_TOM"] = True
        config["USE_INTRINSIC_REWARD"] = True
        config["SOCIAL_INFLUENCE_COEFF"] = 0.1
        config["SUPERVISED_BELIEF"] = "ground_truth"
        config["SUPERVISED_LOSS_COEF"] = 1.0
    else:
        raise ValueError(f"Unknown LG-TOM variant: {variant}")
    
    return config

def configure_infopg(config):
    """Configure InfoPG"""
    config["K_LEVELS"] = 1
    config["LATENT_SIZE"] = 64
    config["COMMUNICATION_RANGE"] = 1.0
    return config

def configure_autoencoder(config):
    """Configure AutoEncoder"""
    config["USE_COMM"] = True
    config["COMM_DIM"] = 64
    config["NUM_PROTOS"] = 10
    config["HIDDEN_DIM"] = 128
    config["COMM_MODE"] = "avg"
    config["AUTOENCODER_LOSS_COEF"] = 1  # reconstructed_embedding_loss_coef
    config["USE_SEPARATE_REWARDS"] = False
    config["COMM_LOSS_COEF"] = 1
    return config

def run_training(config, algorithm_name, variant=None):
    """Run training for a specific algorithm configuration"""
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    
    # Select appropriate training function
    if algorithm_name == "lgtom":
        train_fn = make_train_lgtom(config)
    elif algorithm_name == "infopg":
        train_fn = make_train_infopg(config)
    elif algorithm_name == "autoencoder":
        train_fn = make_train_autoencoder(config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Run training
    train_vjit = jax.jit(jax.vmap(train_fn))
    outs = jax.block_until_ready(train_vjit(rngs))
    train_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0])
    
    return train_state

def wrapped_make_train():
    """Wrapped training function for wandb sweep"""
    wandb.init(project="socialjax")
    
    # Get sweep parameters
    experiment_id = wandb.config.experiment_id
    
    # Define all experiments explicitly
    experiments = {
        0: {"algorithm": "lgtom", "variant": "no_tom_no_intrinsic"},
        1: {"algorithm": "lgtom", "variant": "no_tom_intrinsic"},
        2: {"algorithm": "lgtom", "variant": "tom_intrinsic"},
        3: {"algorithm": "infopg", "variant": None},
        4: {"algorithm": "autoencoder", "variant": None},
    }
    
    if experiment_id not in experiments:
        raise ValueError(f"Unknown experiment_id: {experiment_id}")
    
    experiment = experiments[experiment_id]
    algorithm = experiment["algorithm"]
    variant = experiment["variant"]
    
    # Load base config for the algorithm
    base_config = load_config(algorithm)
    
    # Create unified base config with common settings
    unified_config = create_base_config()
    
    # Merge with algorithm-specific defaults
    for key, value in base_config.items():
        if key not in unified_config or key in ["ENV_KWARGS"]:
            if key == "ENV_KWARGS":
                unified_config[key].update(value)
            else:
                unified_config[key] = value
    
    # Apply algorithm-specific configuration
    if algorithm == "lgtom":
        unified_config = configure_lgtom(unified_config, variant)
        run_name = f"lgtom_{variant}_s{unified_config['SEED']}"
        tags = ["LGTOM", "COMM", "IND", "INDIVIDUAL_REWARD"]
        if variant == "no_tom_no_intrinsic":
            tags.extend(["NO_TOM", "NO_INTRINSIC"])
        elif variant == "no_tom_intrinsic":
            tags.extend(["NO_TOM", "INTRINSIC", f"COEF_{unified_config['SOCIAL_INFLUENCE_COEFF']}"])
        elif variant == "tom_intrinsic":
            tags.extend(["TOM", "INTRINSIC", f"COEF_{unified_config['SOCIAL_INFLUENCE_COEFF']}", "SUPERVISED_BELIEF"])
    elif algorithm == "infopg":
        unified_config = configure_infopg(unified_config)
        run_name = f"infopg_k{unified_config['K_LEVELS']}_s{unified_config['SEED']}"
        tags = ["INFOPG", "IND", "INDIVIDUAL_REWARD", f"k={unified_config['K_LEVELS']}"]
    elif algorithm == "autoencoder":
        unified_config = configure_autoencoder(unified_config)
        run_name = f"autoencoder_ae{unified_config['AUTOENCODER_LOSS_COEF']}_s{unified_config['SEED']}"
        tags = ["AUTOENCODER", "COMM", "IND", "INDIVIDUAL_REWARD", f"AE_COEF_{unified_config['AUTOENCODER_LOSS_COEF']}"]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Set wandb run name and tags
    wandb.run.name = run_name
    wandb.run.tags = tags
    
    # Log configuration
    print("="*70)
    print(f"Running experiment: {run_name}")
    print(f"  Algorithm: {algorithm}")
    if variant:
        print(f"  Variant: {variant}")
    print(f"  SEED: {unified_config['SEED']}")
    print(f"  TOTAL_TIMESTEPS: {unified_config['TOTAL_TIMESTEPS']:.0e}")
    print(f"  PARAMETER_SHARING: {unified_config['PARAMETER_SHARING']}")
    print(f"  Individual Rewards: {not unified_config['ENV_KWARGS']['shared_rewards']}")
    if algorithm == "lgtom":
        print(f"  USE_TOM: {unified_config.get('USE_TOM', False)}")
        print(f"  USE_INTRINSIC_REWARD: {unified_config.get('USE_INTRINSIC_REWARD', False)}")
        if unified_config.get('USE_INTRINSIC_REWARD', False):
            print(f"  SOCIAL_INFLUENCE_COEFF: {unified_config.get('SOCIAL_INFLUENCE_COEFF', 0.0)}")
        if unified_config.get('USE_TOM', False):
            print(f"  SUPERVISED_LOSS_COEF: {unified_config.get('SUPERVISED_LOSS_COEF', 0.0)}")
    elif algorithm == "autoencoder":
        print(f"  AUTOENCODER_LOSS_COEF: {unified_config.get('AUTOENCODER_LOSS_COEF', 0.0)}")
    print(f"  Tags: {tags}")
    print("="*70)
    
    # Run training
    try:
        train_state = run_training(unified_config, algorithm, variant)
        print(f"Training completed for {run_name}")
    except Exception as e:
        print(f"Error during training for {run_name}: {e}")
        raise

def main():
    """Main function to set up and run the wandb sweep"""
    
    # Define sweep configuration with explicit experiment IDs
    sweep_config = {
        "name": "compare_algorithms_coins_sweep",
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "experiment_id": {
                "values": [0, 1, 2, 3, 4]  # 5 experiments total
            }
        },
    }
    
    # Login to wandb
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        entity="",  # Set your entity if needed
        project="socialjax"
    )
    
    print("\n" + "="*70)
    print("Starting WandB Sweep: Algorithm Comparison on Coin Game")
    print(f"Sweep ID: {sweep_id}")
    print(f"\nExperiments:")
    print(f"  0. LG-TOM: no_ToM + no_Intrinsic")
    print(f"  1. LG-TOM: No_ToM + Intrinsic (coef=0.1, COMM_LOSS_COEF=1)")
    print(f"  2. LG-TOM: ToM + Intrinsic (coef=0.1, COMM_LOSS_COEF=1, supervised_loss_coeff=1)")
    print(f"  3. InfoPG: k=1")
    print(f"  4. AutoEncoder: AUTOENCODER_LOSS_COEF=1")
    print(f"\nCommon Settings:")
    print(f"  - Seed: 110")
    print(f"  - Total Timesteps: 1e8")
    print(f"  - Individual Rewards (shared_rewards=False)")
    print(f"  - No Parameter Sharing (PARAMETER_SHARING=False)")
    print("="*70 + "\n")
    
    # Run sweep agent - 5 experiments total
    wandb.agent(sweep_id, wrapped_make_train, count=5)

if __name__ == "__main__":
    main()

