"""
Unified WandB sweep script to compare LGTOM and LangGround on Coop Mining environment.

Usage:
    python compare_algorithms_sweep.py

Running this file launches a WandB sweep comparing:
   - LGTOM (LG-TOM with LLM supervision on beliefs)
   - LangGround (LLM supervision on communication, no ToM)
   - Seeds: [1, 2, 5, 42, 52, 62, 110, 222]
   - Uses LLM dataset: llms/llms/llm_datasets/llm_dataset_coop_mining_semantic_key.pkl
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
# Configure JAX for reproducibility
jax.config.update("jax_default_prng_impl", "rbg")  # Use ThreeFry PRNG for reproducibility
import wandb
import copy
from omegaconf import OmegaConf
from pathlib import Path

# Import configs
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import a module from a file path, handling special characters in directory names"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import training functions from LGTOM algorithm
# Handle hyphenated directory name "LG-TOM"
base_path = Path(__file__).parent
lgtom_module = import_module_from_path(
    "lgtom_cnn_coop_mining",
    base_path / "algorithms" / "LG-TOM" / "lgtom_cnn_coop_mining.py"
)
infopg_module = import_module_from_path(
    "infopg_cnn_coop_mining",
    base_path / "algorithms" / "InfoPG" / "infopg_cnn_coop_mining.py"
)
autoencoder_module = import_module_from_path(
    "autoencoder_cnn_coop_mining",
    base_path / "algorithms" / "AutoEncoder" / "autoencoder_cnn_coop_mining.py"
)

make_train_lgtom = lgtom_module.make_train_comm
make_train_infopg = infopg_module.make_train
make_train_autoencoder = autoencoder_module.make_train_comm

def load_config(algorithm_name):
    """Load default config for each algorithm by reading YAML directly"""
    config_name_map = {
        "lgtom": "lgtom_cnn_coop_mining",
        "infopg": "infopg_cnn_coop_mining",
        "autoencoder": "autoencoder_cnn_coop_mining"
    }
    
    # Handle hyphenated directory name for LG-TOM
    algorithm_dir_map = {
        "lgtom": "LG-TOM",
        "infopg": "InfoPG",
        "autoencoder": "AutoEncoder"
    }
    
    # Load YAML file directly using OmegaConf (handles quoted keys)
    base_path = Path(__file__).parent
    config_file = base_path / "algorithms" / algorithm_dir_map[algorithm_name] / "config" / f"{config_name_map[algorithm_name]}.yaml"
    
    cfg = OmegaConf.load(config_file)
    return OmegaConf.to_container(cfg, resolve=True)

def create_base_config():
    """Create base configuration with common settings"""
    return {
        "SEED": 1,  # Default seed, will be overridden by sweep parameter
        "TOTAL_TIMESTEPS": 2e7,
        "REWARD": "individual",
        "PARAMETER_SHARING": False,
        "USE_SEPARATE_REWARDS": False,  # Joint rewards
        "ENV_KWARGS": {
            "shared_rewards": False,
            "num_agents": 6,
            "num_inner_steps": 996,
            "num_outer_steps": 1,
            "max_miners": 4,
            "min_gold_miners": 2,
            "mining_range": 3,
            "reward_iron": 1.0,
            "reward_gold": 8.0,
            "gold_mining_window": 3,
            "regrowth_prob_iron": 0.0004,
            "regrowth_prob_gold": 0.00016,
            "cnn": True,
            "jit": True,
        },
        # Common training hyperparameters
        "LR": 0.0005,
        "NUM_ENVS": 256,
        "NUM_STEPS": 996,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 256,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "coop_mining",
        "REW_SHAPING_HORIZON": 2.5e6,
        "SHAPING_BEGIN": 1e6,
        "ANNEAL_LR": True,
        "NUM_SEEDS": 1,
        "GIF_NUM_FRAMES": 250,
        'SUPERVISED_LOSS_TYPE': "mse",
        # LLM dataset path for supervision
        "LLM_DATA_PATH": "llms/llms/llm_datasets/llm_dataset_coop_mining_semantic_key.pkl",
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
    config["USE_SEPARATE_REWARDS"] = False  # Joint rewards
    config["COMM_LOSS_COEF"] = 1
    
    if variant == "social_influence":
        # Condition 1: use_ToM, ground_truth_supervision, use intrinsic
        config["USE_TOM"] = True
        config["USE_INTRINSIC_REWARD"] = True
        config["SOCIAL_INFLUENCE_COEFF"] = 0.1
        config["SUPERVISED_BELIEF"] = "ground_truth"
        config["SUPERVISED_COMM"] = "none"
        config["SUPERVISED_LOSS_COEF"] = 1.0
    elif variant == "lgtom":
        # Condition 2: use_ToM, llm_supervision, use intrinsic
        config["USE_TOM"] = True
        config["USE_INTRINSIC_REWARD"] = True
        config["SOCIAL_INFLUENCE_COEFF"] = 0.1
        config["SUPERVISED_BELIEF"] = "llm"
        config["SUPERVISED_COMM"] = "none"
        config["SUPERVISED_LOSS_COEF"] = 1.0
    elif variant == "langground":
        # Condition 3: no_ToM, llm_supervision on communication, no intrinsic
        config["USE_TOM"] = False
        config["USE_INTRINSIC_REWARD"] = False
        config["SOCIAL_INFLUENCE_COEFF"] = 0.0
        config["SUPERVISED_BELIEF"] = "none"
        config["SUPERVISED_COMM"] = "llm"
        config["SUPERVISED_LOSS_COEF"] = 1.0
    elif variant == "proto":
        # Condition 4: no_ToM, no intrinsic
        config["USE_TOM"] = False
        config["USE_INTRINSIC_REWARD"] = False
        config["SOCIAL_INFLUENCE_COEFF"] = 0.0
        config["SUPERVISED_BELIEF"] = "none"
        config["SUPERVISED_COMM"] = "none"
        config["SUPERVISED_LOSS_COEF"] = 0.0
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
    config["USE_SEPARATE_REWARDS"] = False  # Joint rewards
    config["COMM_LOSS_COEF"] = 1
    return config

def run_training(config, algorithm_name, variant=None):
    """Run training for LGTOM algorithm configuration"""
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
    seed = wandb.config.SEED
    
    # Define all experiments explicitly
    experiments = {
        # 0: {"algorithm": "lgtom", "variant": "social_influence", "name": "Social Influence"},
        # 1: {"algorithm": "lgtom", "variant": "proto", "name": "Proto"},
        # 2: {"algorithm": "infopg", "variant": None, "name": "InfoPG"},
        # 3: {"algorithm": "autoencoder", "variant": None, "name": "AutoEncoder"},
        5: {"algorithm": "lgtom", "variant": "lgtom", "name": "LGTOM"},
        6: {"algorithm": "lgtom", "variant": "langground", "name": "LangGround"},
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
    
    # Override seed with sweep parameter
    unified_config["SEED"] = seed
    config_get = lambda key, default: getattr(wandb.config, key, default)
    shared_rewards = config_get("shared_rewards", unified_config["ENV_KWARGS"].get("shared_rewards", False))
    parameter_sharing = config_get("PARAMETER_SHARING", unified_config.get("PARAMETER_SHARING", False))
    use_separate_rewards = config_get("USE_SEPARATE_REWARDS", unified_config.get("USE_SEPARATE_REWARDS", False))
    
    # Merge with algorithm-specific defaults
    for key, value in base_config.items():
        if key not in unified_config or key in ["ENV_KWARGS"]:
            if key == "ENV_KWARGS":
                unified_config[key].update(value)
            else:
                unified_config[key] = value
    
    unified_config["ENV_KWARGS"]["shared_rewards"] = shared_rewards
    unified_config["PARAMETER_SHARING"] = parameter_sharing
    unified_config["USE_SEPARATE_REWARDS"] = use_separate_rewards
    
    # Apply algorithm-specific configuration
    experiment_name = experiment.get("name", variant or algorithm)
    if algorithm == "lgtom":
        unified_config = configure_lgtom(unified_config, variant)
        run_name = f"{variant}_s{unified_config['SEED']}"
        tags = ["LGTOM", "COMM", "IND", "INDIVIDUAL_REWARD", "JOINT_REWARD", "BELIEF"]
        if variant == "social_influence":
            tags.extend(["TOM", "INTRINSIC", "GROUND_TRUTH", f"INTR_COEF_{unified_config['SOCIAL_INFLUENCE_COEFF']}"])
        elif variant == "lgtom":
            tags.extend(["TOM", "INTRINSIC", "LLM_SUPERVISION", f"INTR_COEF_{unified_config['SOCIAL_INFLUENCE_COEFF']}"])
        elif variant == "langground":
            tags.extend(["NO_TOM", "NO_INTRINSIC", "LLM_COMM"])
        elif variant == "proto":
            tags.extend(["NO_TOM", "NO_INTRINSIC"])
    elif algorithm == "infopg":
        unified_config = configure_infopg(unified_config)
        run_name = f"infopg_k{unified_config['K_LEVELS']}_s{unified_config['SEED']}"
        tags = ["INFOPG", "IND", "INDIVIDUAL_REWARD", f"k={unified_config['K_LEVELS']}"]
    elif algorithm == "autoencoder":
        unified_config = configure_autoencoder(unified_config)
        run_name = f"autoencoder_ae{unified_config['AUTOENCODER_LOSS_COEF']}_s{unified_config['SEED']}"
        tags = ["AUTOENCODER", "COMM", "IND", "INDIVIDUAL_REWARD", "JOINT_REWARD", f"AE_COEF_{unified_config['AUTOENCODER_LOSS_COEF']}"]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Set wandb run name and tags
    wandb.run.name = run_name
    wandb.run.tags = tags
    
    # Log configuration
    print("="*70)
    print(f"Running experiment: {experiment_name} ({run_name})")
    print(f"  Algorithm: {algorithm}")
    if variant:
        print(f"  Variant: {variant}")
    print(f"  SEED: {unified_config['SEED']}")
    print(f"  TOTAL_TIMESTEPS: {unified_config['TOTAL_TIMESTEPS']:.0e}")
    print(f"  PARAMETER_SHARING: {unified_config['PARAMETER_SHARING']}")
    print(f"  USE_SEPARATE_REWARDS: {unified_config.get('USE_SEPARATE_REWARDS', False)}")
    print(f"  Shared Rewards: {unified_config['ENV_KWARGS']['shared_rewards']}")
    if algorithm == "lgtom":
        print(f"  USE_TOM: {unified_config.get('USE_TOM', False)}")
        print(f"  USE_INTRINSIC_REWARD: {unified_config.get('USE_INTRINSIC_REWARD', False)}")
        print(f"  INFLUENCE_TARGET: {unified_config.get('INFLUENCE_TARGET', 'belief')}")
        print(f"  COMM_LOSS_COEF: {unified_config.get('COMM_LOSS_COEF', 0.0)}")
        if unified_config.get('USE_INTRINSIC_REWARD', False):
            print(f"  SOCIAL_INFLUENCE_COEFF: {unified_config.get('SOCIAL_INFLUENCE_COEFF', 0.0)}")
        if unified_config.get('USE_TOM', False):
            print(f"  SUPERVISED_BELIEF: {unified_config.get('SUPERVISED_BELIEF', 'none')}")
            print(f"  SUPERVISED_LOSS_COEF: {unified_config.get('SUPERVISED_LOSS_COEF', 0.0)}")
        if unified_config.get('SUPERVISED_COMM', 'none') != 'none':
            print(f"  SUPERVISED_COMM: {unified_config.get('SUPERVISED_COMM', 'none')}")
    elif algorithm == "autoencoder":
        print(f"  AUTOENCODER_LOSS_COEF: {unified_config.get('AUTOENCODER_LOSS_COEF', 0.0)}")
        print(f"  COMM_LOSS_COEF: {unified_config.get('COMM_LOSS_COEF', 0.0)}")
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
    """Set up and run the two required wandb sweeps."""

    def make_sweep_config(name, experiment_ids, seeds, extra_parameters=None):
        parameters = {
            "experiment_id": {"values": experiment_ids},
            "SEED": {"values": seeds},
        }
        if extra_parameters:
            for key, value in extra_parameters.items():
                parameters[key] = {"value": value}
        return {
            "name": name,
            "method": "grid",
            "metric": {
                "name": "returned_episode_returns",
                "goal": "maximize",
            },
            "parameters": parameters,
        }

    sweeps = [
        {
            "title": "Coop Mining Comparison: Social Influence, Proto, InfoPG, AutoEncoder",
            "experiment_ids": [5,6],
            "seeds": [110],
            "extra_params": {
                "shared_rewards": False,
                "PARAMETER_SHARING": False,
                "USE_SEPARATE_REWARDS": False,
            },
            "description": (
                "Comparing LGTOM vs LangGround on coop_mining environment using LLM dataset. "
                "LGTOM: LLM supervision on beliefs with ToM. LangGround: LLM supervision on communication without ToM. "
                "Dataset: llms/llms/llm_datasets/llm_dataset_coop_mining_semantic_key.pkl"
            ),
        },
    ]

    wandb.login()

    for sweep in sweeps:
        sweep_config = make_sweep_config(
            name=sweep["title"],
            experiment_ids=sweep["experiment_ids"],
            seeds=sweep["seeds"],
            extra_parameters=sweep["extra_params"],
        )

        sweep_id = wandb.sweep(
            sweep_config,
            entity="",
            project="socialjax",
        )

        total_runs = len(sweep["experiment_ids"]) * len(sweep["seeds"])
        print("\n" + "=" * 70)
        print(f"Starting WandB Sweep: {sweep['title']}")
        print(f"Description: {sweep['description']}")
        print(f"Sweep ID: {sweep_id}")
        print(f"Experiments: {sweep['experiment_ids']}")
        print(f"Seeds: {sweep['seeds']}")
        if sweep["extra_params"]:
            print("Extra Parameters:")
            for key, value in sweep["extra_params"].items():
                print(f"  - {key}: {value}")
        else:
            print("Extra Parameters: (defaults)")
        print(f"Total Runs: {total_runs}")
        print("=" * 70 + "\n")

        wandb.agent(sweep_id, wrapped_make_train, count=total_runs)

if __name__ == "__main__":
    main()

