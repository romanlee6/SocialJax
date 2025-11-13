# Running 3 LLM Simulation Runs with 100 Steps Each

This document describes how to collect 3 runs of the LLM coins game simulation with 100 steps per run, including additional metrics saved as numpy arrays for visualization.

## Quick Start

```bash
# Navigate to the llms directory
cd /home/huao/Research/SocialJax/llms

# Run with default settings (o3 model, seeds 42/123/456, 100 steps)
./run_3runs_100steps.sh
```

## Script Details

The `run_3runs_100steps.sh` script will:
- Run 3 simulations with different random seeds (default: 42, 123, 456)
- Execute 100 timesteps per simulation (default)
- Use the "o3" model (default)
- Save all standard outputs plus additional metrics

## Additional Metrics Saved

For each run, the following numpy arrays are saved in `.npy` format:

### 1. `accumulated_rewards.npy`
- **Shape**: `(timesteps, num_agents)`
- **Content**: Cumulative rewards for each agent at each timestep
- **Usage**: Plot reward accumulation over time
```python
import numpy as np
rewards = np.load("accumulated_rewards.npy")
# rewards[t, agent_id] = cumulative reward at timestep t for agent
```

### 2. `coins_in_state.npy`
- **Shape**: `(timesteps, 2)`
- **Content**: Number of red and green coins in the environment at each timestep
- **Usage**: Track coin availability over time
```python
coins = np.load("coins_in_state.npy")
# coins[t, 0] = red coins at timestep t
# coins[t, 1] = green coins at timestep t
```

### 3. `own_color_coins_eaten.npy`
- **Shape**: `(timesteps, num_agents)`
- **Content**: Cumulative count of own-color coins eaten by each agent
- **Usage**: Track cooperative behavior (eating own color coins)
```python
own_eaten = np.load("own_color_coins_eaten.npy")
# own_eaten[t, agent_id] = cumulative own-color coins eaten by agent at timestep t
```

### 4. `other_color_coins_eaten.npy`
- **Shape**: `(timesteps, num_agents)`
- **Content**: Cumulative count of other-agent's color coins eaten by each agent
- **Usage**: Track defection behavior (eating opponent's coins)
```python
other_eaten = np.load("other_color_coins_eaten.npy")
# other_eaten[t, agent_id] = cumulative other-color coins eaten by agent at timestep t
```

## Command-Line Options

```bash
./run_3runs_100steps.sh [OPTIONS]

Options:
  --model MODEL          Model name (default: o3)
  --seeds SEEDS          Space-separated list of 3 seeds (default: 42 123 456)
  --steps STEPS          Number of steps per run (default: 100)
  --temperature TEMP     Sampling temperature (default: 0.7)
  --output-dir DIR       Output directory (default: ./llm_simulation_output)
  --help, -h             Show this help
```

## Examples

### Use Different Seeds
```bash
./run_3runs_100steps.sh --seeds 10 20 30
```

### Use Different Model
```bash
./run_3runs_100steps.sh --model gpt-4
```

### Custom Steps and Temperature
```bash
./run_3runs_100steps.sh --steps 50 --temperature 0.5
```

### Combine Multiple Options
```bash
./run_3runs_100steps.sh --model o3 --seeds 100 200 300 --steps 150
```

## Output Structure

After running, you'll get:

```
llm_simulation_output/
├── collection_o3_YYYY-MM-DD_HH-MM-SS/
│   ├── comparative_summary.txt       # Summary across all runs
│   ├── comparative_summary.json      # Machine-readable summary
│   └── runs_manifest.json            # List of run directories
├── o3_temp0.7_seed42_YYYY-MM-DD_HH-MM-SS/
│   ├── accumulated_rewards.npy       # NEW: Cumulative rewards
│   ├── coins_in_state.npy           # NEW: Coin counts over time
│   ├── own_color_coins_eaten.npy    # NEW: Own-color coins eaten
│   ├── other_color_coins_eaten.npy  # NEW: Other-color coins eaten
│   ├── trajectory_raw.json
│   ├── trajectory_parsed.json
│   ├── human_summary.txt
│   ├── timestep_XXXX.png (100 files)
│   └── simulation.gif
├── o3_temp0.7_seed123_YYYY-MM-DD_HH-MM-SS/
│   └── ... (same structure)
└── o3_temp0.7_seed456_YYYY-MM-DD_HH-MM-SS/
    └── ... (same structure)
```

## Visualization Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Load data from a run
run_dir = "llm_simulation_output/o3_temp0.7_seed42_..."

accumulated_rewards = np.load(f"{run_dir}/accumulated_rewards.npy")
coins_in_state = np.load(f"{run_dir}/coins_in_state.npy")
own_eaten = np.load(f"{run_dir}/own_color_coins_eaten.npy")
other_eaten = np.load(f"{run_dir}/other_color_coins_eaten.npy")

# Plot accumulated rewards
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accumulated rewards
axes[0, 0].plot(accumulated_rewards[:, 0], label='Agent 0 (Red)')
axes[0, 0].plot(accumulated_rewards[:, 1], label='Agent 1 (Green)')
axes[0, 0].set_xlabel('Timestep')
axes[0, 0].set_ylabel('Accumulated Reward')
axes[0, 0].set_title('Accumulated Rewards Over Time')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Coins in state
axes[0, 1].plot(coins_in_state[:, 0], label='Red Coins', color='red')
axes[0, 1].plot(coins_in_state[:, 1], label='Green Coins', color='green')
axes[0, 1].set_xlabel('Timestep')
axes[0, 1].set_ylabel('Number of Coins')
axes[0, 1].set_title('Coins in Environment')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Own color coins eaten
axes[1, 0].plot(own_eaten[:, 0], label='Agent 0 (Red)')
axes[1, 0].plot(own_eaten[:, 1], label='Agent 1 (Green)')
axes[1, 0].set_xlabel('Timestep')
axes[1, 0].set_ylabel('Own Color Coins Eaten')
axes[1, 0].set_title('Own Color Coins Eaten (Cooperation)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Other color coins eaten
axes[1, 1].plot(other_eaten[:, 0], label='Agent 0 (Red)')
axes[1, 1].plot(other_eaten[:, 1], label='Agent 1 (Green)')
axes[1, 1].set_xlabel('Timestep')
axes[1, 1].set_ylabel('Other Color Coins Eaten')
axes[1, 1].set_title('Other Color Coins Eaten (Defection)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('llm_metrics_visualization.png', dpi=150)
plt.show()
```

## Prerequisites

- OpenAI API key must be set:
  ```bash
  export OPENAI_API_KEY='your-api-key-here'
  ```

- For custom endpoints:
  ```bash
  export OPENAI_BASE_URL='https://your-endpoint.com/v1'
  ```

## Notes

- Agent 0 is the **red** agent (collects red coins for +1, green coins for +1 but causes -2 to Agent 1)
- Agent 1 is the **green** agent (collects green coins for +1, red coins for +1 but causes -2 to Agent 0)
- The simulation tracks both cooperative (own-color) and defection (other-color) coin collection
- Each simulation creates a unique timestamped directory with all outputs

