# Trajectory Logging

The LLM simulation automatically logs all data needed for training RL/IL agents.

## Output Files

When running `coins_llm_simulation.py`, three JSON files are created:

1. **`trajectory_raw.json`** - Full LLM inputs/outputs and complete environment states
2. **`trajectory_parsed.json`** - Structured data for RL/IL training (observations, actions, rewards)
3. **`trajectory_stats.json`** - Summary statistics including average returns

## Usage

Run simulation:
```bash
python llms/coins_llm_simulation.py --steps 50 --seed 42
```

The script will output performance summary:
```
======================================================================
PERFORMANCE SUMMARY
======================================================================
Episode Length: 50

Agent 0:
  Total Return: 15.00
  Average Return: 0.3000

Agent 1:
  Total Return: 12.00
  Average Return: 0.2400
======================================================================
```

## Data Format

### trajectory_parsed.json
```json
{
  "metadata": {"model": "gpt-5", "seed": 42, ...},
  "trajectory": [
    {
      "timestep": 0,
      "agents": [
        {
          "agent_id": 0,
          "observation": "...",
          "belief": "...",
          "action": "up",
          "action_idx": 4,
          "communication": "...",
          "reward": 0.0
        }
      ],
      "env_obs": [...],  // (num_agents, 11, 11, channels)
      "env_state_compact": {...}
    }
  ]
}
```

### Loading for Training

```python
import json
import numpy as np

# Load parsed trajectory
with open('trajectory_parsed.json') as f:
    data = json.load(f)

# Extract arrays for training
observations = []
actions = []
rewards = []

for step in data['trajectory']:
    for agent in step['agents']:
        observations.append(step['env_obs'][agent['agent_id']])
        actions.append(agent['action_idx'])
        rewards.append(agent['reward'])

observations = np.array(observations)  # (N, 11, 11, C)
actions = np.array(actions)            # (N,)
rewards = np.array(rewards)            # (N,)
```

## Action Mapping

- 0: turn_left
- 1: turn_right  
- 2: left (west)
- 3: right (east)
- 4: up (north)
- 5: down (south)
- 6: stay

