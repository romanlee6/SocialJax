# LG-TOM Tune Function Setup

## Overview
The `tune()` function has been configured to run a WandB sweep testing **6 experimental conditions** combining Theory of Mind (ToM) and Intrinsic Rewards, with experiments ordered by ToM=True first.

## Fixed Parameters (All Runs)
```yaml
PARAMETER_SHARING: False        # Individual policies (non-parameter-sharing)
INFLUENCE_TARGET: "belief"      # Use belief-based influence (cosine similarity)
SEED: 42                        # Fixed seed for reproducibility
REWARD: "individual"            # Individual rewards (not shared)
USE_COMM: True                  # Communication always enabled
```

## Sweep Variables (2×2×2=8 combinations, 6 valid)
```yaml
USE_TOM: [True, False]                # Enable/disable Theory of Mind (ToM first)
USE_INTRINSIC_REWARD: [False, True]   # Enable/disable intrinsic reward
USE_SEPARATE_REWARDS: [False, True]   # Joint vs separate reward structure
```

**Note**: 2 combinations are invalid and filtered out:
- `USE_INTRINSIC_REWARD=False` + `USE_SEPARATE_REWARDS=True` (comm would get 0 reward)

## Conditional Settings

### If USE_INTRINSIC_REWARD = True:
```yaml
SOCIAL_INFLUENCE_COEFF: 0.1
USE_SEPARATE_REWARDS: [False, True]  # Both variants tested
```
- **Separate rewards**: Action learns from task reward, comm learns from intrinsic
- **Joint rewards**: Both learn from task + intrinsic combined

### If USE_INTRINSIC_REWARD = False:
```yaml
SOCIAL_INFLUENCE_COEFF: 0.0
USE_SEPARATE_REWARDS: False      # Forced to False (filtered if True)
```
**Rationale**: Without intrinsic rewards, separating rewards would give the communication policy zero reward. Only joint rewards are valid.

### If USE_TOM = True:
```yaml
SUPERVISED_BELIEF: "ground_truth"
SUPERVISED_LOSS_COEF: 0.1
```

### If USE_TOM = False:
```yaml
SUPERVISED_BELIEF: "none"
SUPERVISED_LOSS_COEF: 0.0
```

## Experiment Matrix (Ordered by ToM First)

| # | USE_TOM | USE_INTRINSIC | USE_SEPARATE | COEFF | SUPERVISED | Description |
|---|---------|--------------|--------------|-------|------------|-------------|
| 1 | **True** | False | False (joint) | 0.0 | ground_truth | **ToM Only** - Supervised belief |
| 2 | **True** | True | True (sep) | 0.1 | ground_truth | **ToM + Intr (sep)** - Supervised + intrinsic separate |
| 3 | **True** | True | False (joint) | 0.1 | ground_truth | **ToM + Intr (joint)** - Supervised + intrinsic joint |
| 4 | False | False | False (joint) | 0.0 | none | **Baseline** - No ToM, No Intrinsic |
| 5 | False | True | True (sep) | 0.1 | none | **Intr Only (sep)** - Intrinsic separate |
| 6 | False | True | False (joint) | 0.1 | none | **Intr Only (joint)** - Intrinsic joint |

## Run Names (Ordered by ToM First)
Experiments will be named with format: `lgtom_{tom}_{intrinsic}_{reward_structure}_{coeff}_s{seed}`

1. `lgtom_tom_nointr_joint_c0.0_s42` - ToM only
2. `lgtom_tom_intr_sep_c0.1_s42` - ToM + Intrinsic (separate)
3. `lgtom_tom_intr_joint_c0.1_s42` - ToM + Intrinsic (joint)
4. `lgtom_notom_nointr_joint_c0.0_s42` - Baseline
5. `lgtom_notom_intr_sep_c0.1_s42` - Intrinsic only (separate)
6. `lgtom_notom_intr_joint_c0.1_s42` - Intrinsic only (joint)

## WandB Tags (Ordered by Priority)
Each run will be tagged with:
- Base tags: `["LGTOM", "COMM", "IND", "BELIEF"]`
- **ToM tags** (first priority): `["TOM", "SUPERVISED_BELIEF"]` or `["NO_TOM"]`
- Reward structure tags: `["SEPARATE_REWARDS"]` or `["JOINT_REWARDS"]`
- Intrinsic tags: `["INTRINSIC", "COEF_0.1"]` or `["NO_INTRINSIC"]`

## How to Run

### 1. Set TUNE=True in config file:
```yaml
"TUNE": True
```

### 2. Run the script:
```bash
python algorithms/LG-TOM/lgtom_cnn_coins.py
```

### 3. The sweep will automatically:
- Create a WandB sweep
- Run 6 valid experiments (2 invalid combinations filtered out)
- Log results to WandB
- Experiments run in order: ToM=True first, then ToM=False

## Expected Training Time
- Each run: ~2e7 timesteps (20M)
- Total: 6 runs × 20M timesteps = 120M timesteps
- Estimated time: Depends on hardware (likely several hours per run)

## Key Features (Ordered by ToM First)

### Experiment 1: ToM Only (Joint Rewards)
- ToM belief prediction module
- ToM supervised on ground truth beliefs (cosine similarity loss)
- **Joint rewards**: Both action and comm policies trained on task rewards
- No intrinsic reward - pure supervised learning
- Tests if ToM helps communication without explicit influence reward

### Experiment 2: ToM + Intrinsic (Separate Rewards)
- Combines ToM and intrinsic reward
- **Separate rewards**: Action learns from task reward, comm learns from intrinsic reward
- ToM predictions used for counterfactual reasoning (efficient)
- Tests if learned ToM model can guide social influence

### Experiment 3: ToM + Intrinsic (Joint Rewards)
- Combines ToM and intrinsic reward
- **Joint rewards**: Both policies learn from combined task + intrinsic rewards
- ToM predictions used for counterfactual reasoning
- Tests alternative reward structure for ToM + intrinsic combination

### Experiment 4: Baseline (Joint Rewards)
- Standard communication without intrinsic rewards or ToM
- Agents learn purely from task rewards
- **Joint rewards**: Both action and comm policies trained on task rewards
- No belief prediction or social influence modeling

### Experiment 5: Intrinsic Only (Separate Rewards)
- Adds social influence intrinsic reward (0.1 coefficient)
- **Separate rewards**: Action learns from task reward, comm learns from intrinsic reward
- Uses ground truth beliefs for counterfactual reasoning
- No ToM model - direct access to actual beliefs

### Experiment 6: Intrinsic Only (Joint Rewards)
- Adds social influence intrinsic reward (0.1 coefficient)
- **Joint rewards**: Both policies learn from combined task + intrinsic rewards
- Uses ground truth beliefs for counterfactual reasoning
- Tests alternative reward structure for intrinsic motivation

## Output Files
After completion, you'll have:
- 6 WandB runs with complete training curves (ordered by ToM first)
- Metrics logged: returns, intrinsic rewards, supervised losses
- Tags for easy filtering and comparison

## Comparison Questions

This sweep is designed to answer:

### 1. **Does ToM help?**
- Compare Exp 1 (ToM only) vs Exp 4 (Baseline)
- Both use joint rewards, no intrinsic - clean comparison

### 2. **Does intrinsic reward help (without ToM)?**
- Compare Exp 4 (Baseline) vs Exp 5 (Intrinsic sep) vs Exp 6 (Intrinsic joint)
- Tests if intrinsic rewards improve learning
- Tests separate vs joint reward structures for intrinsic motivation

### 3. **Does ToM + Intrinsic synergize?**
- Compare Exp 1 (ToM only) + Exp 5/6 (Intrinsic only) vs Exp 2/3 (ToM + Intrinsic)
- Tests if combining ToM and intrinsic is better than either alone

### 4. **Separate vs Joint rewards for intrinsic motivation?**
- **Without ToM**: Compare Exp 5 (sep) vs Exp 6 (joint)
- **With ToM**: Compare Exp 2 (sep) vs Exp 3 (joint)
- Tests which reward structure works better for intrinsic motivation

### 5. **ToM predictions vs Ground Truth for counterfactuals?**
- Compare Exp 5 (no ToM, uses ground truth) vs Exp 2 (ToM, uses predictions)
- Both use intrinsic reward with separate structure
- Tests if learned ToM is as effective as ground truth access

### 6. **Best overall configuration?**
- Compare all 6 experiments to identify best performing setup
- Considers task performance, learning efficiency, and sample complexity

## Important Notes

### Reward Structure
- **Experiments with no intrinsic** (1, 4): Always use **joint rewards** (both policies learn from task rewards)
- **Experiments with intrinsic** (2, 3, 5, 6): Test both **separate** and **joint** reward structures
  - **Separate**: Action learns from task reward, comm learns from intrinsic reward
  - **Joint**: Both learn from combined task + intrinsic rewards
- Invalid configurations (no intrinsic + separate rewards) are automatically filtered out

### Other Settings
- All experiments use **individual policies** (PARAMETER_SHARING=False)
- Belief influence uses **cosine similarity** (not KL divergence)
- ToM supervision uses **cosine similarity loss** (1 - cos_sim)
- Communication supervision is **not enabled** (only belief supervision)
- All experiments use **individual rewards** (not shared between agents)

