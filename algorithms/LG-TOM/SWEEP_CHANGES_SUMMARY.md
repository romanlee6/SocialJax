# Sweep Configuration Changes Summary

## Changes Made

### 1. Expanded Experiment Count: 4 → 6
Added joint reward variants for intrinsic reward conditions (Exp 2 and 4), resulting in 6 total experiments.

### 2. Reordered Experiments
Changed order to run **ToM=True experiments first**, then ToM=False experiments.

## New Experiment Matrix (Ordered by ToM First)

| # | ToM | Intrinsic | Rewards | Coeff | Supervised | Description |
|---|-----|-----------|---------|-------|------------|-------------|
| **1** | ✅ | ❌ | Joint | 0.0 | ground_truth | ToM only |
| **2** | ✅ | ✅ | Separate | 0.1 | ground_truth | ToM + Intr (sep) |
| **3** | ✅ | ✅ | Joint | 0.1 | ground_truth | ToM + Intr (joint) |
| **4** | ❌ | ❌ | Joint | 0.0 | none | Baseline |
| **5** | ❌ | ✅ | Separate | 0.1 | none | Intr only (sep) |
| **6** | ❌ | ✅ | Joint | 0.1 | none | Intr only (joint) |

## Key Differences from Previous Setup

### Previous (4 experiments):
- Exp 1: Baseline (joint)
- Exp 2: Intrinsic only (separate)
- Exp 3: ToM only (joint)
- Exp 4: ToM + Intrinsic (separate)

### New (6 experiments):
- **Reordered**: ToM experiments first (1-3), then non-ToM (4-6)
- **Added Exp 3**: ToM + Intrinsic with joint rewards
- **Added Exp 6**: Intrinsic only with joint rewards

## Run Names (New Naming Convention)

Format: `lgtom_{tom}_{intrinsic}_{reward_structure}_{coeff}_s{seed}`

1. `lgtom_tom_nointr_joint_c0.0_s42`
2. `lgtom_tom_intr_sep_c0.1_s42`
3. `lgtom_tom_intr_joint_c0.1_s42` ← **NEW**
4. `lgtom_notom_nointr_joint_c0.0_s42`
5. `lgtom_notom_intr_sep_c0.1_s42`
6. `lgtom_notom_intr_joint_c0.1_s42` ← **NEW**

## Additional Research Questions Now Answerable

### 1. Separate vs Joint Rewards for Intrinsic Motivation
- **Without ToM**: Compare Exp 5 (sep) vs Exp 6 (joint)
- **With ToM**: Compare Exp 2 (sep) vs Exp 3 (joint)
- Tests which reward structure is better for intrinsic motivation

### 2. Consistency Across Reward Structures
- Does ToM help equally with both reward structures?
- Does intrinsic reward benefit from joint training?

## Implementation Details

### Sweep Configuration
- Grid sweep over 3 parameters: `USE_TOM` × `USE_INTRINSIC_REWARD` × `USE_SEPARATE_REWARDS`
- Total combinations: 2 × 2 × 2 = 8
- Valid combinations: 6 (2 filtered out)

### Filtered Combinations
Invalid configs automatically skipped:
- `USE_INTRINSIC_REWARD=False` + `USE_SEPARATE_REWARDS=True` (both filtered)

### WandB Agent Count
- Set to `count=8` (accounts for 2 filtered combinations)
- Results in 6 successful runs

## Benefits of New Setup

1. **More comprehensive**: Tests both reward structures for intrinsic conditions
2. **Better ordered**: ToM experiments first for priority results
3. **Cleaner comparisons**: Can isolate reward structure effects
4. **Flexible analysis**: More data points for understanding interactions

## Files Modified

1. ✅ `/algorithms/LG-TOM/lgtom_cnn_coins.py` - tune() function
2. ✅ `/algorithms/LG-TOM/TUNE_SWEEP_SETUP.md` - Documentation updated
3. ✅ `/algorithms/LG-TOM/SWEEP_CHANGES_SUMMARY.md` - This file

## Total Training Time

- **Previous**: 4 runs × 20M timesteps = 80M timesteps
- **New**: 6 runs × 20M timesteps = 120M timesteps
- **Increase**: 50% more compute, but 2x more combinations for intrinsic conditions

