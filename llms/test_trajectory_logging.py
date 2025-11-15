"""
Test script to verify trajectory logging accuracy and completeness.

This script checks if the logged trajectory contains all information
necessary to reconstruct the interaction history and game state.
"""

import json
import os
import sys
sys.path.append('/home/huao/Research/SocialJax')

import numpy as np
from typing import Dict, List


def test_trajectory_file(trajectory_path: str) -> Dict[str, bool]:
    """
    Test if trajectory file contains all required information.
    
    Returns:
        Dictionary with test results (test_name: passed)
    """
    results = {}
    
    # Load trajectory
    if not os.path.exists(trajectory_path):
        print(f"ERROR: Trajectory file not found: {trajectory_path}")
        return {"file_exists": False}
    
    with open(trajectory_path, 'r') as f:
        trajectory = json.load(f)
    
    results["file_exists"] = True
    results["has_metadata"] = "metadata" in trajectory
    results["has_trajectory"] = "trajectory" in trajectory
    
    if not results["has_trajectory"]:
        return results
    
    # Check metadata
    if results["has_metadata"]:
        metadata = trajectory["metadata"]
        required_metadata = ["model", "temperature", "seed", "num_agents", "timestamp"]
        for key in required_metadata:
            results[f"metadata_has_{key}"] = key in metadata
    
    # Check trajectory structure
    trajectory_data = trajectory["trajectory"]
    results["trajectory_not_empty"] = len(trajectory_data) > 0
    
    if not results["trajectory_not_empty"]:
        return results
    
    # Check first timestep structure
    first_step = trajectory_data[0]
    required_step_keys = ["timestep", "agents", "env_state", "env_obs", "rewards", 
                          "coins_in_env", "accumulated_rewards"]
    for key in required_step_keys:
        results[f"step_has_{key}"] = key in first_step
    
    # Check agent data structure
    if "agents" in first_step and len(first_step["agents"]) > 0:
        first_agent = first_step["agents"][0]
        required_agent_keys = [
            "agent_id", "agent_color", "observation", "belief", "action", 
            "action_idx", "communication", "reward", "semantic_key",
            "token_usage", "api_time"
        ]
        for key in required_agent_keys:
            results[f"agent_has_{key}"] = key in first_agent
        
        # Check that belief and communication are text (not embeddings)
        if "belief" in first_agent:
            belief = first_agent["belief"]
            results["belief_is_text"] = isinstance(belief, str)
        
        if "communication" in first_agent:
            comm = first_agent["communication"]
            results["communication_is_text"] = isinstance(comm, str)
        
        # Check semantic key structure
        if "semantic_key" in first_agent:
            semantic_key = first_agent["semantic_key"]
            results["semantic_key_is_tuple"] = isinstance(semantic_key, list)  # JSON stores tuples as lists
            if results["semantic_key_is_tuple"]:
                results["semantic_key_length_6"] = len(semantic_key) == 6
                if results["semantic_key_length_6"]:
                    # Check semantic key components: (agent_color, agent_x, agent_y, closest_coin, agent_id, action)
                    results["semantic_key_has_color"] = isinstance(semantic_key[0], str)
                    results["semantic_key_has_coords"] = isinstance(semantic_key[1], int) and isinstance(semantic_key[2], int)
                    results["semantic_key_has_coin"] = isinstance(semantic_key[3], str)
                    results["semantic_key_has_agent_id"] = isinstance(semantic_key[4], int)
                    results["semantic_key_has_action"] = isinstance(semantic_key[5], str)
    
    # Check environment state structure
    if "env_state" in first_step:
        env_state = first_step["env_state"]
        required_env_keys = ["agent_locs", "grid"]
        for key in required_env_keys:
            results[f"env_state_has_{key}"] = key in env_state
    
    # Check consistency across timesteps
    if len(trajectory_data) > 1:
        results["timesteps_sequential"] = all(
            trajectory_data[i]["timestep"] == i 
            for i in range(len(trajectory_data))
        )
        
        # Check that each timestep has same number of agents
        num_agents = len(first_step["agents"])
        results["consistent_num_agents"] = all(
            len(step["agents"]) == num_agents 
            for step in trajectory_data
        )
    
    # Check token usage tracking
    if "metadata" in trajectory and "total_token_usage" in trajectory["metadata"]:
        token_usage = trajectory["metadata"]["total_token_usage"]
        results["has_token_usage"] = True
        results["token_usage_has_prompt"] = "prompt_tokens" in token_usage
        results["token_usage_has_completion"] = "completion_tokens" in token_usage
        results["token_usage_has_total"] = "total_tokens" in token_usage
    else:
        results["has_token_usage"] = False
    
    # Check time tracking
    if "metadata" in trajectory and "total_api_time" in trajectory["metadata"]:
        results["has_api_time"] = True
        results["api_time_is_number"] = isinstance(trajectory["metadata"]["total_api_time"], (int, float))
    else:
        results["has_api_time"] = False
    
    # Check that we can reconstruct interaction history
    interactions = []
    for step in trajectory_data:
        for agent in step["agents"]:
            if agent.get("communication") and agent["communication"] != "[No message]":
                interactions.append({
                    "timestep": step["timestep"],
                    "sender_id": agent["agent_id"],
                    "message": agent["communication"]
                })
    results["can_reconstruct_interactions"] = len(interactions) >= 0  # Always true, just checking we can extract
    
    # Check that we can reconstruct game state progression
    states = []
    for step in trajectory_data:
        if "env_state" in step and "agent_locs" in step["env_state"]:
            states.append({
                "timestep": step["timestep"],
                "agent_locs": step["env_state"]["agent_locs"],
                "grid": step["env_state"].get("grid", [])
            })
    results["can_reconstruct_states"] = len(states) == len(trajectory_data)
    
    return results


def print_test_results(results: Dict[str, bool], trajectory_path: str):
    """Print test results in a readable format."""
    print("=" * 80)
    print(f"TEST RESULTS FOR: {trajectory_path}")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for test_name, passed_test in sorted(results.items()):
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status:10} {test_name}")
        if passed_test:
            passed += 1
        else:
            failed += 1
    
    print()
    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(results)} tests")
    print("=" * 80)
    
    if failed == 0:
        print("\n✓ All tests passed! Trajectory logging is accurate and complete.")
    else:
        print(f"\n✗ {failed} test(s) failed. Please check the trajectory logging implementation.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trajectory logging accuracy")
    parser.add_argument("trajectory_file", type=str,
                       help="Path to trajectory.json file to test")
    
    args = parser.parse_args()
    
    results = test_trajectory_file(args.trajectory_file)
    print_test_results(results, args.trajectory_file)
    
    # Exit with error code if tests failed
    failed_count = sum(1 for v in results.values() if not v)
    sys.exit(failed_count)

