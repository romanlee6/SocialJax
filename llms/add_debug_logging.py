"""
Script to add debug logging to coins_llm_simulation.py

This script modifies the existing simulation file to add detailed logging
of LLM inputs and outputs to help diagnose the "always up" issue.

Usage:
    python add_debug_logging.py          # Add logging
    python add_debug_logging.py --revert # Remove logging (restore original)
"""

import sys
import os
import shutil
from datetime import datetime


def add_logging_to_simulation(filepath: str = "./coins_llm_simulation.py"):
    """Add debug logging to the simulation file."""
    
    # Create backup
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✓ Created backup: {backup_path}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "### DEBUG LOGGING START ###" in content:
        print("⚠️  File already contains debug logging")
        return backup_path
    
    # Patch 1: Add logging after LLM API call (around line 316)
    patch1_search = '            llm_output = response.output[1].content[0].text\n            print(f"LLM output: {llm_output}")'
    
    patch1_replace = '''            llm_output = response.output[1].content[0].text
            
            ### DEBUG LOGGING START ###
            print(f"\\n{'='*70}")
            print(f"Agent {self.agent_id} - Timestep {timestep}")
            print(f"{'='*70}")
            print(f"\\nLLM RAW OUTPUT:")
            print(llm_output)
            print(f"\\n{'='*70}\\n")
            ### DEBUG LOGGING END ###
            
            print(f"LLM output: {llm_output}")'''
    
    if patch1_search in content:
        content = content.replace(patch1_search, patch1_replace)
        print("✓ Added logging after LLM API call")
    else:
        print("⚠️  Could not find exact location for patch 1 (LLM output logging)")
    
    # Patch 2: Add logging after parsing (around line 329)
    patch2_search = '''        print(f"Belief: {belief}")
        print(f"Action: {action}")
        print(f"Communication: {communication}")'''
    
    patch2_replace = '''        print(f"Belief: {belief}")
        print(f"Action: {action}")
        print(f"Communication: {communication}")
        
        ### DEBUG LOGGING START ###
        from coins_llm_simulation import ActionParser
        action_idx = ActionParser.parse(action)
        print(f"\\n{'='*70}")
        print(f"Agent {self.agent_id} - PARSED RESULTS")
        print(f"{'='*70}")
        print(f"Belief: {belief[:100]}...")
        print(f"Action String: '{action}'")
        print(f"Action Index: {action_idx}")
        print(f"Expected Indices: turn_left=0, turn_right=1, left=2, right=3, up=4, down=5, stay=6")
        print(f"Communication: {communication[:100]}...")
        print(f"{'='*70}\\n")
        ### DEBUG LOGGING END ###'''
    
    if patch2_search in content:
        content = content.replace(patch2_search, patch2_replace)
        print("✓ Added logging after parsing")
    else:
        print("⚠️  Could not find exact location for patch 2 (parsing logging)")
    
    # Patch 3: Add action distribution tracking at end of simulation
    patch3_search = '    print(f"\\nSimulation complete! Results saved to: {save_dir}")'
    
    patch3_replace = '''    ### DEBUG LOGGING START ###
    print("\\n" + "="*70)
    print("DEBUG: ACTION DISTRIBUTION ANALYSIS")
    print("="*70)
    # Note: This is a placeholder - action tracking would need to be added to the simulation loop
    print("To track actions, add a list in run_simulation() to collect all actions.")
    print("="*70 + "\\n")
    ### DEBUG LOGGING END ###
    
    print(f"\\nSimulation complete! Results saved to: {save_dir}")'''
    
    if patch3_search in content:
        content = content.replace(patch3_search, patch3_replace)
        print("✓ Added action distribution placeholder")
    else:
        print("⚠️  Could not find exact location for patch 3 (action distribution)")
    
    # Write modified content
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Debug logging added to {filepath}")
    print(f"✓ Original backed up to {backup_path}")
    
    return backup_path


def revert_logging(filepath: str = "./coins_llm_simulation.py"):
    """Remove debug logging from the simulation file."""
    
    # Find most recent backup
    import glob
    backups = sorted(glob.glob(f"{filepath}.backup_*"), reverse=True)
    
    if not backups:
        print("✗ No backup files found")
        return
    
    latest_backup = backups[0]
    print(f"Found backup: {latest_backup}")
    
    # Ask for confirmation
    response = input(f"Restore from {latest_backup}? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    # Restore backup
    shutil.copy2(latest_backup, filepath)
    print(f"✓ Restored {filepath} from backup")
    
    # Optionally delete backup
    response = input(f"Delete backup file? [y/N]: ")
    if response.lower() == 'y':
        os.remove(latest_backup)
        print(f"✓ Deleted {latest_backup}")


def create_action_tracker_patch():
    """Create a comprehensive patch file for action tracking."""
    
    patch_content = '''# Comprehensive Patch for coins_llm_simulation.py
# This patch adds full action tracking to the simulation

# ==============================================================================
# PATCH 1: Add action tracking list in run_simulation()
# ==============================================================================
# Location: After line 690 (rewards = np.array([0.0, 0.0]))

# ADD THESE LINES:
    # Track all actions for debugging
    all_actions_taken = []  # List of (timestep, agent_id, action_str, action_idx)

# ==============================================================================
# PATCH 2: Record actions in simulation loop
# ==============================================================================
# Location: After line 733 (actions = [ActionParser.parse(a) for a in actions_str])

# ADD THESE LINES:
        # Record actions for debugging
        for i, (action_str, action_idx) in enumerate(zip(actions_str, actions)):
            all_actions_taken.append((t, i, action_str, action_idx))

# ==============================================================================
# PATCH 3: Analyze actions at end of simulation
# ==============================================================================
# Location: Replace line 763 with:

    # Analyze action distribution
    print("\\n" + "="*70)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("="*70)
    
    action_counts = {}
    for timestep, agent_id, action_str, action_idx in all_actions_taken:
        key = (agent_id, action_str)
        action_counts[key] = action_counts.get(key, 0) + 1
    
    for agent_id in range(2):
        print(f"\\nAgent {agent_id}:")
        agent_actions = [(action_str, count) for (aid, action_str), count in action_counts.items() if aid == agent_id]
        agent_actions.sort(key=lambda x: x[1], reverse=True)
        
        total = sum(count for _, count in agent_actions)
        for action_str, count in agent_actions:
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {action_str:12s}: {count:3d} times ({percentage:5.1f}%)")
        
        # Warning if all same action
        if len(agent_actions) == 1:
            print(f"  ⚠️  WARNING: Agent {agent_id} only used '{agent_actions[0][0]}' action!")
    
    print("="*70)
    
    print(f"\\nSimulation complete! Results saved to: {save_dir}")

# ==============================================================================
# HOW TO APPLY THIS PATCH
# ==============================================================================
# 1. Open coins_llm_simulation.py
# 2. Find each location mentioned above
# 3. Add or replace the lines as indicated
# 4. Save and run the simulation
# 5. Check the ACTION DISTRIBUTION ANALYSIS at the end
'''
    
    patch_file = "./action_tracker_patch.txt"
    with open(patch_file, 'w') as f:
        f.write(patch_content)
    
    print(f"✓ Created patch file: {patch_file}")
    print("\nThis file contains instructions for manually adding")
    print("comprehensive action tracking to your simulation.")
    
    return patch_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Add/remove debug logging from coins_llm_simulation.py")
    parser.add_argument("--revert", action="store_true", help="Revert to backup")
    parser.add_argument("--file", type=str, default="./coins_llm_simulation.py",
                       help="Path to simulation file")
    parser.add_argument("--create-patch", action="store_true",
                       help="Create manual patch file for action tracking")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Debug Logging Patcher for coins_llm_simulation.py")
    print("="*70)
    
    if args.create_patch:
        patch_file = create_action_tracker_patch()
        print(f"\n✓ Created {patch_file}")
        print("\nRead this file for instructions on adding full action tracking.")
        return
    
    if args.revert:
        revert_logging(args.file)
    else:
        backup = add_logging_to_simulation(args.file)
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Run your simulation:")
        print("   python coins_llm_simulation.py --steps 10")
        print("\n2. Look for the debug output sections marked with === lines")
        print("\n3. Check if LLM is outputting 'up' or if parsing fails")
        print("\n4. To remove logging and restore original:")
        print(f"   python add_debug_logging.py --revert")
        print("\n5. For comprehensive action tracking:")
        print("   python add_debug_logging.py --create-patch")
        print("="*70)


if __name__ == "__main__":
    main()

