#!/usr/bin/env python3
"""
Analyze what actions the DRL agent is actually taking
"""
import sys
sys.path.append('..')

import numpy as np
from stable_baselines3 import DQN
from environments.proactive_sdn_env_real import ProactiveSDNEnvReal
from collections import Counter

def analyze_agent_actions(model_path, num_episodes=5):
    """Analyze which actions the agent takes"""
    
    model = DQN.load(model_path)
    env = ProactiveSDNEnv()
    
    # Track actions
    all_actions = []
    action_types = {0: 'do_nothing', 1: 'migrate', 2: 'evoke', 3: 'park'}
    action_type_counts = Counter()
    
    print(f"\n{'='*70}")
    print(f"ANALYZING AGENT ACTIONS")
    print(f"{'='*70}\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_actions = []
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial State:")
        print(f"  Active Controllers: {env.num_active}")
        print(f"  Parked Controllers: {env.num_parked}")
        print(f"  Initial Energy: {env.current_energy}W")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Decode action to understand what it is
            if action == 0:
                action_type = 'do_nothing'
            elif action == 1 or action == 2:
                action_type = 'evoke'
            elif action == 3:
                action_type = 'park'
            else:
                action_type = 'migrate'
            
            episode_actions.append(action_type)
            all_actions.append(action)
            action_type_counts[action_type] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Count actions in this episode
        episode_counter = Counter(episode_actions)
        print(f"\nActions taken in Episode {episode + 1}:")
        for act_type, count in episode_counter.items():
            print(f"  {act_type}: {count} times ({count/len(episode_actions)*100:.1f}%)")
        
        print(f"Final Energy: {info['energy']}W")
        print(f"Active Controllers: {env.num_active + int(np.sum(env.parked_status))}")
    
    # Overall statistics
    print(f"\n{'='*70}")
    print(f"OVERALL ACTION DISTRIBUTION")
    print(f"{'='*70}")
    
    total_actions = sum(action_type_counts.values())
    for action_type in ['do_nothing', 'migrate', 'evoke', 'park']:
        count = action_type_counts[action_type]
        percentage = (count / total_actions) * 100
        print(f"{action_type:15s}: {count:5d} times ({percentage:5.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"INTERPRETATION:")
    print(f"{'='*70}")
    
    if action_type_counts['park'] == 0 and action_type_counts['evoke'] == 0:
        print("❌ Agent has NOT learned to park/evoke controllers!")
        print("   It only does switch migration and do-nothing.")
        print("\n💡 To fix this:")
        print("   1. Increase energy reward weight")
        print("   2. Add dynamic traffic patterns")
        print("   3. Stronger energy penalty in reward function")
    elif action_type_counts['park'] > 0 or action_type_counts['evoke'] > 0:
        print("✅ Agent IS using park/evoke actions!")
        print(f"   Park actions: {action_type_counts['park']}")
        print(f"   Evoke actions: {action_type_counts['evoke']}")
    else:
        print("⚠️  Mixed behavior - needs more investigation")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    analyze_agent_actions("../models/proactive_dqn_final.zip", num_episodes=5)
