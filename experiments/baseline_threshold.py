#!/usr/bin/env python3
"""
Threshold-based policy baseline
Reactive approach: migrate switches when controller exceeds threshold
"""
import sys
sys.path.append('..')

import numpy as np
from environments.proactive_sdn_env import ProactiveSDNEnv

class ThresholdPolicy:
    """Simple threshold-based load balancing"""
    def __init__(self, high_threshold=0.7, low_threshold=0.3):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
    
    def select_action(self, env):
        """
        Select action based on thresholds:
        - If controller load > high_threshold: migrate switches away
        - If controller load < low_threshold: consider parking
        """
        loads = env.controller_loads
        
        # Find overloaded controller
        overloaded = np.where(loads > self.high_threshold)[0]
        underloaded = np.where(loads < self.low_threshold)[0]
        
        if len(overloaded) > 0 and len(underloaded) > 0:
            # Migrate switch from overloaded to underloaded
            overloaded_id = overloaded[0]
            underloaded_id = underloaded[0]
            
            # Find switches on overloaded controller
            switches_to_migrate = np.where(env.switch_mappings == overloaded_id)[0]
            
            if len(switches_to_migrate) > 0:
                switch_id = switches_to_migrate[0]
                # Action encoding: 4 + switch_id * num_active + target_controller
                action = 4 + switch_id * env.num_active + underloaded_id
                return action
        
        # No action needed
        return 0

def test_threshold_policy(num_episodes=10):
    """Test threshold-based policy"""
    env = ProactiveSDNEnv()
    policy = ThresholdPolicy(high_threshold=0.7, low_threshold=0.3)
    
    episode_rewards = []
    episode_latencies = []
    episode_energies = []
    episode_load_vars = []
    
    print("Testing Threshold-Based Policy Baseline...")
    print("="*70)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        latencies = []
        energies = []
        load_vars = []
        
        while not done:
            # Threshold-based action
            action = policy.select_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            latencies.append(info['latency'])
            energies.append(info['energy'])
            load_vars.append(info['load_variance'])
        
        episode_rewards.append(episode_reward)
        episode_latencies.append(np.mean(latencies))
        episode_energies.append(np.mean(energies))
        episode_load_vars.append(np.mean(load_vars))
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Latency={np.mean(latencies):.2f}ms, "
              f"Energy={np.mean(energies):.2f}W")
    
    print("\n" + "="*70)
    print("THRESHOLD POLICY RESULTS:")
    print("="*70)
    print(f"Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Avg Latency: {np.mean(episode_latencies):.2f} ± {np.std(episode_latencies):.2f} ms")
    print(f"Avg Energy: {np.mean(episode_energies):.2f} ± {np.std(episode_energies):.2f} W")
    print(f"Avg Load Variance: {np.mean(episode_load_vars):.4f} ± {np.std(episode_load_vars):.4f}")
    print("="*70)

if __name__ == "__main__":
    test_threshold_policy(num_episodes=10)
