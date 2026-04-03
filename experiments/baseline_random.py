#!/usr/bin/env python3
"""
Random policy baseline for comparison
"""
import sys
sys.path.append('..')

import numpy as np
from environments.proactive_sdn_env import ProactiveSDNEnv

def test_random_policy(num_episodes=10):
    """Test random action selection"""
    env = ProactiveSDNEnv()
    
    episode_rewards = []
    episode_latencies = []
    episode_energies = []
    episode_load_vars = []
    
    print("Testing Random Policy Baseline...")
    print("="*70)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        latencies = []
        energies = []
        load_vars = []
        
        while not done:
            # Random action
            action = env.action_space.sample()
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
    print("RANDOM POLICY RESULTS:")
    print("="*70)
    print(f"Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Avg Latency: {np.mean(episode_latencies):.2f} ± {np.std(episode_latencies):.2f} ms")
    print(f"Avg Energy: {np.mean(episode_energies):.2f} ± {np.std(episode_energies):.2f} W")
    print(f"Avg Load Variance: {np.mean(episode_load_vars):.4f} ± {np.std(episode_load_vars):.4f}")
    print("="*70)

if __name__ == "__main__":
    test_random_policy(num_episodes=10)
