#!/usr/bin/env python3
"""Simple test - just use standard environment"""
import sys
sys.path.append('../..')

import torch
import numpy as np
from environments.threshold_proactive_sdn_env import ThresholdBasedProactiveSDN
from environments.rainbow_dqn_model import RainbowDQN

# Load model
print("Loading model...")
checkpoint = torch.load('../models/LATEST_rainbow_proactive_os3e.pth', 
                       map_location='cpu', weights_only=False)

state_dim = checkpoint['state_dim']
action_dim = checkpoint['action_dim']

agent = RainbowDQN(state_dim, action_dim, device='cpu')
agent.online_net.load_state_dict(checkpoint['online_net'])
agent.online_net.eval()

print(f"Model loaded: state_dim={state_dim}, action_dim={action_dim}")

# Test with standard environment
env = ThresholdBasedProactiveSDN(
    topology_name='Os3e',
    num_slave_controllers=3,
    num_parked_controllers=2
)

print("\nTesting for 5 episodes...")

total_parking = 0
total_evoking = 0

for ep in range(5):
    state, _ = env.reset()
    ep_parking = 0
    ep_evoking = 0
    
    for step in range(500):
        action = agent.select_action(state, training=False)
        next_state, reward, done, truncated, info = env.step(action)
        
        if 'park' in info.get('action_type', '') and info.get('action_success'):
            ep_parking += 1
        elif 'evoke' in info.get('action_type', '') and info.get('action_success'):
            ep_evoking += 1
        
        state = next_state
        if done or truncated:
            break
    
    total_parking += ep_parking
    total_evoking += ep_evoking
    print(f"Episode {ep+1}: Parking={ep_parking}, Evoking={ep_evoking}")

print(f"\n✅ Total: Parking={total_parking}, Evoking={total_evoking}")
print(f"P/E Ratio: {total_parking/total_evoking if total_evoking > 0 else 'N/A'}")
