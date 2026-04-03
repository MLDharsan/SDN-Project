import sys
sys.path.append('../..')
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from environments.threshold_proactive_sdn_env import ThresholdBasedProactiveSDN

topologies = {
    'gridnet': {'switches': 9, 'gml': 'Gridnet'},
    'bellcanada': {'switches': 48, 'gml': 'BellCanada'},
    'os3e': {'switches': 34, 'gml': 'Os3e'},
    'interoute': {'switches': 110, 'gml': 'Interoute'},
    'cogentco': {'switches': 197, 'gml': 'Cogentco'}
}

print("\n🧪 TESTING ALL MODELS\n")

for topo, config in topologies.items():
    print(f"Testing {topo.upper()}...")
    
    traffic = np.load(f'../../data/traffic/processed/{topo}/{topo}_synthetic_traffic.npy')
    env = ThresholdBasedProactiveSDN(topology_name=config['gml'], num_slave_controllers=3, num_parked_controllers=2)
    
    rewards, parking, evoking = [], 0, 0
    
    for ep in range(5):  # 5 episodes
        state, _ = env.reset()
        ep_reward = 0
        
        for step in range(300):  # 300 steps
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            
            if 'park' in info.get('action_type', '') and info.get('action_success'):
                parking += 1
            elif 'evoke' in info.get('action_type', '') and info.get('action_success'):
                evoking += 1
            
            if done or truncated:
                break
        
        rewards.append(ep_reward)
    
    env.close()
    
    results = {
        'topology': topo,
        'avg_reward': float(np.mean(rewards)),
        'parking': parking,
        'evoking': evoking,
        'pe_ratio': parking/evoking if evoking > 0 else 0
    }
    
    output = Path(f'../../experiments/results/test_on_real/{topo}_results.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✅ Reward: {results['avg_reward']:.1f} | Park: {parking} | Evoke: {evoking} | P/E: {results['pe_ratio']:.2f}\n")

print("✅ All tests complete! Results in: experiments/results/test_on_real/")
