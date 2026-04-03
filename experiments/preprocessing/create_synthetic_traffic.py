import numpy as np
from pathlib import Path
import json
from datetime import datetime

def create_traffic(num_timesteps=5000, num_switches=34, topology='os3e'):
    traffic = np.zeros((num_timesteps, num_switches))
    
    for t in range(num_timesteps):
        hour = (t // 60) % 24
        base = 0.7 if 9 <= hour <= 17 else (0.5 if 18 <= hour <= 22 else 0.2)
        if np.random.random() < 0.05:
            base *= 1.5
        
        for s in range(num_switches):
            load = base * np.random.uniform(0.8, 1.2)
            if t > 0:
                load = 0.8 * traffic[t-1, s] + 0.2 * load
            traffic[t, s] = np.clip(load, 0, 1)
    
    output = Path(f'../../data/traffic/processed/{topology}/{topology}_synthetic_traffic.npy')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, traffic)
    
    metadata = {'topology': topology, 'timesteps': num_timesteps, 'switches': num_switches,
                'mean': float(traffic.mean()), 'created': datetime.now().isoformat()}
    with open(output.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ {topology.upper():<12} | Shape: {traffic.shape} | Mean: {traffic.mean():.3f}")

# Create for all 5
topologies = {'gridnet': 9, 'bellcanada': 48, 'os3e': 34, 'interoute': 110, 'cogentco': 197}
print("\n🌐 Creating synthetic traffic...\n")
for name, switches in topologies.items():
    create_traffic(5000, switches, name)
print("\n✅ All done!")
