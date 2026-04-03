import sys
sys.path.append('../..')

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import glob

# Your environment
from environments.threshold_proactive_sdn_env import ThresholdBasedProactiveSDN


class ModelTester:
    """Test pre-trained models on new traffic data"""
    
    def __init__(self, model_path, traffic_path, topology_name):
        self.model_path = Path(model_path)
        self.traffic_path = Path(traffic_path)
        self.topology_name = topology_name
        
        print(f"\n{'='*70}")
        print(f"🧪 MODEL TESTING")
        print(f"{'='*70}\n")
        print(f"Model: {self.model_path.name}")
        print(f"Traffic: {self.traffic_path.name}")
        print(f"Topology: {topology_name.upper()}")
        
        # Load traffic data
        self.traffic_data = np.load(traffic_path)
        print(f"\n✅ Traffic loaded: {self.traffic_data.shape}")
        print(f"   Mean load: {self.traffic_data.mean():.4f}")
    
    def test(self, num_episodes=10, steps_per_episode=500):
        """
        Test model on traffic data
        
        NOTE: This is a simplified version that tests the environment
        with your traffic data. For full testing with your actual model,
        you'll need to load the model and use it for action selection.
        """
        
        print(f"\n{'='*70}")
        print(f"🏃 RUNNING TEST")
        print(f"{'='*70}\n")
        print(f"Episodes: {num_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        
        # Map topology names to match your GML files
        topology_map = {
            'bellcanada': 'BellCanada',
            'cogentco': 'Cogentco',
            'gridnet': 'Gridnet',
            'interoute': 'Interoute',
            'os3e': 'Os3e'
        }
        
        topo_file = topology_map.get(self.topology_name.lower(), self.topology_name)
        
        # Create environment
        env = ThresholdBasedProactiveSDN(
            topology_name=topo_file,
            num_slave_controllers=3,
            num_parked_controllers=2
        )
        
        # Test metrics
        episode_rewards = []
        parking_events = []
        evoking_events = []
        latencies = []
        energies = []
        load_variances = []
        
        traffic_idx = 0  # Index into traffic data
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            ep_parking = 0
            ep_evoking = 0
            ep_latencies = []
            ep_energies = []
            ep_load_vars = []
            
            print(f"Episode {ep+1}/{num_episodes}: ", end='', flush=True)
            
            for step in range(steps_per_episode):
                # Get traffic for this step
                if traffic_idx < len(self.traffic_data):
                    current_traffic = self.traffic_data[traffic_idx]
                    traffic_idx += 1
                else:
                    traffic_idx = 0
                    current_traffic = self.traffic_data[traffic_idx]
                
                # Apply traffic to environment (simplified)
                # In full version, you'd update env.controller_loads based on traffic
                
                # For now, use environment's default action selection
                # In full version, load your model and use: action = model.select_action(state)
                action = env.action_space.sample()  # Random for demonstration
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                ep_latencies.append(info.get('latency', 0))
                ep_energies.append(info.get('energy', 0))
                ep_load_vars.append(info.get('load_variance', 0))
                
                # Track events
                action_type = info.get('action_type', '')
                if 'park' in action_type and info.get('action_success'):
                    ep_parking += 1
                elif 'evoke' in action_type and info.get('action_success'):
                    ep_evoking += 1
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            parking_events.append(ep_parking)
            evoking_events.append(ep_evoking)
            latencies.append(np.mean(ep_latencies) if ep_latencies else 0)
            energies.append(np.mean(ep_energies) if ep_energies else 0)
            load_variances.append(np.mean(ep_load_vars) if ep_load_vars else 0)
            
            print(f"✅ R={episode_reward:.1f} | P={ep_parking} | E={ep_evoking}")
        
        env.close()
        
        # Calculate results
        total_parking = int(np.sum(parking_events))
        total_evoking = int(np.sum(evoking_events))
        
        results = {
            'model': str(self.model_path.name),
            'traffic_data': str(self.traffic_path.name),
            'topology': self.topology_name,
            'test_date': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            
            # Performance metrics
            'avg_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'avg_latency': float(np.mean(latencies)),
            'std_latency': float(np.std(latencies)),
            'avg_energy': float(np.mean(energies)),
            'avg_load_variance': float(np.mean(load_variances)),
            
            # Proactive behavior
            'total_parking': total_parking,
            'total_evoking': total_evoking,
            'avg_parking_per_ep': float(np.mean(parking_events)),
            'avg_evoking_per_ep': float(np.mean(evoking_events)),
            'pe_ratio': float(total_parking / total_evoking) if total_evoking > 0 else 0,
            
            # Raw data
            'episode_rewards': [float(r) for r in episode_rewards],
            'parking_per_episode': [int(p) for p in parking_events],
            'evoking_per_episode': [int(e) for e in evoking_events]
        }
        
        return results
    
    def print_results(self, results):
        """Print formatted results"""
        print(f"\n{'='*70}")
        print(f"📊 TEST RESULTS - {results['topology'].upper()}")
        print(f"{'='*70}\n")
        
        print(f"Performance:")
        print(f"  Avg Reward: {results['avg_reward']:.2f} (±{results['std_reward']:.2f})")
        print(f"  Avg Latency: {results['avg_latency']:.2f} ms")
        print(f"  Avg Energy: {results['avg_energy']:.2f} W")
        print(f"  Load Variance: {results['avg_load_variance']:.4f}")
        
        print(f"\nProactive Behavior:")
        print(f"  Total Parking: {results['total_parking']}")
        print(f"  Total Evoking: {results['total_evoking']}")
        print(f"  Avg Park/Episode: {results['avg_parking_per_ep']:.1f}")
        print(f"  Avg Evoke/Episode: {results['avg_evoking_per_ep']:.1f}")
        print(f"  P/E Ratio: {results['pe_ratio']:.2f}")
        
        # Evaluation
        if results['total_parking'] > 50:
            print(f"\n  ✅ EXCELLENT proactive behavior!")
        elif results['total_parking'] > 20:
            print(f"\n  ✅ GOOD proactive behavior")
        else:
            print(f"\n  ⚠️  LIMITED proactive behavior")
    
    def save_results(self, results, output_dir='../../experiments/results/test_on_real'):
        """Save results to JSON"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"{results['topology']}_test_results_{timestamp}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved: {output_path}")
        return output_path


def test_all_models():
    """Test all 5 topology models"""
    
    print(f"\n{'='*70}")
    print(f"🧪 TESTING ALL MODELS")
    print(f"{'='*70}\n")
    
    topologies = ['gridnet', 'bellcanada', 'os3e', 'interoute', 'cogentco']
    
    all_results = {}
    
    for topo in topologies:
        print(f"\n{'─'*70}")
        print(f"Testing {topo.upper()}")
        print(f"{'─'*70}")
        
        # Find model (use LATEST version)
        model_pattern = f'../models/LATEST_rainbow_proactive_{topo}.pth'
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            # Try timestamped version
            model_pattern = f'../models/rainbow_proactive_{topo}_*.pth'
            model_files = glob.glob(model_pattern)
        
        if not model_files:
            print(f"⚠️  No model found for {topo}")
            continue
        
        model_path = model_files[0]
        traffic_path = f'../../data/traffic/processed/{topo}/{topo}_synthetic_traffic.npy'
        
        if not Path(traffic_path).exists():
            print(f"⚠️  Traffic data not found: {traffic_path}")
            continue
        
        # Test
        tester = ModelTester(model_path, traffic_path, topo)
        results = tester.test(num_episodes=10, steps_per_episode=500)
        tester.print_results(results)
        output_path = tester.save_results(results)
        
        all_results[topo] = results
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY - ALL TOPOLOGIES")
    print(f"{'='*70}\n")
    
    print(f"{'Topology':<15} {'Reward':<12} {'Parking':<10} {'P/E Ratio':<10}")
    print(f"{'─'*70}")
    
    for topo, res in all_results.items():
        print(f"{topo.upper():<15} "
              f"{res['avg_reward']:<12.2f} "
              f"{res['total_parking']:<10} "
              f"{res['pe_ratio']:<10.2f}")
    
    print(f"\n✅ All tests complete!")
    print(f"   Results in: experiments/results/test_on_real/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Trained Models')
    parser.add_argument('--model', help='Model path')
    parser.add_argument('--traffic', help='Traffic data path')
    parser.add_argument('--topology', help='Topology name')
    parser.add_argument('--test-all', action='store_true', 
                       help='Test all 5 models')
    
    args = parser.parse_args()
    
    if args.test_all:
        test_all_models()
    elif args.model and args.traffic and args.topology:
        tester = ModelTester(args.model, args.traffic, args.topology)
        results = tester.test()
        tester.print_results(results)
        tester.save_results(results)
    else:
        print("\nUsage:")
        print("  Test all models:")
        print("    python test_trained_models.py --test-all")
        print("\n  Test single model:")
        print("    python test_trained_models.py \\")
        print("      --model ../models/LATEST_rainbow_proactive_os3e.pth \\")
        print("      --traffic ../../data/traffic/processed/os3e/os3e_synthetic_traffic.npy \\")
        print("      --topology os3e")
