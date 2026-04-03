#!/usr/bin/env python3
"""
COMPLETE FIXED VERSION - Test Trained Models with Actual Model Loading

This version:
1. ✅ Loads your trained Rainbow DQN model
2. ✅ Uses model for action selection (not random!)
3. ✅ Applies real traffic data to environment
4. ✅ Gets accurate performance metrics
"""

import sys
sys.path.append('../..')

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import glob

# Import your Rainbow DQN model
from environments.rainbow_dqn_model import RainbowDQN

# Import environment
from environments.threshold_proactive_sdn_env import ThresholdBasedProactiveSDN


class RealTrafficEnvironment(ThresholdBasedProactiveSDN):
    """
    Extended environment that uses REAL traffic data
    """
    
    def __init__(self, traffic_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_traffic_data = traffic_data
        self.traffic_index = 0
        self.max_traffic_steps = len(traffic_data)
        print(f"   ✅ Real traffic loaded: {self.max_traffic_steps} timesteps")
    
    def _simulate_traffic_variation(self):
        """
        Override to use REAL traffic data instead of simulation
        """
        # Get current traffic from real data
        if self.traffic_index >= self.max_traffic_steps:
            self.traffic_index = 0  # Loop back
        
        current_traffic = self.real_traffic_data[self.traffic_index]
        self.traffic_index += 1
        
        # Reset controller loads
        self.controller_loads = {i: 0.0 for i in range(self.total_controllers)}
        
        # Distribute real traffic to controllers
        for switch_id in range(self.num_switches):
            controller_id = self.switch_to_controller[switch_id]
            
            # Skip parked controllers
            if controller_id in self.parked_slaves:
                continue
            
            # Get traffic for this switch from real data
            if switch_id < len(current_traffic):
                switch_load = current_traffic[switch_id]
            else:
                switch_load = current_traffic[switch_id % len(current_traffic)]
            
            # Add to controller's load
            self.controller_loads[controller_id] += switch_load
        
        # Average load per switch for each controller
        for controller_id in self.active_slaves:
            num_assigned = sum(
                1 for s, c in self.switch_to_controller.items() 
                if c == controller_id
            )
            
            if num_assigned > 0:
                self.controller_loads[controller_id] /= num_assigned
                self.controller_loads[controller_id] = np.clip(
                    self.controller_loads[controller_id], 0, 1
                )


class ModelTester:
    """Test pre-trained Rainbow DQN models on real traffic"""
    
    def __init__(self, model_path, traffic_path, topology_name):
        self.model_path = Path(model_path)
        self.traffic_path = Path(traffic_path)
        self.topology_name = topology_name.strip().lower()
        
        print(f"\n{'='*70}")
        print(f"🧪 COMPLETE MODEL TESTING (WITH ACTUAL MODEL)")
        print(f"{'='*70}\n")
        print(f"Model: {self.model_path.name}")
        print(f"Traffic: {self.traffic_path.name}")
        print(f"Topology: {topology_name.upper()}")
        
        # Load traffic data
        self.traffic_data = np.load(traffic_path)
        print(f"\n✅ Traffic loaded: {self.traffic_data.shape}")
        print(f"   Mean load: {self.traffic_data.mean():.4f}")
        
        # Load trained model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load trained Rainbow DQN model"""
        print(f"\n📦 Loading trained model...")
        
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Get dimensions from checkpoint
        if isinstance(checkpoint, dict):
            state_dim = checkpoint.get('state_dim')
            action_dim = checkpoint.get('action_dim')
            
            # 🔥 CRITICAL: Get hidden_dim from checkpoint
            hidden_dim = None
            
            # Try to get from hyperparameters first
            if 'hyperparameters' in checkpoint:
                hidden_dim = checkpoint['hyperparameters'].get('hidden_dim')
            
            # If not found, infer from weight shapes
            if hidden_dim is None and 'online_net' in checkpoint:
                # Get hidden dim from first feature layer
                first_layer_key = 'feature.0.weight'
                if first_layer_key in checkpoint['online_net']:
                    hidden_dim = checkpoint['online_net'][first_layer_key].shape[0]
                else:
                    hidden_dim = 256  # Fallback
            
            # If state/action dims missing, infer them too
            if state_dim is None or action_dim is None:
                if 'online_net' in checkpoint:
                    first_layer_key = 'feature.0.weight'
                    if first_layer_key in checkpoint['online_net']:
                        state_dim = checkpoint['online_net'][first_layer_key].shape[1]
                        hidden_dim = checkpoint['online_net'][first_layer_key].shape[0]
                    
                    # Get action dim from advantage stream
                    adv_keys = [k for k in checkpoint['online_net'].keys() 
                            if 'advantage_stream.2' in k and 'weight_mu' in k]
                    if adv_keys:
                        # Total advantage weights / n_atoms = action_dim
                        action_dim = checkpoint['online_net'][adv_keys[0]].shape[0] // 51
                    else:
                        action_dim = 40
                else:
                    state_dim = 46
                    action_dim = 40
                    hidden_dim = 256
        else:
            state_dim = 46
            action_dim = 40
            hidden_dim = 256
        
        print(f"   State dim: {state_dim}, Action dim: {action_dim}, Hidden dim: {hidden_dim}")
        self.model_state_dim = state_dim
        self.model_action_dim = action_dim
        
        # 🔥 CREATE MODEL WITH CORRECT HIDDEN_DIM!
        model = RainbowDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,  # ← Use inferred value!
            lr=1e-4,
            gamma=0.99,
            n_step=3,
            device='cpu'
        )
        
        # Load weights
        if isinstance(checkpoint, dict) and 'online_net' in checkpoint:
            model.online_net.load_state_dict(checkpoint['online_net'])
        else:
            model.online_net.load_state_dict(checkpoint)
        
        model.online_net.eval()
        
        print(f"   ✅ Model loaded successfully")
        
        return model
    
    def test(self, num_episodes=10, steps_per_episode=500):
        """
        Test model on REAL traffic data using ACTUAL model
        """
        
        print(f"\n{'='*70}")
        print(f"🏃 RUNNING TEST (USING TRAINED MODEL)")
        print(f"{'='*70}\n")
        print(f"Episodes: {num_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        
        # Create environment WITH REAL TRAFFIC
        env = RealTrafficEnvironment(
            traffic_data=self.traffic_data,
            topology_name=self.topology_name,
            num_slave_controllers=3,
            num_parked_controllers=2
        )

        env_state_dim = env.observation_space.shape[0]
        env_action_dim = env.action_space.n
        if env_state_dim != self.model_state_dim or env_action_dim != self.model_action_dim:
            raise ValueError(
                f"Model/environment mismatch: model expects state_dim={self.model_state_dim}, "
                f"action_dim={self.model_action_dim}, but env provides "
                f"state_dim={env_state_dim}, action_dim={env_action_dim}."
            )
        
        # Test metrics
        episode_rewards = []
        parking_events = []
        evoking_events = []
        latencies = []
        energies = []
        load_variances = []
        
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
                # 🔥 USE TRAINED MODEL (not random!)
                action = self.model.select_action(state, training=False)
                
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
            'model_used': 'Rainbow DQN (trained)',  # ✅ Not random!
            
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
            'pe_ratio': float(total_parking / total_evoking) if total_evoking > 0 else None,
            
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
        
        print(f"Model: {results['model_used']} ✅")
        print(f"\nPerformance:")
        print(f"  Avg Reward: {results['avg_reward']:.2f} (±{results['std_reward']:.2f})")
        print(f"  Avg Latency: {results['avg_latency']:.2f} ms")
        print(f"  Avg Energy: {results['avg_energy']:.2f} W")
        print(f"  Load Variance: {results['avg_load_variance']:.4f}")
        
        print(f"\nProactive Behavior:")
        print(f"  Total Parking: {results['total_parking']}")
        print(f"  Total Evoking: {results['total_evoking']}")
        print(f"  Avg Park/Episode: {results['avg_parking_per_ep']:.1f}")
        print(f"  Avg Evoke/Episode: {results['avg_evoking_per_ep']:.1f}")
        pe_ratio = results['pe_ratio']
        print(f"  P/E Ratio: {pe_ratio:.2f}" if pe_ratio is not None else "  P/E Ratio: N/A (no evoke events)")
        
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
        filename = f"{results['topology']}_REAL_MODEL_test_{timestamp}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved: {output_path}")
        return output_path


def test_all_models():
    """Test all 5 topology models WITH ACTUAL MODEL LOADING"""
    
    print(f"\n{'='*70}")
    print(f"🧪 TESTING ALL MODELS (USING TRAINED MODELS - NOT RANDOM!)")
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
        
        try:
            # Test with REAL MODEL
            tester = ModelTester(model_path, traffic_path, topo)
            results = tester.test(num_episodes=10, steps_per_episode=500)
            tester.print_results(results)
            output_path = tester.save_results(results)
            
            all_results[topo] = results
        except Exception as e:
            print(f"❌ Error testing {topo}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"📊 SUMMARY - ALL TOPOLOGIES (REAL MODEL PERFORMANCE)")
        print(f"{'='*70}\n")
        
        print(f"{'Topology':<15} {'Reward':<12} {'Parking':<10} {'P/E Ratio':<10}")
        print(f"{'─'*70}")
        
        for topo, res in all_results.items():
            pe_ratio_text = f"{res['pe_ratio']:.2f}" if res['pe_ratio'] is not None else 'N/A'
            print(f"{topo.upper():<15} "
                  f"{res['avg_reward']:<12.2f} "
                  f"{res['total_parking']:<10} "
                  f"{pe_ratio_text:<10}")
        
        print(f"\n✅ All tests complete!")
        print(f"   Results in: experiments/results/test_on_real/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Trained Models (PROPERLY!)')
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
        print("    python test_trained_models_FIXED.py --test-all")
        print("\n  Test single model:")
        print("    python test_trained_models_FIXED.py \\")
        print("      --model ../models/LATEST_rainbow_proactive_os3e.pth \\")
        print("      --traffic ../../data/traffic/processed/os3e/os3e_synthetic_traffic.npy \\")
        print("      --topology os3e")
