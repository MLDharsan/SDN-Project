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
from collections import Counter

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
    
    def __init__(self, model_path, traffic_path, topology_name, mask_invalid_actions=False):
        self.model_path = Path(model_path)
        self.traffic_path = Path(traffic_path)
        self.topology_name = topology_name.strip().lower()
        self.mask_invalid_actions = mask_invalid_actions
        
        print(f"\n{'='*70}")
        print(f"🧪 COMPLETE MODEL TESTING (WITH ACTUAL MODEL)")
        print(f"{'='*70}\n")
        print(f"Model: {self.model_path.name}")
        print(f"Traffic: {self.traffic_path.name}")
        print(f"Topology: {topology_name.upper()}")
        print(f"Invalid action masking: {'ON' if self.mask_invalid_actions else 'OFF'}")
        
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

    @staticmethod
    def _apply_training_reward_logic(env):
        """Match the context-aware reward shaping used during training."""
        def fixed_calculate_reward(action_type: str, success: bool):
            latency = env._calculate_latency()
            worst_case_latency = env._calculate_worst_case_latency()
            energy = env._calculate_energy()
            load_variance = env._calculate_load_variance()
            load_balance_index = env._calculate_load_balance_index()

            norm_latency = latency / 20.0
            norm_energy = energy / 1000.0
            norm_load_var = load_variance / 1.0

            reward = -1.0 * norm_latency - 2.0 * norm_energy - 1.0 * norm_load_var

            thresholds = env._check_thresholds()
            has_overload = bool(thresholds['overloaded'])
            has_underload = bool(thresholds['underloaded'])
            can_still_park = len(env.parked_slaves) < env.num_parked_controllers

            if action_type == 'park' and success:
                reward += 200.0 if (has_underload and can_still_park) else -20.0
            elif action_type == 'evoke' and success:
                reward += 200.0 if has_overload else -30.0
            elif action_type == 'migrate' and success:
                reward += 2.0

            if action_type == 'noop':
                if has_overload:
                    reward -= 100.0
                if has_underload and can_still_park:
                    reward -= 80.0

            if not success and action_type != 'noop':
                reward -= 10.0

            info = {
                'latency': latency,
                'cs_avg_latency': latency,
                'worst_case_latency': worst_case_latency,
                'cs_worst_latency': worst_case_latency,
                'energy': energy,
                'load_variance': load_variance,
                'load_balance': np.sqrt(load_variance),
                'load_balance_index': load_balance_index,
                'active_controllers': len(env.active_slaves),
                'parked_controllers': len(env.parked_slaves),
                'action_type': action_type,
                'action_success': success,
                'overloaded_count': len(thresholds['overloaded']),
                'underloaded_count': len(thresholds['underloaded'])
            }

            return reward, info

        env._calculate_reward = fixed_calculate_reward

    @staticmethod
    def _format_action_id_counts(action_counts, env, limit=3):
        """Format action id counters using readable action labels."""
        items = []
        for action_id, count in action_counts.most_common(limit):
            items.append(f"{action_id}:{env.describe_action(action_id)}:{count}")
        return ", ".join(items) if items else "none"

    def _select_action(self, state, env):
        """Select greedy action, optionally masking invalid choices."""
        self.model.online_net.train(False)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
            q_values = self.model.online_net.get_q_values(state_tensor).squeeze(0)
            raw_action = int(q_values.argmax().item())

            if not self.mask_invalid_actions:
                return raw_action, raw_action

            valid_actions = env.get_valid_actions()
            masked_q_values = q_values.clone()
            invalid_actions = torch.ones_like(masked_q_values, dtype=torch.bool)
            invalid_actions[valid_actions] = False
            masked_q_values[invalid_actions] = -torch.inf
            chosen_action = int(masked_q_values.argmax().item())

        return raw_action, chosen_action
    
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
        self._apply_training_reward_logic(env)

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
        worst_case_latencies = []
        energies = []
        load_variances = []
        load_balance_indices = []
        action_type_counts = Counter()
        successful_action_counts = Counter()
        failed_action_counts = Counter()
        raw_action_counts = Counter()
        executed_action_counts = Counter()
        masked_action_override_count = 0
        overload_observations = 0
        underload_observations = 0
        overload_with_parked_capacity = 0
        underload_with_parking_capacity = 0
        successful_evokes_on_overload = 0
        successful_parks_on_underload = 0
        action_legends = {}
        oscillation_events = 0
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            ep_parking = 0
            ep_evoking = 0
            ep_latencies = []
            ep_energies = []
            ep_load_vars = []
            ep_worst_latencies = []
            ep_load_balance_indices = []
            ep_action_counts = Counter()
            ep_failed_action_counts = Counter()
            ep_raw_action_counts = Counter()
            ep_executed_action_counts = Counter()
            ep_masked_overrides = 0
            ep_oscillations = 0
            last_successful_proactive_action = None
            
            print(f"Episode {ep+1}/{num_episodes}: ", end='', flush=True)
            
            for step in range(steps_per_episode):
                # 🔥 USE TRAINED MODEL (not random!)
                raw_action, action = self._select_action(state, env)
                action_legends[str(raw_action)] = env.describe_action(raw_action)
                action_legends[str(action)] = env.describe_action(action)
                raw_action_counts[raw_action] += 1
                executed_action_counts[action] += 1
                ep_raw_action_counts[raw_action] += 1
                ep_executed_action_counts[action] += 1
                if raw_action != action:
                    masked_action_override_count += 1
                    ep_masked_overrides += 1
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                ep_latencies.append(info.get('latency', 0))
                ep_energies.append(info.get('energy', 0))
                ep_load_vars.append(info.get('load_variance', 0))
                ep_worst_latencies.append(info.get('worst_case_latency', 0))
                ep_load_balance_indices.append(info.get('load_balance_index', 0))
                
                # Track events
                action_type = info.get('action_type', '')
                action_success = info.get('action_success', False)
                action_type_counts[action_type] += 1
                ep_action_counts[action_type] += 1
                if action_success:
                    successful_action_counts[action_type] += 1
                else:
                    failed_action_counts[action_type] += 1
                    ep_failed_action_counts[action_type] += 1
                has_overload = info.get('overloaded_count', 0) > 0
                has_underload = info.get('underloaded_count', 0) > 0
                has_parked_to_evoke = info.get('parked_controllers', 0) > 0
                has_room_to_park = info.get('parked_controllers', 0) < env.num_parked_controllers
                overload_observations += int(has_overload)
                underload_observations += int(has_underload)
                overload_with_parked_capacity += int(has_overload and has_parked_to_evoke)
                underload_with_parking_capacity += int(has_underload and has_room_to_park)

                if 'park' in action_type and info.get('action_success'):
                    ep_parking += 1
                    if has_underload:
                        successful_parks_on_underload += 1
                elif 'evoke' in action_type and info.get('action_success'):
                    ep_evoking += 1
                    if has_overload:
                        successful_evokes_on_overload += 1

                if action_success and action_type in {'park', 'evoke'}:
                    if last_successful_proactive_action and last_successful_proactive_action != action_type:
                        oscillation_events += 1
                        ep_oscillations += 1
                    last_successful_proactive_action = action_type
                elif action_success and action_type == 'noop':
                    last_successful_proactive_action = None
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            parking_events.append(ep_parking)
            evoking_events.append(ep_evoking)
            latencies.append(np.mean(ep_latencies) if ep_latencies else 0)
            worst_case_latencies.append(np.max(ep_worst_latencies) if ep_worst_latencies else 0)
            energies.append(np.mean(ep_energies) if ep_energies else 0)
            load_variances.append(np.mean(ep_load_vars) if ep_load_vars else 0)
            load_balance_indices.append(np.mean(ep_load_balance_indices) if ep_load_balance_indices else 0)

            top_actions = ", ".join(
                f"{name}:{count}" for name, count in ep_action_counts.most_common(3)
            ) or "none"
            top_failures = ", ".join(
                f"{name}:{count}" for name, count in ep_failed_action_counts.most_common(2)
            ) or "none"
            top_raw_ids = self._format_action_id_counts(ep_raw_action_counts, env)
            top_exec_ids = self._format_action_id_counts(ep_executed_action_counts, env)
            print(
                f"✅ R={episode_reward:.1f} | P={ep_parking} | E={ep_evoking} | "
                f"Actions[{top_actions}] | Fail[{top_failures}] | "
                f"RawIDs[{top_raw_ids}] | ExecIDs[{top_exec_ids}] | "
                f"Masked={ep_masked_overrides} | Osc={ep_oscillations}"
            )
        
        env.close()
        
        # Calculate results
        total_parking = int(np.sum(parking_events))
        total_evoking = int(np.sum(evoking_events))
        total_steps = sum(action_type_counts.values())
        
        results = {
            'model': str(self.model_path.name),
            'traffic_data': str(self.traffic_path.name),
            'topology': self.topology_name,
            'test_date': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'model_used': 'Rainbow DQN (trained)',  # ✅ Not random!
            'reward_logic': 'CONTEXT-AWARE training reward matched during evaluation',
            'mask_invalid_actions': bool(self.mask_invalid_actions),
            
            # Performance metrics
            'avg_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'avg_latency': float(np.mean(latencies)),
            'cs_avg_latency': float(np.mean(latencies)),
            'cs_worst_latency': float(np.mean(worst_case_latencies)),
            'std_latency': float(np.std(latencies)),
            'avg_energy': float(np.mean(energies)),
            'avg_load_variance': float(np.mean(load_variances)),
            'load_balance_index': float(np.mean(load_balance_indices)),
            
            # Proactive behavior
            'total_parking': total_parking,
            'total_evoking': total_evoking,
            'avg_parking_per_ep': float(np.mean(parking_events)),
            'avg_evoking_per_ep': float(np.mean(evoking_events)),
            'pe_ratio': float(total_parking / total_evoking) if total_evoking > 0 else None,
            'total_steps': int(total_steps),
            'action_type_counts': dict(action_type_counts),
            'successful_action_counts': dict(successful_action_counts),
            'failed_action_counts': dict(failed_action_counts),
            'raw_action_id_counts': {str(k): int(v) for k, v in raw_action_counts.items()},
            'executed_action_id_counts': {str(k): int(v) for k, v in executed_action_counts.items()},
            'action_id_legend': action_legends,
            'masked_action_override_count': int(masked_action_override_count),
            'masked_action_override_fraction': float(masked_action_override_count / total_steps) if total_steps else 0.0,
            'overload_step_fraction': float(overload_observations / total_steps) if total_steps else 0.0,
            'underload_step_fraction': float(underload_observations / total_steps) if total_steps else 0.0,
            'overload_with_parked_capacity_fraction': float(overload_with_parked_capacity / total_steps) if total_steps else 0.0,
            'underload_with_parking_capacity_fraction': float(underload_with_parking_capacity / total_steps) if total_steps else 0.0,
            'successful_evokes_on_overload': int(successful_evokes_on_overload),
            'successful_parks_on_underload': int(successful_parks_on_underload),
            'evoke_response_rate_when_needed': (
                float(successful_evokes_on_overload / overload_with_parked_capacity)
                if overload_with_parked_capacity > 0 else None
            ),
            'park_response_rate_when_needed': (
                float(successful_parks_on_underload / underload_with_parking_capacity)
                if underload_with_parking_capacity > 0 else None
            ),
            'oscillation_events': int(oscillation_events),
            'oscillation_fraction': float(oscillation_events / total_steps) if total_steps else 0.0,
            
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
        print(f"Reward logic: {results['reward_logic']}")
        print(f"Invalid action masking: {'ON' if results['mask_invalid_actions'] else 'OFF'}")
        print(f"\nPerformance:")
        print(f"  Avg Reward: {results['avg_reward']:.2f} (±{results['std_reward']:.2f})")
        print(f"  Avg Latency: {results['avg_latency']:.2f} ms")
        print(f"  Worst-case Latency: {results['cs_worst_latency']:.2f} ms")
        print(f"  Avg Energy: {results['avg_energy']:.2f} W")
        print(f"  Load Variance: {results['avg_load_variance']:.4f}")
        print(f"  Load Balance Index: {results['load_balance_index']:.4f}")
        
        print(f"\nProactive Behavior:")
        print(f"  Total Parking: {results['total_parking']}")
        print(f"  Total Evoking: {results['total_evoking']}")
        print(f"  Avg Park/Episode: {results['avg_parking_per_ep']:.1f}")
        print(f"  Avg Evoke/Episode: {results['avg_evoking_per_ep']:.1f}")
        pe_ratio = results['pe_ratio']
        print(f"  P/E Ratio: {pe_ratio:.2f}" if pe_ratio is not None else "  P/E Ratio: N/A (no evoke events)")
        print(f"  Overload steps: {results['overload_step_fraction']*100:.1f}%")
        print(f"  Underload steps: {results['underload_step_fraction']*100:.1f}%")
        evoke_rr = results['evoke_response_rate_when_needed']
        park_rr = results['park_response_rate_when_needed']
        print(
            f"  Evoke response (when needed): {evoke_rr*100:.1f}%"
            if evoke_rr is not None else
            "  Evoke response (when needed): N/A (no overload+parked-capacity demand)"
        )
        print(
            f"  Park response (when needed): {park_rr*100:.1f}%"
            if park_rr is not None else
            "  Park response (when needed): N/A (no underload+parking-capacity demand)"
        )

        print(f"\nDebug:")
        print(f"  Action counts: {results['action_type_counts']}")
        print(f"  Successful actions: {results['successful_action_counts']}")
        print(f"  Failed actions: {results['failed_action_counts']}")
        print(f"  Raw action IDs: {results['raw_action_id_counts']}")
        print(f"  Executed action IDs: {results['executed_action_id_counts']}")
        print(
            f"  Mask overrides: {results['masked_action_override_count']} "
            f"({results['masked_action_override_fraction']*100:.1f}%)"
        )
        print(
            f"  Park/Evoke oscillations: {results['oscillation_events']} "
            f"({results['oscillation_fraction']*100:.1f}%)"
        )
        if results['action_id_legend']:
            legend_preview = ", ".join(
                f"{action_id}={label}"
                for action_id, label in sorted(results['action_id_legend'].items(), key=lambda item: int(item[0]))[:8]
            )
            print(f"  Action legend: {legend_preview}")
        
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


def test_all_models(mask_invalid_actions=False):
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
        model_pattern = f'../../models/LATEST_rainbow_proactive_{topo}.pth'
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            model_pattern = f'../../models/rainbow_proactive_{topo}_*.pth'
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
            tester = ModelTester(
                model_path,
                traffic_path,
                topo,
                mask_invalid_actions=mask_invalid_actions
            )
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
    parser.add_argument(
        '--mask-invalid-actions',
        action='store_true',
        help='Mask impossible actions during evaluation and choose the best valid action instead'
    )
    
    args = parser.parse_args()
    
    if args.test_all:
        test_all_models(mask_invalid_actions=args.mask_invalid_actions)
    elif args.model and args.traffic and args.topology:
        tester = ModelTester(
            args.model,
            args.traffic,
            args.topology,
            mask_invalid_actions=args.mask_invalid_actions
        )
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
