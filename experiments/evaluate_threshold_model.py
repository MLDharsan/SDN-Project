#!/usr/bin/env python3
"""
Evaluate Rainbow checkpoints with the same environment family used in training/testing.

Focus:
1. Paper-style metrics: controller-switch average latency, worst-case latency,
   load-balance index, and training time when metadata is available.
2. Consistent model loading for custom Rainbow `.pth` checkpoints.
3. Optional valid-action masking to match the improved evaluation path.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.rainbow_dqn_model import RainbowDQN
from environments.threshold_proactive_sdn_env import ThresholdBasedProactiveSDN


PAPER_RESULTS = {
    'gridnet': {'cs_avg_latency': 3.34, 'cs_worst_latency': 8.71, 'load_balance_index': 2.68},
    'bellcanada': {'cs_avg_latency': 5.01, 'cs_worst_latency': 7.38, 'load_balance_index': 4.00},
    'os3e': {'cs_avg_latency': 2.91, 'cs_worst_latency': 8.65, 'load_balance_index': 3.20},
    'interoute': {'cs_avg_latency': 3.45, 'cs_worst_latency': 3.45, 'load_balance_index': 2.98},
    'cogentco': {'cs_avg_latency': 7.88, 'cs_worst_latency': 16.37, 'load_balance_index': 6.45},
}

PAPER_RESULTS_STATUS = "UNVERIFIED_MANUAL_ENTRY"


class RealTrafficEvaluationEnvironment(ThresholdBasedProactiveSDN):
    """Use processed traffic traces instead of synthetic day/night sampling."""

    def __init__(self, traffic_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_traffic_data = traffic_data
        self.traffic_index = 0
        self.max_traffic_steps = len(traffic_data)

    def _simulate_traffic_variation(self):
        if self.traffic_index >= self.max_traffic_steps:
            self.traffic_index = 0

        current_traffic = self.real_traffic_data[self.traffic_index]
        self.traffic_index += 1
        self.controller_loads = {i: 0.0 for i in range(self.total_controllers)}

        for switch_id in range(self.num_switches):
            controller_id = self.switch_to_controller[switch_id]
            if controller_id in self.parked_slaves:
                continue

            if switch_id < len(current_traffic):
                switch_load = current_traffic[switch_id]
            else:
                switch_load = current_traffic[switch_id % len(current_traffic)]

            self.controller_loads[controller_id] += switch_load

        for controller_id in self.active_slaves:
            num_assigned = sum(
                1 for switch, assigned_controller in self.switch_to_controller.items()
                if assigned_controller == controller_id
            )
            if num_assigned > 0:
                self.controller_loads[controller_id] /= num_assigned
                self.controller_loads[controller_id] = np.clip(self.controller_loads[controller_id], 0, 1)


def apply_training_reward_logic(env):
    """Match the aggressive reward shaping used in training."""
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

        if action_type == 'park' and success:
            reward += 200.0
        elif action_type == 'evoke' and success:
            reward += 150.0
        elif action_type == 'migrate' and success:
            reward += 2.0

        thresholds = env._check_thresholds()
        if action_type == 'noop':
            if thresholds['overloaded']:
                reward -= 100.0
            if thresholds['underloaded'] and len(env.parked_slaves) < env.num_parked_controllers:
                reward -= 80.0

        if not success and action_type != 'noop':
            reward -= 10.0

        return reward, {
            'latency': latency,
            'cs_avg_latency': latency,
            'worst_case_latency': worst_case_latency,
            'cs_worst_latency': worst_case_latency,
            'energy': energy,
            'load_variance': load_variance,
            'load_balance_index': load_balance_index,
            'active_controllers': len(env.active_slaves),
            'parked_controllers': len(env.parked_slaves),
            'action_type': action_type,
            'action_success': success,
            'overloaded_count': len(thresholds['overloaded']),
            'underloaded_count': len(thresholds['underloaded']),
        }

    env._calculate_reward = fixed_calculate_reward


def load_rainbow_model(model_path: str):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        state_dim = checkpoint.get('state_dim')
        action_dim = checkpoint.get('action_dim')
        hidden_dim = None

        if 'hyperparameters' in checkpoint:
            hidden_dim = checkpoint['hyperparameters'].get('hidden_dim')

        if hidden_dim is None and 'online_net' in checkpoint and 'feature.0.weight' in checkpoint['online_net']:
            hidden_dim = checkpoint['online_net']['feature.0.weight'].shape[0]

        if state_dim is None or action_dim is None:
            if 'online_net' in checkpoint and 'feature.0.weight' in checkpoint['online_net']:
                state_dim = checkpoint['online_net']['feature.0.weight'].shape[1]
                hidden_dim = checkpoint['online_net']['feature.0.weight'].shape[0]
            adv_keys = [
                key for key in checkpoint.get('online_net', {})
                if 'advantage_stream.2' in key and 'weight_mu' in key
            ]
            if action_dim is None and adv_keys:
                action_dim = checkpoint['online_net'][adv_keys[0]].shape[0] // 51
    else:
        raise ValueError("Unsupported checkpoint format")

    if hidden_dim is None:
        hidden_dim = 256

    model = RainbowDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=1e-4,
        gamma=0.99,
        n_step=3,
        device='cpu',
    )
    model.online_net.load_state_dict(checkpoint['online_net'])
    model.online_net.eval()
    return model, checkpoint


def select_action(model, state, env, mask_invalid_actions: bool):
    model.online_net.train(False)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model.device)
        q_values = model.online_net.get_q_values(state_tensor).squeeze(0)
        raw_action = int(q_values.argmax().item())

        if not mask_invalid_actions:
            return raw_action, raw_action

        valid_actions = env.get_valid_actions()
        masked_q_values = q_values.clone()
        invalid_actions = torch.ones_like(masked_q_values, dtype=torch.bool)
        invalid_actions[valid_actions] = False
        masked_q_values[invalid_actions] = -torch.inf
        chosen_action = int(masked_q_values.argmax().item())
        return raw_action, chosen_action


def evaluate_model(
    model_path: str,
    topology: str = 'gridnet',
    episodes: int = 10,
    steps_per_episode: int = 500,
    mask_invalid_actions: bool = False,
    traffic_path: str | None = None,
):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model, checkpoint = load_rainbow_model(str(model_path))
    metadata_path = model_path.with_name(model_path.stem.replace('LATEST_', '') + '_metadata.json')
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    traffic_data = None
    if traffic_path:
        traffic_path = Path(traffic_path)
        if traffic_path.exists():
            traffic_data = np.load(traffic_path)

    if traffic_data is not None:
        env = RealTrafficEvaluationEnvironment(
            traffic_data=traffic_data,
            topology_name=topology,
            num_slave_controllers=3,
            num_parked_controllers=2,
        )
    else:
        env = ThresholdBasedProactiveSDN(
            topology_name=topology,
            num_slave_controllers=3,
            num_parked_controllers=2,
        )

    apply_training_reward_logic(env)

    metrics = {
        'cs_avg_latency': [],
        'cs_worst_latency': [],
        'load_balance_index': [],
        'energy': [],
        'reward': [],
        'parking_events': [],
        'evoking_events': [],
        'migration_events': [],
        'mask_overrides': 0,
    }

    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_avg_latencies = []
        ep_worst_latencies = []
        ep_load_balance = []
        ep_energy = []
        ep_park = 0
        ep_evoke = 0
        ep_migrate = 0

        for _step in range(steps_per_episode):
            raw_action, action = select_action(model, state, env, mask_invalid_actions)
            if raw_action != action:
                metrics['mask_overrides'] += 1

            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_avg_latencies.append(float(info.get('cs_avg_latency', info.get('latency', 0.0))))
            ep_worst_latencies.append(float(info.get('cs_worst_latency', info.get('worst_case_latency', 0.0))))
            ep_load_balance.append(float(info.get('load_balance_index', 0.0)))
            ep_energy.append(float(info.get('energy', 0.0)))

            action_type = info.get('action_type', '')
            if 'park' in action_type and info.get('action_success'):
                ep_park += 1
            elif 'evoke' in action_type and info.get('action_success'):
                ep_evoke += 1
            elif 'migrate' in action_type and info.get('action_success'):
                ep_migrate += 1

            state = next_state
            if terminated or truncated:
                break

        metrics['reward'].append(ep_reward)
        metrics['cs_avg_latency'].append(float(np.mean(ep_avg_latencies)) if ep_avg_latencies else 0.0)
        metrics['cs_worst_latency'].append(float(np.max(ep_worst_latencies)) if ep_worst_latencies else 0.0)
        metrics['load_balance_index'].append(float(np.mean(ep_load_balance)) if ep_load_balance else 0.0)
        metrics['energy'].append(float(np.mean(ep_energy)) if ep_energy else 0.0)
        metrics['parking_events'].append(ep_park)
        metrics['evoking_events'].append(ep_evoke)
        metrics['migration_events'].append(ep_migrate)

    env.close()

    summary = {
        'topology': topology,
        'model_path': str(model_path),
        'episodes': episodes,
        'steps_per_episode': steps_per_episode,
        'mask_invalid_actions': bool(mask_invalid_actions),
        'paper_results_status': PAPER_RESULTS_STATUS,
        'training_time_seconds': float(metadata.get('training_time_seconds', 0.0)) if metadata else 0.0,
        'cs_avg_latency': float(np.mean(metrics['cs_avg_latency'])),
        'cs_worst_latency': float(np.mean(metrics['cs_worst_latency'])),
        'load_balance_index': float(np.mean(metrics['load_balance_index'])),
        'avg_energy': float(np.mean(metrics['energy'])),
        'avg_reward': float(np.mean(metrics['reward'])),
        'avg_parking_events': float(np.mean(metrics['parking_events'])),
        'avg_evoking_events': float(np.mean(metrics['evoking_events'])),
        'avg_migration_events': float(np.mean(metrics['migration_events'])),
        'mask_overrides': int(metrics['mask_overrides']),
        'raw_metrics': metrics,
    }

    if topology in PAPER_RESULTS:
        paper = PAPER_RESULTS[topology]
        summary['paper_reference'] = paper
        summary['paper_deltas_pct'] = {
            'cs_avg_latency': float((summary['cs_avg_latency'] - paper['cs_avg_latency']) / paper['cs_avg_latency'] * 100),
            'cs_worst_latency': float((summary['cs_worst_latency'] - paper['cs_worst_latency']) / paper['cs_worst_latency'] * 100),
            'load_balance_index': float((summary['load_balance_index'] - paper['load_balance_index']) / paper['load_balance_index'] * 100),
        }

    return summary


def print_evaluation_report(summary):
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT: {summary['topology'].upper()}")
    print(f"{'='*70}\n")
    print(f"Model: {summary['model_path']}")
    print(f"Episodes: {summary['episodes']} | Steps/Episode: {summary['steps_per_episode']}")
    print(f"Invalid action masking: {'ON' if summary['mask_invalid_actions'] else 'OFF'}")
    print(f"Paper reference values: {summary['paper_results_status']}")
    print()
    print(f"CS average latency: {summary['cs_avg_latency']:.2f} ms")
    print(f"CS worst-case latency: {summary['cs_worst_latency']:.2f} ms")
    print(f"Load balance index: {summary['load_balance_index']:.4f}")
    print(f"Training time: {summary['training_time_seconds']:.1f} s")
    print(f"Average reward: {summary['avg_reward']:.2f}")
    print(f"Average energy: {summary['avg_energy']:.2f} W")
    print(f"Parking/Evoking/Migration per episode: "
          f"{summary['avg_parking_events']:.1f} / {summary['avg_evoking_events']:.1f} / {summary['avg_migration_events']:.1f}")
    print(f"Mask overrides: {summary['mask_overrides']}")

    if 'paper_reference' in summary:
        print("\nComparison against stored paper reference values:")
        print(f"  Paper CS average latency: {summary['paper_reference']['cs_avg_latency']:.2f} ms")
        print(f"  Paper CS worst latency: {summary['paper_reference']['cs_worst_latency']:.2f} ms")
        print(f"  Paper load balance index: {summary['paper_reference']['load_balance_index']:.2f}")
        print(f"  Delta avg latency: {summary['paper_deltas_pct']['cs_avg_latency']:+.1f}%")
        print(f"  Delta worst latency: {summary['paper_deltas_pct']['cs_worst_latency']:+.1f}%")
        print(f"  Delta load balance: {summary['paper_deltas_pct']['load_balance_index']:+.1f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Rainbow checkpoint with paper-style metrics')
    parser.add_argument('--topology', default='gridnet',
                        choices=['gridnet', 'bellcanada', 'os3e', 'interoute', 'cogentco'])
    parser.add_argument('--model', help='Path to Rainbow checkpoint (.pth)')
    parser.add_argument('--traffic', help='Optional real traffic .npy path')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--steps-per-episode', type=int, default=500)
    parser.add_argument('--mask-invalid-actions', action='store_true')

    args = parser.parse_args()

    model_path = args.model or f'../models/LATEST_rainbow_proactive_{args.topology}.pth'
    traffic_path = args.traffic or f'../data/traffic/processed/{args.topology}/{args.topology}_synthetic_traffic.npy'

    summary = evaluate_model(
        model_path=model_path,
        topology=args.topology,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        mask_invalid_actions=args.mask_invalid_actions,
        traffic_path=traffic_path,
    )
    print_evaluation_report(summary)

    results_dir = Path('../results/evaluations')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"eval_{args.topology}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_clean = {key: value for key, value in summary.items() if key != 'raw_metrics'}
    results_file.write_text(json.dumps(summary_clean, indent=2))
    print(f"\nSaved: {results_file}")


if __name__ == '__main__':
    main()
