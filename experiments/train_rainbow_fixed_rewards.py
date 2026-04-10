#!/usr/bin/env python3
"""
Rainbow DQN Training - CONTEXT-AWARE PROACTIVE REWARDS

FIXES:
- Context-aware proactive bonuses (Park/Evoke only when needed)
- Strong penalties for inaction when corrective action is needed
- All JSON serialization issues fixed
- get_topology_size() function included
"""

import sys
import os
import torch
import numpy as np
import time
from datetime import datetime
from collections import deque

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def train_rainbow_fixed_rewards(
    topology='gridnet',
    timesteps=20000,
    mode='proactive',
    train_freq=4,
    convergence_window=3,
    eval_freq=2000,
    mask_invalid_actions=False
):
    """Train Rainbow DQN with context-aware proactive rewards."""
    
    topology = topology.strip().lower()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"🚀 RAINBOW DQN TRAINING - CONTEXT-AWARE PROACTIVE REWARDS")
    print(f"{'='*70}\n")
    
    print(f"Configuration:")
    print(f"  Topology: {topology} ({get_topology_size(topology)} nodes)")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Mode: {mode}")
    print(f"  Mask invalid actions: {'ON' if mask_invalid_actions else 'OFF'}")
    print(f"  ⚡ CONTEXT-AWARE: Park/Evoke bonuses only when needed")
    print(f"  ⚡ PENALTIES: NoAction when corrective action is needed")
    print()
    
    # Import
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments')
    sys.path.insert(0, env_path)
    
    from threshold_proactive_sdn_env import ThresholdBasedProactiveSDN
    from rainbow_dqn_model import RainbowDQN
    
    # Create environment
    env = ThresholdBasedProactiveSDN(
        topology_name=topology,
        num_slave_controllers=3,
        num_parked_controllers=2
    )
    
    # ========================================================================
    # CONTEXT-AWARE REWARD FUNCTION
    # ========================================================================
    original_calculate_reward = env._calculate_reward
    
    def fixed_calculate_reward(action_type: str, success: bool):
        """Context-aware reward to avoid unnecessary park/evoke actions."""
        # Calculate base metrics
        latency = env._calculate_latency()
        worst_case_latency = env._calculate_worst_case_latency()
        energy = env._calculate_energy()
        load_variance = env._calculate_load_variance()
        load_balance_index = env._calculate_load_balance_index()
        
        # Base reward (lighter scale)
        norm_latency = latency / 20.0
        norm_energy = energy / 1000.0
        norm_load_var = load_variance / 1.0
        
        reward = -1.0 * norm_latency - 2.0 * norm_energy - 1.0 * norm_load_var
        
        thresholds = env._check_thresholds()
        has_overload = bool(thresholds['overloaded'])
        has_underload = bool(thresholds['underloaded'])
        can_still_park = len(env.parked_slaves) < env.num_parked_controllers

        if action_type == 'park' and success:
            # Reward parking only when there is true underload and spare parking capacity.
            reward += 200.0 if (has_underload and can_still_park) else -20.0
        elif action_type == 'evoke' and success:
            # Reward evoke only as overload relief; discourage needless waking.
            reward += 200.0 if has_overload else -30.0
        elif action_type == 'migrate' and success:
            reward += 2.0
        
        if action_type == 'noop':
            # Heavy penalty for ignoring overload
            if has_overload:
                reward -= 100.0
            
            # Penalty for not parking when there is underload and parking capacity.
            if has_underload and can_still_park:
                reward -= 80.0
        
        # Failure penalty
        if not success and action_type != 'noop':
            reward -= 10.0
        
        # Info dict
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
    
    # Override the reward calculation
    env._calculate_reward = fixed_calculate_reward
    
    # ========================================================================
    # MODEL SETUP
    # ========================================================================
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Topology-specific hyperparameters
    if topology in ['interoute', 'cogentco']:
        hidden_dim = 512
        lr = 5e-5
        n_step = 5
        buffer_size = 100000
        batch_size = 256
    elif topology in ['bellcanada', 'os3e']:
        hidden_dim = 256
        lr = 1e-4
        n_step = 3
        buffer_size = 50000
        batch_size = 128
    else:
        hidden_dim = 128
        lr = 2e-4
        n_step = 3
        buffer_size = 30000
        batch_size = 64
    
    agent = RainbowDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=0.99,
        n_step=n_step,
        buffer_size=buffer_size,
        batch_size=batch_size,
        device=device
    )

    tensorboard_writer = None
    tensorboard_run_dir = os.path.join(
        "logs",
        "proactive_dqn_tensorboard",
        f"rainbow_{mode}_{topology}_{run_timestamp}"
    )
    if SummaryWriter is not None:
        tensorboard_writer = SummaryWriter(log_dir=tensorboard_run_dir)
        tensorboard_writer.add_text(
            "run/config",
            (
                f"topology={topology}, timesteps={timesteps}, mode={mode}, device={device}, "
                f"hidden_dim={hidden_dim}, lr={lr}, n_step={n_step}, "
                f"buffer_size={buffer_size}, batch_size={batch_size}"
            ),
        )
        print(f"TensorBoard log dir: {tensorboard_run_dir}")
    else:
        print("TensorBoard unavailable: install `tensorboard>=1.15` to enable event logging.")
    
    # Exploration
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = timesteps * 0.5
    
    def get_epsilon(step):
        return epsilon_end + (epsilon_start - epsilon_end) * \
               np.exp(-1.0 * step / epsilon_decay)

    def select_masked_action(current_state):
        """Choose the best valid action from the current Q-values."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(agent.device)
            q_values = agent.online_net.get_q_values(state_tensor).squeeze(0)
            raw_action = int(q_values.argmax().item())
            valid_actions = env.get_valid_actions()
            masked_q_values = q_values.clone()
            invalid_actions = torch.ones_like(masked_q_values, dtype=torch.bool)
            invalid_actions[valid_actions] = False
            masked_q_values[invalid_actions] = -torch.inf
            chosen_action = int(masked_q_values.argmax().item())
            return raw_action, chosen_action, valid_actions
    
    # Tracking
    episode_rewards = []
    episode_latencies = []
    episode_worst_latencies = []
    episode_energies = []
    episode_load_variances = []
    episode_load_balance_indices = []
    eval_rewards = deque(maxlen=convergence_window)
    parking_events = 0
    evoking_events = 0
    migrate_events = 0
    no_action_count = 0
    invalid_action_overrides = 0
    losses = []
    recent_actions = deque(maxlen=200)
    
    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    current_episode_latencies = []
    current_episode_worst_latencies = []
    current_episode_energies = []
    current_episode_load_variances = []
    current_episode_load_balance_indices = []
    
    print(f"Starting training on {device}...\n")
    print("Progress: [", end='', flush=True)
    progress_interval = max(1, timesteps // 50)
    
    for step in range(timesteps):
        if step % progress_interval == 0:
            print('=', end='', flush=True)
        
        # Traffic variation
        if step % 100 == 0:
            env.traffic_phase = (env.traffic_phase + 1) % 24
            if hasattr(env, '_simulate_traffic_variation'):
                env._simulate_traffic_variation()
        
        # Epsilon-greedy
        epsilon = get_epsilon(step)
        
        if np.random.random() < epsilon:
            if mask_invalid_actions:
                valid_actions = env.get_valid_actions()
                action = int(np.random.choice(valid_actions))
            else:
                action = np.random.randint(action_dim)
        else:
            if mask_invalid_actions:
                raw_action, action, _ = select_masked_action(state)
                if raw_action != action:
                    invalid_action_overrides += 1
            else:
                action = agent.select_action(state, training=True)
        
        # Step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("train/step_reward", float(reward), step)
            tensorboard_writer.add_scalar("train/latency", float(info.get('latency', 0.0)), step)
            tensorboard_writer.add_scalar("train/energy", float(info.get('energy', 0.0)), step)
            tensorboard_writer.add_scalar(
                "train/load_variance", float(info.get('load_variance', 0.0)), step
            )
            tensorboard_writer.add_scalar(
                "train/overloaded_count", float(info.get('overloaded_count', 0.0)), step
            )
            tensorboard_writer.add_scalar(
                "train/underloaded_count", float(info.get('underloaded_count', 0.0)), step
            )
            tensorboard_writer.add_scalar(
                "train/active_controllers", float(info.get('active_controllers', 0.0)), step
            )
            tensorboard_writer.add_scalar(
                "train/parked_controllers", float(info.get('parked_controllers', 0.0)), step
            )
            if mask_invalid_actions:
                tensorboard_writer.add_scalar(
                    "train/invalid_action_overrides", float(invalid_action_overrides), step
                )
        
        # Track events
        action_type = info.get('action_type', '')
        recent_actions.append(action_type)
        current_episode_latencies.append(float(info.get('latency', 0.0)))
        current_episode_worst_latencies.append(float(info.get('worst_case_latency', 0.0)))
        current_episode_energies.append(float(info.get('energy', 0.0)))
        current_episode_load_variances.append(float(info.get('load_variance', 0.0)))
        current_episode_load_balance_indices.append(float(info.get('load_balance_index', 0.0)))
        
        if 'park' in action_type and info.get('action_success'):
            parking_events += 1
        elif 'evoke' in action_type and info.get('action_success'):
            evoking_events += 1
        elif 'migrate' in action_type and info.get('action_success'):
            migrate_events += 1
        elif action_type == 'noop':
            no_action_count += 1
        
        # Store transition
        agent.push_n_step(state, action, reward, next_state, done)
        
        # Train
        adaptive_train_freq = max(1, train_freq - (step // 10000))
        
        if step > 1000 and step % adaptive_train_freq == 0 and len(agent.memory) >= agent.batch_size:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("train/loss", float(loss), step)
        
        # Update target
        if step % 1000 == 0 and step > 0:
            agent.update_target_network()
        
        episode_reward += reward
        state = next_state
        
        # Episode end
        if done:
            episode_rewards.append(episode_reward)
            episode_latencies.append(
                float(np.mean(current_episode_latencies)) if current_episode_latencies else 0.0
            )
            episode_energies.append(
                float(np.mean(current_episode_energies)) if current_episode_energies else 0.0
            )
            episode_load_variances.append(
                float(np.mean(current_episode_load_variances))
                if current_episode_load_variances else 0.0
            )
            episode_worst_latencies.append(
                float(np.max(current_episode_worst_latencies))
                if current_episode_worst_latencies else 0.0
            )
            episode_load_balance_indices.append(
                float(np.mean(current_episode_load_balance_indices))
                if current_episode_load_balance_indices else 0.0
            )
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("episode/reward", float(episode_reward), episode_count)
                tensorboard_writer.add_scalar(
                    "episode/avg_latency", episode_latencies[-1], episode_count
                )
                tensorboard_writer.add_scalar(
                    "episode/worst_latency", episode_worst_latencies[-1], episode_count
                )
                tensorboard_writer.add_scalar(
                    "episode/avg_energy", episode_energies[-1], episode_count
                )
                tensorboard_writer.add_scalar(
                    "episode/avg_load_variance", episode_load_variances[-1], episode_count
                )
                tensorboard_writer.add_scalar(
                    "episode/load_balance_index", episode_load_balance_indices[-1], episode_count
                )
            episode_count += 1
            state, _ = env.reset()
            episode_reward = 0
            current_episode_latencies = []
            current_episode_worst_latencies = []
            current_episode_energies = []
            current_episode_load_variances = []
            current_episode_load_balance_indices = []
        
        # Evaluation
        if step % eval_freq == 0 and step > 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards) if episode_rewards else 0
            eval_rewards.append(avg_reward)
            recent_avg_latency = float(np.mean(episode_latencies[-100:])) if episode_latencies else 0.0
            recent_avg_energy = float(np.mean(episode_energies[-100:])) if episode_energies else 0.0
            recent_avg_load_var = (
                float(np.mean(episode_load_variances[-100:])) if episode_load_variances else 0.0
            )
            
            print(f"\n\n{'='*70}")
            print(f"📊 Evaluation at Step {step:,} / {timesteps:,}")
            print(f"{'='*70}")
            print(f"Episodes: {episode_count} | Avg Reward: {avg_reward:.2f}")
            print(f"Epsilon: {epsilon:.3f}")
            if episode_latencies:
                print(
                    f"Avg Latency: {recent_avg_latency:.2f} ms | "
                    f"Avg Energy: {recent_avg_energy:.2f} W | "
                    f"Avg Load Var: {recent_avg_load_var:.4f}"
                )
            print(f"Actions - Park: {parking_events} | Evoke: {evoking_events} | Migrate: {migrate_events} | NoAction: {no_action_count}")
            if mask_invalid_actions:
                print(f"Masked invalid greedy choices: {invalid_action_overrides}")
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("eval/avg_reward_last100", float(avg_reward), step)
                tensorboard_writer.add_scalar("eval/avg_latency_last100", recent_avg_latency, step)
                tensorboard_writer.add_scalar("eval/avg_energy_last100", recent_avg_energy, step)
                tensorboard_writer.add_scalar(
                    "eval/avg_load_variance_last100", recent_avg_load_var, step
                )
                tensorboard_writer.add_scalar("eval/epsilon", float(epsilon), step)
                tensorboard_writer.add_scalar("actions/parking_events", float(parking_events), step)
                tensorboard_writer.add_scalar("actions/evoking_events", float(evoking_events), step)
                tensorboard_writer.add_scalar("actions/migrate_events", float(migrate_events), step)
                tensorboard_writer.add_scalar("actions/no_action_count", float(no_action_count), step)
                if mask_invalid_actions:
                    tensorboard_writer.add_scalar(
                        "actions/invalid_action_overrides", float(invalid_action_overrides), step
                    )
            
            if len(losses) > 0:
                print(f"Avg Loss: {np.mean(losses[-1000:]):.4f}")
            
            # Convergence check
            if len(eval_rewards) == convergence_window:
                reward_std = np.std(eval_rewards)
                reward_mean = np.mean(eval_rewards)
                
                if reward_std < 50.0 and reward_mean > -200:
                    print(f"\n🎉 CONVERGED! Reward std: {reward_std:.2f} < 50.0")
                    print(f"Mean reward: {reward_mean:.2f} > -200")
                    print(f"Stopping early at step {step}")
                    break
            
            print(f"{'='*70}\n")
            print("Progress: [", end='', flush=True)
    
    print("]\n")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    
    timestamp = run_timestamp
    model_path = f'models/rainbow_{mode}_{topology}_{timesteps}steps_{timestamp}.pth'
    
    # Save with dimensions
    torch.save({
        'online_net': agent.online_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'state_dim': state_dim,
        'action_dim': action_dim
    }, model_path)
    
    # Metadata with proper type conversions
    metadata_path = model_path.replace('.pth', '_metadata.json')
    import json
    
    final_avg_reward = float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 \
                      else float(np.mean(episode_rewards)) if len(episode_rewards) > 0 \
                      else 0.0
    final_avg_latency = float(np.mean(episode_latencies)) if episode_latencies else 0.0
    final_avg_cs_latency = final_avg_latency
    final_worst_latency = float(np.mean(episode_worst_latencies)) if episode_worst_latencies else 0.0
    final_std_latency = float(np.std(episode_latencies)) if episode_latencies else 0.0
    final_avg_energy = float(np.mean(episode_energies)) if episode_energies else 0.0
    final_std_energy = float(np.std(episode_energies)) if episode_energies else 0.0
    final_avg_load_variance = (
        float(np.mean(episode_load_variances)) if episode_load_variances else 0.0
    )
    final_std_load_variance = (
        float(np.std(episode_load_variances)) if episode_load_variances else 0.0
    )
    training_time_seconds = float(time.time() - training_start_time)
    final_avg_load_balance_index = (
        float(np.mean(episode_load_balance_indices)) if episode_load_balance_indices else 0.0
    )
    
    metadata = {
        'topology': str(topology),
        'topology_size': int(get_topology_size(topology)) if get_topology_size(topology) != 'unknown' else 'unknown',
        'timesteps': int(timesteps),
        'mode': str(mode),
        'timestamp': datetime.now().isoformat(),
        'reward_function': (
            'CONTEXT_AWARE (Park=+200 if needed else -20, '
            'Evoke=+200 if overloaded else -30, NoAction penalties on unmet overload/underload)'
        ),
        'mask_invalid_actions': bool(mask_invalid_actions),
        
        # Reward metrics (ALL converted to float)
        'final_avg_reward_last100': float(final_avg_reward),
        'final_avg_reward_all': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'total_cumulative_reward': float(np.sum(episode_rewards)) if episode_rewards else 0.0,
        'best_episode_reward': float(np.max(episode_rewards)) if episode_rewards else 0.0,
        'worst_episode_reward': float(np.min(episode_rewards)) if episode_rewards else 0.0,
        'avg_latency': final_avg_latency,
        'cs_avg_latency': final_avg_cs_latency,
        'cs_worst_latency': final_worst_latency,
        'std_latency': final_std_latency,
        'avg_energy': final_avg_energy,
        'std_energy': final_std_energy,
        'avg_load_variance': final_avg_load_variance,
        'std_load_variance': final_std_load_variance,
        'load_balance_index': final_avg_load_balance_index,
        
        # Events (ALL converted to int)
        'parking_events': int(parking_events),
        'evoking_events': int(evoking_events),
        'migrate_events': int(migrate_events),
        'no_action_count': int(no_action_count),
        'invalid_action_overrides': int(invalid_action_overrides),
        'pe_ratio': float(parking_events / evoking_events) if evoking_events > 0 else None,
        
        # Training stats
        'total_episodes': int(episode_count),
        'avg_loss': float(np.mean(losses)) if losses else 0.0,
        'final_epsilon': float(get_epsilon(timesteps)),
        'training_time_seconds': training_time_seconds,
        
        # Model architecture
        'state_dim': int(state_dim),
        'action_dim': int(action_dim),
        
        # Hyperparameters (ALL converted)
        'hyperparameters': {
            'learning_rate': float(lr),
            'gamma': 0.99,
            'n_step': int(n_step),
            'batch_size': int(batch_size),
            'buffer_size': int(buffer_size),
            'hidden_dim': int(hidden_dim),
            'epsilon_start': float(epsilon_start),
            'epsilon_end': float(epsilon_end)
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"\n📊 Final Results:")
    print(f"  Episodes: {episode_count}")
    print(f"  Avg Reward (last 100): {final_avg_reward:.2f}")
    print(f"  Avg Reward (all): {metadata['final_avg_reward_all']:.2f}")
    print(f"  Total Reward: {metadata['total_cumulative_reward']:.2f}")
    print(f"  Avg Latency: {final_avg_latency:.2f} ms")
    print(f"  Avg Energy: {final_avg_energy:.2f} W")
    print(f"  Avg Load Variance: {final_avg_load_variance:.4f}")
    print(f"\n🎬 Action Summary:")
    print(f"  Parking: {parking_events} | Evoking: {evoking_events}")
    print(f"  Migrate: {migrate_events} | NoAction: {no_action_count}")
    if mask_invalid_actions:
        print(f"  Invalid greedy overrides: {invalid_action_overrides}")
    if evoking_events > 0:
        print(f"  P/E Ratio: {parking_events / evoking_events:.2f}")
    print(f"{'='*70}\n")
    
    # Save as latest
    latest_path = f'models/LATEST_rainbow_{mode}_{topology}.pth'
    torch.save({
        'online_net': agent.online_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'state_dim': state_dim,
        'action_dim': action_dim
    }, latest_path)
    print(f"💡 Latest model also saved as: {latest_path}\n")
    if tensorboard_writer is not None:
        tensorboard_writer.add_hparams(
            {
                'hidden_dim': hidden_dim,
                'learning_rate': lr,
                'n_step': n_step,
                'batch_size': batch_size,
                'buffer_size': buffer_size,
                'timesteps': timesteps,
                'mask_invalid_actions': int(mask_invalid_actions),
            },
            {
                'hparam/final_avg_reward': final_avg_reward,
                'hparam/avg_latency': final_avg_latency,
                'hparam/avg_energy': final_avg_energy,
                'hparam/avg_load_variance': final_avg_load_variance,
                'hparam/parking_events': float(parking_events),
                'hparam/evoking_events': float(evoking_events),
                'hparam/invalid_action_overrides': float(invalid_action_overrides),
            }
        )
        tensorboard_writer.close()

    env.close()
    return agent, model_path


def get_topology_size(topology):
    """Get number of nodes in topology"""
    sizes = {
        'gridnet': 9,
        'bellcanada': 48,
        'os3e': 34,
        'interoute': 110,
        'cogentco': 197
    }
    return sizes.get(topology, 'unknown')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rainbow DQN Training - Aggressive Proactive Rewards')
    parser.add_argument('--topology', default='os3e',
                       choices=['gridnet', 'bellcanada', 'os3e', 'interoute', 'cogentco'])
    parser.add_argument('--timesteps', type=int, default=20000)
    parser.add_argument('--mode', default='proactive',
                       choices=['proactive', 'reactive'])
    parser.add_argument('--train-freq', type=int, default=4)
    parser.add_argument(
        '--mask-invalid-actions',
        action='store_true',
        help='During training, sample only valid actions and mask invalid greedy choices'
    )
    
    args = parser.parse_args()
    
    agent, model_path = train_rainbow_fixed_rewards(
        topology=args.topology,
        timesteps=args.timesteps,
        mode=args.mode,
        train_freq=args.train_freq,
        mask_invalid_actions=args.mask_invalid_actions
    )
    
    print(f"\n✅ Training complete with CONTEXT-AWARE proactive rewards!")
    print(f"\nTo train all topologies, run:")
    print(f"  for topo in gridnet bellcanada os3e interoute cogentco; do")
    print(f"    python train_rainbow_fixed.py --topology $topo --timesteps 20000")
    print(f"  done\n")
