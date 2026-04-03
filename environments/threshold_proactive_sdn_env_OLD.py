	
#!/usr/bin/env python3
"""
Rainbow DQN Training - FIXED REWARD FUNCTION (Proactive Behavior)

ONLY CHANGE: Reward function updated to encourage proactive behavior
- Smaller bonuses (50/30 instead of 500/300)
- Scale base rewards 10x
- Penalties for inaction
- Everything else unchanged
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime
from collections import deque


def train_rainbow_fixed_rewards(
    topology='gridnet',
    timesteps=20000,
    mode='proactive',
    train_freq=4,
    convergence_window=3,
    eval_freq=2000
):
    """
    Train Rainbow DQN with FIXED reward function
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 RAINBOW DQN TRAINING - FIXED REWARDS")
    print(f"{'='*70}\n")
    
    print(f"Configuration:")
    print(f"  Topology: {topology} ({get_topology_size(topology)} nodes)")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Mode: {mode}")
    print(f"  ⚡ AGGRESSIVE: Proactive bonuses (Park=+200, Evoke=+150)")
    print(f"  ⚡ PENALTIES: NoAction when needed (-100/-80)")
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
    # 🔥 FIXED REWARD FUNCTION - ONLY THIS SECTION CHANGED
    # ========================================================================
    original_step = env.step
    original_calculate_reward = env._calculate_reward
    
    def fixed_calculate_reward(action_type: str, success: bool):
        """
        FIXED reward function with proper scaling
        """
        # Calculate base metrics (unchanged)
        latency = env._calculate_latency()
        energy = env._calculate_energy()
        load_variance = env._calculate_load_variance()
        
        # 🔥 FIX 1: Scale base rewards 10x (from -0.3/-0.5/-0.2 to -3/-5/-2)
        norm_latency = latency / 20.0
        norm_energy = energy / 1000.0
        norm_load_var = load_variance / 1.0
        
        reward = -1.0 * norm_latency - 2.0 * norm_energy - 1.0 * norm_load_var
        
        # 🔥 AGGRESSIVE BONUSES for proactive actions
        if action_type == 'park' and success:
            reward += 200.0   # BIG bonus to encourage parking
        elif action_type == 'evoke' and success:
            reward += 150.0   # BIG bonus to encourage evoking
        elif action_type == 'migrate' and success:
            reward += 2.0     # Small bonus
        
        # 🔥 STRONG PENALTIES for inaction when should act
        thresholds = env._check_thresholds()
        
        if action_type == 'noop':
            # Heavy penalty for ignoring overload
            if thresholds['overloaded']:
                reward -= 100.0
            
            # Heavy penalty for not parking when possible
            if thresholds['underloaded'] and len(env.parked_slaves) < env.num_parked_controllers:
                reward -= 80.0
        
        # Failure penalty (unchanged)
        if not success and action_type != 'noop':
            reward -= 10.0
        
        # Info dict (unchanged)
        info = {
            'latency': latency,
            'energy': energy,
            'load_variance': load_variance,
            'load_balance': np.sqrt(load_variance),
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
    # END OF REWARD FUNCTION CHANGES
    # ========================================================================
    
    # Everything else unchanged from original script
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
    
    # Exploration
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = timesteps * 0.5
    
    def get_epsilon(step):
        return epsilon_end + (epsilon_start - epsilon_end) * \
               np.exp(-1.0 * step / epsilon_decay)
    
    # Tracking
    episode_rewards = []
    eval_rewards = deque(maxlen=convergence_window)
    parking_events = 0
    evoking_events = 0
    migrate_events = 0
    no_action_count = 0
    losses = []
    recent_actions = deque(maxlen=200)
    
    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    
    print(f"Starting training on {device}...\n")
    print("Progress: [", end='', flush=True)
    
    for step in range(timesteps):
        if step % (timesteps // 50) == 0:
            print('=', end='', flush=True)
        
        # Traffic variation
        if step % 100 == 0:
            env.traffic_phase = (env.traffic_phase + 1) % 24
            if hasattr(env, '_simulate_traffic_variation'):
                env._simulate_traffic_variation()
        
        # Epsilon-greedy
        epsilon = get_epsilon(step)
        
        if np.random.random() < epsilon:
            action = np.random.randint(action_dim)
        else:
            action = agent.select_action(state, training=True)
        
        # Step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track events
        action_type = info.get('action_type', '')
        recent_actions.append(action_type)
        
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
        
        # Update target
        if step % 1000 == 0 and step > 0:
            agent.update_target_network()
        
        episode_reward += reward
        state = next_state
        
        # Episode end
        if done:
            episode_rewards.append(episode_reward)
            episode_count += 1
            state, _ = env.reset()
            episode_reward = 0
        
        # Evaluation
        if step % eval_freq == 0 and step > 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards) if episode_rewards else 0
            eval_rewards.append(avg_reward)
            
            print(f"\n\n{'='*70}")
            print(f"📊 Evaluation at Step {step:,} / {timesteps:,}")
            print(f"{'='*70}")
            print(f"Episodes: {episode_count} | Avg Reward: {avg_reward:.2f}")
            print(f"Epsilon: {epsilon:.3f}")
            print(f"Actions - Park: {parking_events} | Evoke: {evoking_events} | Migrate: {migrate_events} | NoAction: {no_action_count}")
            
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/rainbow_{mode}_{topology}_{timesteps}steps_{timestamp}.pth'
    
    # Save with dimensions
    torch.save({
        'online_net': agent.online_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'state_dim': state_dim,  # Save dimensions!
        'action_dim': action_dim
    }, model_path)
    
    # Metadata
    metadata_path = model_path.replace('.pth', '_metadata.json')
    import json
    
    final_avg_reward = float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 \
                      else float(np.mean(episode_rewards)) if len(episode_rewards) > 0 \
                      else 0.0
    
    metadata = {
        'topology': str(topology),
        'topology_size': int(get_topology_size(topology)) if get_topology_size(topology) != 'unknown' else 'unknown',
        'timesteps': int(timesteps),
        'mode': str(mode),
        'timestamp': datetime.now().isoformat(),
        'reward_function': 'AGGRESSIVE (Park=+200, Evoke=+150, NoAction=-100)',
        
        # Reward metrics (ALL converted to float)
        'final_avg_reward_last100': float(final_avg_reward),
        'final_avg_reward_all': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'total_cumulative_reward': float(np.sum(episode_rewards)) if episode_rewards else 0.0,
        'best_episode_reward': float(np.max(episode_rewards)) if episode_rewards else 0.0,
        'worst_episode_reward': float(np.min(episode_rewards)) if episode_rewards else 0.0,
        
        # Events (ALL converted to int)
        'parking_events': int(parking_events),
        'evoking_events': int(evoking_events),
        'migrate_events': int(migrate_events),
        'no_action_count': int(no_action_count),
        'pe_ratio': float(parking_events / evoking_events) if evoking_events > 0 else None,
        
        # Training stats
        'total_episodes': int(episode_count),
        'avg_loss': float(np.mean(losses)) if losses else 0.0,
        'final_epsilon': float(get_epsilon(timesteps)),
        
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
    print(f"\n🎬 Action Summary:")
    print(f"  Parking: {parking_events} | Evoking: {evoking_events}")
    print(f"  Migrate: {migrate_events} | NoAction: {no_action_count}")
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
    
    parser = argparse.ArgumentParser(description='Rainbow DQN Training - Fixed Rewards')
    parser.add_argument('--topology', default='os3e',
                       choices=['gridnet', 'bellcanada', 'os3e', 'interoute', 'cogentco'])
    parser.add_argument('--timesteps', type=int, default=20000)
    parser.add_argument('--mode', default='proactive',
                       choices=['proactive', 'reactive'])
    parser.add_argument('--train-freq', type=int, default=4)
    
    args = parser.parse_args()
    
    agent, model_path = train_rainbow_fixed_rewards(
        topology=args.topology,
        timesteps=args.timesteps,
        mode=args.mode,
        train_freq=args.train_freq
    )
    
    print(f"\n✅ Training complete with FIXED rewards!")
    print(f"\nTo train all topologies, run:")
    print(f"  for topo in gridnet bellcanada os3e interoute cogentco; do")
    print(f"    python train_rainbow_fixed.py --topology $topo --timesteps 20000")
    print(f"  done\n")