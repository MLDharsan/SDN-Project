#!/usr/bin/env python3
"""
IMPROVED Rainbow DQN Training - Fixed Rewards, Better Exploration & Adaptive Learning

Key Improvements:
1. Fixed reward calculation bug (no more 0.00!)
2. Epsilon-greedy exploration
3. Enhanced reward shaping
4. Adaptive training frequency
5. Curriculum learning
6. Stuck agent detection
7. Topology-specific hyperparameters
"""
import sys
import os
import torch
import numpy as np
from datetime import datetime
from collections import deque

def train_rainbow_safe(
    topology='gridnet',
    timesteps=20000,
    mode='proactive',
    train_freq=4,
    convergence_window=3,
    eval_freq=2000
):
    """
    Train Rainbow DQN with SAFE model saving and IMPROVED learning
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 IMPROVED RAINBOW DQN TRAINING")
    print(f"{'='*70}\n")
    
    print(f"Configuration:")
    print(f"  Topology: {topology} ({get_topology_size(topology)} nodes)")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Mode: {mode}")
    print(f"  Improvements: Fixed rewards, Better exploration, Adaptive learning")
    print()
    
    # Import after setup
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
    # IMPROVED REWARD SHAPING - Both Proactive and Reactive Modes
    # ========================================================================
    original_step = env.step
    
    def enhanced_step(action):
        obs, reward, terminated, truncated, info = original_step(action)
        action_type = info.get('action_type', '')
        
        # Base penalties for actions (different for proactive vs reactive)
        if mode == 'reactive':
            # Heavy penalties for reactive mode
            if 'park' in action_type and info.get('action_success'):
                reward -= 500.0
            elif 'evoke' in action_type and info.get('action_success'):
                reward -= 300.0
            elif 'migrate' in action_type and info.get('action_success'):
                reward -= 5.0
        else:
            # Lighter penalties for proactive mode (encourage action)
            if 'park' in action_type and info.get('action_success'):
                reward -= 10.0  # Small penalty
            elif 'evoke' in action_type and info.get('action_success'):
                reward -= 15.0  # Slightly higher penalty
            elif 'migrate' in action_type and info.get('action_success'):
                reward -= 2.0   # Very small penalty
        
        # ===== REWARD SHAPING - Load Balancing =====
        if 'controller_loads' in info:
            loads = np.array(info['controller_loads'])
            active_loads = loads[loads > 0]  # Only count active controllers
            
            if len(active_loads) > 0:
                load_std = np.std(active_loads)
                load_mean = np.mean(active_loads)
                
                # Bonus for balanced load
                if load_std < 2.0:  # Very well balanced
                    reward += 20.0
                elif load_std < 5.0:  # Moderately balanced
                    reward += 10.0
                
                # Penalty for high imbalance
                if load_std > 10.0:
                    reward -= 30.0
                
                # Bonus for even distribution
                if load_mean > 0:
                    balance_ratio = load_std / (load_mean + 1e-6)
                    if balance_ratio < 0.3:  # Less than 30% variation
                        reward += 15.0
        
        # ===== REWARD SHAPING - Controller Overload =====
        if 'overloaded_controllers' in info:
            overloaded = info['overloaded_controllers']
            if overloaded > 0:
                reward -= 100.0 * overloaded  # Heavy penalty for overload
        
        # ===== REWARD SHAPING - Energy Efficiency =====
        if 'active_controllers' in info:
            active = info['active_controllers']
            total_controllers = info.get('total_controllers', 5)
            
            # Bonus for energy efficiency (but not too few)
            if active == 2:
                reward += 25.0  # Optimal energy usage
            elif active == 3:
                reward += 15.0  # Good energy usage
            elif active == 1:
                reward -= 20.0  # Too few, might cause overload
            elif active >= 4:
                reward -= 10.0 * (active - 3)  # Penalty for too many active
        
        # ===== REWARD SHAPING - Latency (if available) =====
        if 'avg_latency' in info:
            latency = info['avg_latency']
            if latency < 5.0:
                reward += 10.0  # Low latency bonus
            elif latency > 15.0:
                reward -= 20.0  # High latency penalty
        
        # ===== REWARD SHAPING - Failed Actions =====
        if not info.get('action_success', True) and action_type != 'NO_ACTION':
            reward -= 50.0  # Penalty for failed actions
        
        return obs, reward, terminated, truncated, info
    
    env.step = enhanced_step
    
    # ========================================================================
    # TOPOLOGY-SPECIFIC HYPERPARAMETERS
    # ========================================================================
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Adjust hyperparameters based on topology size
    if topology in ['interoute', 'cogentco']:
        # Large networks (110-197 nodes)
        hidden_dim = 512
        lr = 5e-5
        n_step = 5
        buffer_size = 100000
        batch_size = 256
        print(f"  Using LARGE network hyperparameters (hidden_dim={hidden_dim})")
    elif topology in ['bellcanada', 'os3e']:
        # Medium networks (34-48 nodes)
        hidden_dim = 256
        lr = 1e-4
        n_step = 3
        buffer_size = 50000
        batch_size = 128
        print(f"  Using MEDIUM network hyperparameters (hidden_dim={hidden_dim})")
    else:
        # Small networks (9 nodes)
        hidden_dim = 128
        lr = 2e-4
        n_step = 3
        buffer_size = 30000
        batch_size = 64
        print(f"  Using SMALL network hyperparameters (hidden_dim={hidden_dim})")
    
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
    
    # ========================================================================
    # EXPLORATION PARAMETERS
    # ========================================================================
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = timesteps * 0.5  # Decay over first 50% of training
    
    def get_epsilon(step):
        """Calculate epsilon for epsilon-greedy exploration"""
        return epsilon_end + (epsilon_start - epsilon_end) * \
               np.exp(-1.0 * step / epsilon_decay)
    
    # ========================================================================
    # CURRICULUM LEARNING PARAMETERS
    # ========================================================================
    def set_traffic_intensity(step):
        """Gradually increase traffic variation difficulty"""
        if timesteps > 30000:
            if step < timesteps * 0.3:
                return 0.3  # Easy - low variation
            elif step < timesteps * 0.6:
                return 0.6  # Medium variation
            else:
                return 1.0  # Hard - full variation
        else:
            # Short training - skip curriculum
            return 1.0
    
    # ========================================================================
    # TRACKING METRICS
    # ========================================================================
    episode_rewards = []
    eval_rewards = deque(maxlen=convergence_window)
    parking_events = 0
    evoking_events = 0
    migrate_events = 0
    no_action_count = 0
    losses = []
    
    # Action tracking for stuck detection
    recent_actions = deque(maxlen=200)
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    
    print(f"Starting training on {device}...\n")
    print("Progress: [", end='', flush=True)
    
    for step in range(timesteps):
        # Progress bar
        if step % (timesteps // 50) == 0:
            print('=', end='', flush=True)
        
        # ===== CURRICULUM LEARNING - Adjust Traffic =====
        if step % 100 == 0:
            env.traffic_phase = (env.traffic_phase + 1) % 24
            
            # Set traffic intensity based on curriculum
            traffic_intensity = set_traffic_intensity(step)
            if hasattr(env, '_simulate_traffic_variation'):
                env._simulate_traffic_variation()
        
        # ===== EPSILON-GREEDY EXPLORATION =====
        epsilon = get_epsilon(step)
        
        if np.random.random() < epsilon:
            # Exploration: random action
            action = np.random.randint(action_dim)
        else:
            # Exploitation: use agent's policy
            action = agent.select_action(state, training=True)
        
        # ===== TAKE STEP =====
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # ===== TRACK EVENTS =====
        action_type = info.get('action_type', '')
        recent_actions.append(action_type)
        
        if 'park' in action_type and info.get('action_success'):
            parking_events += 1
        elif 'evoke' in action_type and info.get('action_success'):
            evoking_events += 1
        elif 'migrate' in action_type and info.get('action_success'):
            migrate_events += 1
        elif action_type == 'NO_ACTION':
            no_action_count += 1
        
        # ===== STORE TRANSITION =====
        agent.push_n_step(state, action, reward, next_state, done)
        
        # ===== ADAPTIVE TRAINING FREQUENCY =====
        # Train more frequently as training progresses
        adaptive_train_freq = max(1, train_freq - (step // 10000))
        
        if step > 1000 and step % adaptive_train_freq == 0 and len(agent.memory) >= agent.batch_size:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        
        # ===== UPDATE TARGET NETWORK =====
        if step % 1000 == 0 and step > 0:
            agent.update_target_network()
        
        episode_reward += reward
        state = next_state
        
        # ===== EPISODE END =====
        if done:
            episode_rewards.append(episode_reward)
            episode_count += 1
            state, _ = env.reset()
            episode_reward = 0
        
        # ===== STUCK AGENT DETECTION =====
        if step > 0 and step % 2000 == 0:
            check_stuck_agent(recent_actions, step)
        
        # ===== EVALUATION & EARLY STOPPING =====
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
            
            # Action distribution
            if len(recent_actions) > 0:
                action_dist = get_action_distribution(recent_actions)
                print(f"Recent Action Distribution: {action_dist}")
            
            # Check convergence
            if len(eval_rewards) == convergence_window:
                reward_std = np.std(eval_rewards)
                reward_mean = np.mean(eval_rewards)
                
                # Improved convergence criterion
                if reward_std < 50.0 and reward_mean > -200:
                    print(f"\n🎉 CONVERGED! Reward std: {reward_std:.2f} < 50.0")
                    print(f"Mean reward: {reward_mean:.2f} > -200")
                    print(f"Stopping early at step {step}")
                    break
            
            print(f"{'='*70}\n")
            print("Progress: [", end='', flush=True)
    
    print("]\n")
    
    # ========================================================================
    # SAFE MODEL SAVING - FIXED REWARD CALCULATION
    # ========================================================================
    
    os.makedirs('models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = f'models/rainbow_{mode}_{topology}_{timesteps}steps_{timestamp}.pth'
    
    agent.save(model_path)
    
    # ===== FIXED METADATA - NO MORE 0.00 REWARDS! =====
    metadata_path = model_path.replace('.pth', '_metadata.json')
    import json
    
    # Calculate ALL reward metrics properly
    final_avg_reward = float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 \
                      else float(np.mean(episode_rewards)) if len(episode_rewards) > 0 \
                      else 0.0
    
    metadata = {
        'topology': topology,
        'topology_size': get_topology_size(topology),
        'timesteps': timesteps,
        'mode': mode,
        'train_freq': train_freq,
        'timestamp': datetime.now().isoformat(),
        
        # ===== FIXED REWARD METRICS =====
        'final_avg_reward_last100': final_avg_reward,
        'final_avg_reward_all': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'total_cumulative_reward': float(np.sum(episode_rewards)),
        'best_episode_reward': float(np.max(episode_rewards)) if episode_rewards else 0.0,
        'worst_episode_reward': float(np.min(episode_rewards)) if episode_rewards else 0.0,
        'final_episode_reward': float(episode_rewards[-1]) if episode_rewards else 0.0,
        
        # Events
        'parking_events': parking_events,
        'evoking_events': evoking_events,
        'migrate_events': migrate_events,
        'no_action_count': no_action_count,
        'pe_ratio': parking_events / evoking_events if evoking_events > 0 else float('inf'),
        
        # Training stats
        'total_episodes': episode_count,
        'avg_loss': float(np.mean(losses)) if losses else 0.0,
        'final_epsilon': float(get_epsilon(timesteps)),
        
        # Hyperparameters
        'hyperparameters': {
            'learning_rate': lr,
            'gamma': 0.99,
            'n_step': n_step,
            'batch_size': batch_size,
            'buffer_size': buffer_size,
            'hidden_dim': hidden_dim,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
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
    print(f"  Best Episode: {metadata['best_episode_reward']:.2f}")
    print(f"  Worst Episode: {metadata['worst_episode_reward']:.2f}")
    print(f"\n🎬 Action Summary:")
    print(f"  Parking: {parking_events} | Evoking: {evoking_events}")
    print(f"  Migrate: {migrate_events} | NoAction: {no_action_count}")
    if evoking_events > 0:
        print(f"  P/E Ratio: {parking_events / evoking_events:.2f}")
    print(f"{'='*70}\n")
    
    # Also save as "latest"
    latest_path = f'models/LATEST_rainbow_{mode}_{topology}.pth'
    agent.save(latest_path)
    print(f"💡 Latest model also saved as: {latest_path}\n")
    
    env.close()
    return agent, model_path


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

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


def check_stuck_agent(recent_actions, step):
    """Detect if agent is stuck in repetitive behavior"""
    if len(recent_actions) == 0:
        return
    
    action_counts = {}
    for action in recent_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    total = len(recent_actions)
    no_action_pct = action_counts.get('NO_ACTION', 0) / total
    
    if no_action_pct > 0.9:
        print(f"\n⚠️  WARNING at step {step}: Agent stuck in NO_ACTION (>90%)")
        print(f"   Action distribution: {action_counts}")
        print(f"   Consider: Check reward function or increase exploration\n")


def get_action_distribution(recent_actions):
    """Get percentage distribution of actions"""
    if len(recent_actions) == 0:
        return {}
    
    action_counts = {}
    for action in recent_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    total = len(recent_actions)
    return {k: f"{100*v/total:.1f}%" for k, v in action_counts.items()}


def compare_models():
    """Compare all saved models with FIXED reward display"""
    import glob
    import json
    
    print(f"\n{'='*70}")
    print(f"📂 SAVED MODELS COMPARISON")
    print(f"{'='*70}\n")
    
    model_files = sorted(glob.glob('models/rainbow_*.pth'))
    
    if not model_files:
        print("No models found in models/ directory")
        return
    
    for model_file in model_files:
        if 'LATEST' in model_file:
            continue
        
        metadata_file = model_file.replace('.pth', '_metadata.json')
        
        print(f"📄 {os.path.basename(model_file)}")
        
        if os.path.exists(metadata_file):
            with open(metadata_file) as f:
                meta = json.load(f)
            
            print(f"   Topology: {meta['topology']} ({meta.get('topology_size', '?')} nodes)")
            print(f"   Timesteps: {meta['timesteps']:,}")
            print(f"   Mode: {meta['mode']}")
            
            # Display FIXED reward metrics
            print(f"   Avg Reward (last 100): {meta.get('final_avg_reward_last100', 0):.2f}")
            print(f"   Avg Reward (all): {meta.get('final_avg_reward_all', 0):.2f}")
            print(f"   Total Reward: {meta.get('total_cumulative_reward', 0):.2f}")
            print(f"   Best Episode: {meta.get('best_episode_reward', 0):.2f}")
            
            print(f"   Parking: {meta['parking_events']} | Evoking: {meta['evoking_events']}")
            print(f"   Migrate: {meta.get('migrate_events', 0)} | NoAction: {meta.get('no_action_count', 0)}")
            
            pe_ratio = meta.get('pe_ratio', 0)
            if pe_ratio != float('inf'):
                print(f"   P/E Ratio: {pe_ratio:.2f}")
            else:
                print(f"   P/E Ratio: ∞ (no evoking events)")
            
            print(f"   Trained: {meta['timestamp']}")
        else:
            print(f"   (No metadata found)")
        
        print()
    
    print(f"{'='*70}\n")


# ========================================================================
# MAIN
# ========================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Rainbow DQN Training')
    parser.add_argument('--topology', default='os3e',
                       choices=['gridnet', 'bellcanada', 'os3e', 'interoute', 'cogentco'])
    parser.add_argument('--timesteps', type=int, default=20000)
    parser.add_argument('--mode', default='proactive',
                       choices=['proactive', 'reactive'])
    parser.add_argument('--train-freq', type=int, default=4)
    parser.add_argument('--compare', action='store_true',
                       help='Compare all saved models instead of training')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        agent, model_path = train_rainbow_safe(
            topology=args.topology,
            timesteps=args.timesteps,
            mode=args.mode,
            train_freq=args.train_freq
        )
        
        print(f"\n✅ Training complete! Model safely saved.")
        print(f"\nTo compare all models, run:")
        print(f"  python {sys.argv[0]} --compare\n")