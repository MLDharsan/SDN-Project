#!/usr/bin/env python3
"""
Comprehensive Evaluation of Threshold-Based Proactive System

Evaluates:
1. Proactive Behavior (Parking, Evoking, Migrations)
2. Load Balancing Performance
3. Energy Efficiency
4. Latency Performance
5. Comparison with MOOO-RDQN paper
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Add environments to path
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments')
sys.path.insert(0, env_path)

from threshold_proactive_sdn_env import ThresholdBasedProactiveSDN

# MOOO-RDQN paper results (2 controllers)
PAPER_RESULTS = {
    'gridnet': {'cs_avg_latency': 3.34, 'cs_worst_latency': 8.71, 'load_balance': 2.68},
    'bellcanada': {'cs_avg_latency': 5.01, 'cs_worst_latency': 7.38, 'load_balance': 4.00},
    'os3e': {'cs_avg_latency': 2.91, 'cs_worst_latency': 8.65, 'load_balance': 3.20},
    'interoute': {'cs_avg_latency': 3.45, 'cs_worst_latency': 3.45, 'load_balance': 2.98},
    'cogentco': {'cs_avg_latency': 7.88, 'cs_worst_latency': 16.37, 'load_balance': 6.45}
}

def evaluate_model(model_path, topology='gridnet', episodes=10, steps_per_episode=1000):
    """
    Comprehensive evaluation of trained model
    
    Returns detailed metrics on:
    - Proactive behavior (parking, evoking, migrations)
    - Load balancing (variance, threshold violations)
    - Energy efficiency
    - Latency performance
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {topology.upper()}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Load model
    model = DQN.load(model_path)
    
    # Create environment
    env = ThresholdBasedProactiveSDN(
        topology_name=topology,
        num_slave_controllers=3,
        num_parked_controllers=2
    )
    
    # Metrics storage
    metrics = {
        # Proactive behavior
        'parking_events': [],
        'evoking_events': [],
        'migration_events': [],
        'park_evoke_ratio': [],
        
        # Load balancing
        'load_variance': [],
        'load_balance': [],
        'overload_violations': [],
        'underload_opportunities': [],
        
        # Energy
        'energy_consumption': [],
        'energy_savings': [],
        'active_controllers_avg': [],
        
        # Latency
        'avg_latency': [],
        'worst_latency': [],
        
        # Rewards
        'episode_rewards': [],
        
        # Time-series data for visualization
        'hourly_data': []
    }
    
    print(f"Running {episodes} episodes...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        
        # Episode metrics
        ep_parking = 0
        ep_evoking = 0
        ep_migrations = 0
        ep_load_vars = []
        ep_overloads = []
        ep_underloads = []
        ep_energy = []
        ep_latency = []
        ep_worst_latency = []
        ep_active = []
        ep_reward = 0
        
        # Hourly tracking
        hourly_loads = []
        hourly_active = []
        hourly_actions = []
        
        for step in range(steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            
            # Track proactive actions
            action_type = info.get('action_type', '')
            if 'park' in action_type and info.get('action_success'):
                ep_parking += 1
            elif 'evoke' in action_type and info.get('action_success'):
                ep_evoking += 1
            elif 'migrate' in action_type and info.get('action_success'):
                ep_migrations += 1
            
            # Track metrics
            ep_load_vars.append(info.get('load_variance', 0))
            ep_overloads.append(info.get('overloaded_count', 0))
            ep_underloads.append(info.get('underloaded_count', 0))
            ep_energy.append(info.get('energy', 0))
            ep_latency.append(info.get('latency', 0))
            ep_active.append(info.get('active_controllers', 0))
            
            # Track hourly data (every 50 steps = 1 hour)
            if step % 50 == 0:
                hourly_loads.append(np.mean(ep_load_vars[-50:]) if ep_load_vars else 0)
                hourly_active.append(info.get('active_controllers', 0))
                hourly_actions.append({
                    'park': ep_parking,
                    'evoke': ep_evoking,
                    'migrate': ep_migrations
                })
            
            # Track worst case latency
            if ep_latency:
                ep_worst_latency.append(max(ep_latency))
            
            if terminated or truncated:
                break
        
        # Store episode metrics
        metrics['parking_events'].append(ep_parking)
        metrics['evoking_events'].append(ep_evoking)
        metrics['migration_events'].append(ep_migrations)
        
        pe_ratio = ep_parking / ep_evoking if ep_evoking > 0 else 0
        metrics['park_evoke_ratio'].append(pe_ratio)
        
        metrics['load_variance'].append(np.mean(ep_load_vars))
        metrics['load_balance'].append(np.sqrt(np.mean(ep_load_vars)))
        metrics['overload_violations'].append(np.mean(ep_overloads))
        metrics['underload_opportunities'].append(np.mean(ep_underloads))
        
        metrics['energy_consumption'].append(np.mean(ep_energy))
        
        # Calculate energy savings vs all-active baseline
        baseline_energy = 50 + 3 * 100  # Master + 3 active slaves
        avg_energy = np.mean(ep_energy)
        savings = (baseline_energy - avg_energy) / baseline_energy * 100
        metrics['energy_savings'].append(savings)
        
        metrics['active_controllers_avg'].append(np.mean(ep_active))
        metrics['avg_latency'].append(np.mean(ep_latency))
        
        if ep_worst_latency:
            metrics['worst_latency'].append(np.mean(ep_worst_latency))
        
        metrics['episode_rewards'].append(ep_reward)
        
        metrics['hourly_data'].append({
            'loads': hourly_loads,
            'active': hourly_active,
            'actions': hourly_actions
        })
        
        print(f"  Episode {ep+1}/{episodes}: "
              f"Park={ep_parking}, Evoke={ep_evoking}, Migrate={ep_migrations}, "
              f"Energy={avg_energy:.0f}W, Latency={np.mean(ep_latency):.2f}ms")
    
    env.close()
    
    # Calculate summary statistics
    summary = {
        'topology': topology,
        'num_switches': env.num_switches,
        'episodes': episodes,
        
        # Proactive behavior
        'avg_parking_events': np.mean(metrics['parking_events']),
        'avg_evoking_events': np.mean(metrics['evoking_events']),
        'avg_migration_events': np.mean(metrics['migration_events']),
        'avg_pe_ratio': np.mean(metrics['park_evoke_ratio']),
        
        # Load balancing
        'avg_load_variance': np.mean(metrics['load_variance']),
        'avg_load_balance': np.mean(metrics['load_balance']),
        'avg_overload_violations': np.mean(metrics['overload_violations']),
        'avg_underload_opportunities': np.mean(metrics['underload_opportunities']),
        
        # Energy
        'avg_energy': np.mean(metrics['energy_consumption']),
        'avg_energy_savings': np.mean(metrics['energy_savings']),
        'avg_active_controllers': np.mean(metrics['active_controllers_avg']),
        
        # Latency
        'avg_latency': np.mean(metrics['avg_latency']),
        'worst_latency': np.mean(metrics['worst_latency']) if metrics['worst_latency'] else 0,
        
        # Overall
        'avg_reward': np.mean(metrics['episode_rewards']),
        
        # Raw metrics for visualization
        'raw_metrics': metrics
    }
    
    return summary

def print_evaluation_report(summary):
    """Print comprehensive evaluation report"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT: {summary['topology'].upper()}")
    print(f"{'='*70}\n")
    
    # Network info
    print(f"Network Configuration:")
    print(f"  Topology: {summary['topology']}")
    print(f"  Number of switches: {summary['num_switches']}")
    print(f"  Episodes evaluated: {summary['episodes']}")
    
    # Proactive behavior
    print(f"\n{'='*70}")
    print(f"1. PROACTIVE BEHAVIOR")
    print(f"{'='*70}")
    print(f"  Parking events per episode: {summary['avg_parking_events']:.1f}")
    print(f"  Evoking events per episode: {summary['avg_evoking_events']:.1f}")
    print(f"  Migration events per episode: {summary['avg_migration_events']:.1f}")
    print(f"  Park/Evoke ratio: {summary['avg_pe_ratio']:.2f}")
    
    if summary['avg_parking_events'] > 100:
        behavior_status = "✅ EXCELLENT - Very proactive!"
    elif summary['avg_parking_events'] > 50:
        behavior_status = "✅ GOOD - Proactive behavior learned"
    elif summary['avg_parking_events'] > 10:
        behavior_status = "⚠️  MODERATE - Some proactive behavior"
    else:
        behavior_status = "❌ POOR - Little proactive behavior"
    
    print(f"  Status: {behavior_status}")
    
    # Load balancing
    print(f"\n{'='*70}")
    print(f"2. LOAD BALANCING PERFORMANCE")
    print(f"{'='*70}")
    print(f"  Load variance: {summary['avg_load_variance']:.4f}")
    print(f"  Load balance index: {summary['avg_load_balance']:.4f}")
    print(f"  Avg overload violations: {summary['avg_overload_violations']:.2f}")
    print(f"  Avg underload opportunities: {summary['avg_underload_opportunities']:.2f}")
    
    if summary['avg_load_balance'] < 0.1:
        balance_status = "✅ EXCELLENT - Very balanced"
    elif summary['avg_load_balance'] < 0.3:
        balance_status = "✅ GOOD - Well balanced"
    elif summary['avg_load_balance'] < 0.5:
        balance_status = "⚠️  MODERATE - Somewhat balanced"
    else:
        balance_status = "❌ POOR - Imbalanced"
    
    print(f"  Status: {balance_status}")
    
    # Energy efficiency
    print(f"\n{'='*70}")
    print(f"3. ENERGY EFFICIENCY")
    print(f"{'='*70}")
    print(f"  Average energy consumption: {summary['avg_energy']:.2f} W")
    print(f"  Average energy savings: {summary['avg_energy_savings']:.1f}%")
    print(f"  Average active controllers: {summary['avg_active_controllers']:.2f}")
    print(f"  Baseline (all active): 350 W")
    
    if summary['avg_energy_savings'] > 20:
        energy_status = "✅ EXCELLENT - Significant savings"
    elif summary['avg_energy_savings'] > 10:
        energy_status = "✅ GOOD - Notable savings"
    elif summary['avg_energy_savings'] > 5:
        energy_status = "⚠️  MODERATE - Some savings"
    else:
        energy_status = "❌ POOR - Minimal savings"
    
    print(f"  Status: {energy_status}")
    
    # Latency
    print(f"\n{'='*70}")
    print(f"4. LATENCY PERFORMANCE")
    print(f"{'='*70}")
    print(f"  Average latency: {summary['avg_latency']:.2f} ms")
    print(f"  Worst-case latency: {summary['worst_latency']:.2f} ms")
    
    # Compare with MOOO-RDQN if available
    topo = summary['topology'].lower()
    if topo in PAPER_RESULTS:
        paper = PAPER_RESULTS[topo]
        print(f"\n  Comparison with MOOO-RDQN paper:")
        
        diff_avg = ((paper['cs_avg_latency'] - summary['avg_latency']) / 
                    paper['cs_avg_latency'] * 100)
        status_avg = "✅ Better" if diff_avg > 0 else "⚠️  Higher"
        print(f"    Avg latency: {summary['avg_latency']:.2f} ms vs {paper['cs_avg_latency']:.2f} ms "
              f"({diff_avg:+.1f}%) {status_avg}")
        
        diff_worst = ((paper['cs_worst_latency'] - summary['worst_latency']) / 
                     paper['cs_worst_latency'] * 100)
        status_worst = "✅ Better" if diff_worst > 0 else "⚠️  Higher"
        print(f"    Worst latency: {summary['worst_latency']:.2f} ms vs {paper['cs_worst_latency']:.2f} ms "
              f"({diff_worst:+.1f}%) {status_worst}")
        
        diff_load = ((paper['load_balance'] - summary['avg_load_balance']) / 
                    paper['load_balance'] * 100)
        status_load = "✅ Better" if diff_load > 0 else "⚠️  Higher"
        print(f"    Load balance: {summary['avg_load_balance']:.2f} vs {paper['load_balance']:.2f} "
              f"({diff_load:+.1f}%) {status_load}")
    
    # Overall assessment
    print(f"\n{'='*70}")
    print(f"5. OVERALL ASSESSMENT")
    print(f"{'='*70}")
    print(f"  Average reward: {summary['avg_reward']:.2f}")
    
    # Count excellent/good ratings
    ratings = [
        summary['avg_parking_events'] > 50,  # Proactive
        summary['avg_load_balance'] < 0.3,   # Load balance
        summary['avg_energy_savings'] > 10,  # Energy
    ]
    
    score = sum(ratings)
    
    if score >= 3:
        overall = "✅ EXCELLENT - All objectives achieved!"
    elif score >= 2:
        overall = "✅ GOOD - Most objectives met"
    elif score >= 1:
        overall = "⚠️  MODERATE - Some objectives met"
    else:
        overall = "❌ NEEDS IMPROVEMENT"
    
    print(f"  Overall rating: {overall}")
    print(f"  Score: {score}/3 objectives")
    
    print(f"\n{'='*70}\n")

def create_visualization(summary):
    """Create visualization of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Proactive Actions
    ax = axes[0, 0]
    actions = ['Parking', 'Evoking', 'Migrations']
    values = [
        summary['avg_parking_events'],
        summary['avg_evoking_events'],
        summary['avg_migration_events']
    ]
    ax.bar(actions, values, color=['green', 'blue', 'orange'])
    ax.set_ylabel('Events per Episode')
    ax.set_title('Proactive Actions', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Energy Efficiency
    ax = axes[0, 1]
    baseline = 350
    actual = summary['avg_energy']
    ax.bar(['Baseline\n(All Active)', 'DRL Model'], [baseline, actual], 
           color=['red', 'green'])
    ax.set_ylabel('Energy (Watts)')
    ax.set_title(f"Energy Efficiency ({summary['avg_energy_savings']:.1f}% savings)", 
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Load Balance Over Episodes
    ax = axes[1, 0]
    episodes = range(1, len(summary['raw_metrics']['load_balance']) + 1)
    ax.plot(episodes, summary['raw_metrics']['load_balance'], 'b-o', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Load Balance Index')
    ax.set_title('Load Balance Performance', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Latency Comparison
    ax = axes[1, 1]
    if summary['topology'].lower() in PAPER_RESULTS:
        paper = PAPER_RESULTS[summary['topology'].lower()]
        metrics = ['Avg Latency', 'Worst Latency']
        your_values = [summary['avg_latency'], summary['worst_latency']]
        paper_values = [paper['cs_avg_latency'], paper['cs_worst_latency']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, your_values, width, label='Your DRL', color='blue')
        ax.bar(x + width/2, paper_values, width, label='MOOO-RDQN', color='orange')
        
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Comparison with MOOO-RDQN', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No comparison\ndata available', 
               ha='center', va='center', fontsize=14)
        ax.set_title('Latency Comparison', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_dir = '../results/evaluations/'
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}evaluation_{summary['topology']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    print(f"📊 Visualization saved: {filename}")
    
    try:
        plt.show()
    except:
        pass
    
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate threshold-based proactive system')
    parser.add_argument('--topology', default='gridnet',
                       choices=['gridnet', 'bellcanada', 'os3e', 'interoute', 'cogentco'])
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    model_path = f'../models/threshold_proactive_{args.topology}.zip'
    
    # Run evaluation
    summary = evaluate_model(model_path, args.topology, args.episodes)
    
    if summary is None:
        print("\n❌ Evaluation failed!")
        print(f"Make sure model exists: {model_path}")
        print(f"\nTrain first: python train_threshold_based.py --topology {args.topology}")
        return
    
    # Print report
    print_evaluation_report(summary)
    
    # Create visualization
    if args.visualize:
        create_visualization(summary)
    
    # Save results
    results_dir = '../results/evaluations/'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = f"{results_dir}eval_{args.topology}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Remove raw_metrics for cleaner JSON
    summary_clean = {k: v for k, v in summary.items() if k != 'raw_metrics'}
    
    with open(results_file, 'w') as f:
        json.dump(summary_clean, f, indent=2)
    
    print(f"💾 Results saved: {results_file}")
    print()

if __name__ == "__main__":
    main()
