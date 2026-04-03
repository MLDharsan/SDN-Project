#!/usr/bin/env python3
"""
Traffic Pattern Analyzer
Shows exactly how traffic varies in the threshold-based environment
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add environments to path
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments')
sys.path.insert(0, env_path)

from threshold_proactive_sdn_env import ThresholdBasedProactiveSDN

def analyze_traffic_pattern(topology='gridnet', total_steps=2400):
    """
    Analyze traffic patterns over a full day cycle
    
    Args:
        topology: Network topology
        total_steps: Steps to simulate (2400 = ~1 day at 50 steps/hour)
    """
    print("\n" + "="*70)
    print("TRAFFIC PATTERN ANALYSIS")
    print("="*70 + "\n")
    
    # Create environment
    env = ThresholdBasedProactiveSDN(
        topology_name=topology,
        num_slave_controllers=3,
        num_parked_controllers=2,
        use_real_data=True
    )
    
    # Reset and track traffic
    obs, _ = env.reset()
    
    # Storage
    hours = []
    avg_loads = []
    max_loads = []
    min_loads = []
    active_controllers = []
    
    print(f"Simulating {total_steps} steps (~{total_steps//50} hours)...")
    print("\nTraffic by hour:")
    print(f"{'Hour':<6} {'Phase':<12} {'Avg Load':<12} {'Min Load':<12} {'Max Load':<12}")
    print("-"*60)
    
    for step in range(total_steps):
        # Take a no-op action to just observe
        obs, reward, terminated, truncated, info = env.step(0)
        
        # Every 50 steps = 1 hour change
        if step % 50 == 0:
            hour = env.traffic_phase
            
            # Get current loads
            loads = [env.controller_loads[c] for c in env.active_slaves if c in env.controller_loads]
            
            if loads:
                avg_load = np.mean(loads)
                max_load = np.max(loads)
                min_load = np.min(loads)
            else:
                avg_load = max_load = min_load = 0
            
            # Determine phase
            if 9 <= hour <= 17:
                phase = "PEAK"
            elif 18 <= hour <= 22:
                phase = "EVENING"
            else:
                phase = "NIGHT"
            
            hours.append(hour)
            avg_loads.append(avg_load)
            max_loads.append(max_load)
            min_loads.append(min_load)
            active_controllers.append(len(env.active_slaves))
            
            print(f"{hour:>4}h  {phase:<12} {avg_load:>11.2%} {min_load:>11.2%} {max_load:>11.2%}")
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Analysis
    print("\n" + "="*70)
    print("TRAFFIC STATISTICS")
    print("="*70)
    
    avg_loads_arr = np.array(avg_loads)
    
    print(f"\nOverall Statistics:")
    print(f"  Mean load: {np.mean(avg_loads_arr):.2%}")
    print(f"  Std deviation: {np.std(avg_loads_arr):.2%}")
    print(f"  Min load: {np.min(avg_loads_arr):.2%}")
    print(f"  Max load: {np.max(avg_loads_arr):.2%}")
    
    # Categorize by time
    peak_loads = [avg_loads[i] for i, h in enumerate(hours) if 9 <= h <= 17]
    evening_loads = [avg_loads[i] for i, h in enumerate(hours) if 18 <= h <= 22]
    night_loads = [avg_loads[i] for i, h in enumerate(hours) if h < 9 or h > 22]
    
    print(f"\nBy Time Period:")
    if peak_loads:
        print(f"  Peak hours (9am-5pm):")
        print(f"    Mean: {np.mean(peak_loads):.2%}, Range: {np.min(peak_loads):.2%} - {np.max(peak_loads):.2%}")
    if evening_loads:
        print(f"  Evening (6pm-10pm):")
        print(f"    Mean: {np.mean(evening_loads):.2%}, Range: {np.min(evening_loads):.2%} - {np.max(evening_loads):.2%}")
    if night_loads:
        print(f"  Night/Early morning:")
        print(f"    Mean: {np.mean(night_loads):.2%}, Range: {np.min(night_loads):.2%} - {np.max(night_loads):.2%}")
    
    # Traffic variation assessment
    print(f"\nTraffic Pattern Type:")
    std_ratio = np.std(avg_loads_arr) / np.mean(avg_loads_arr)
    
    if std_ratio > 0.5:
        pattern = "HIGHLY VARIABLE (Strong day/night cycle)"
    elif std_ratio > 0.3:
        pattern = "MODERATELY VARIABLE (Clear day/night pattern)"
    elif std_ratio > 0.1:
        pattern = "SLIGHTLY VARIABLE (Gentle fluctuations)"
    else:
        pattern = "UNIFORM (Almost constant traffic)"
    
    print(f"  Classification: {pattern}")
    print(f"  Coefficient of variation: {std_ratio:.2%}")
    
    # Threshold violations
    overload_threshold = 0.70
    underload_threshold = 0.30
    
    overload_hours = sum(1 for load in avg_loads if load > overload_threshold)
    underload_hours = sum(1 for load in avg_loads if load < underload_threshold)
    normal_hours = len(avg_loads) - overload_hours - underload_hours
    
    print(f"\nThreshold Violations:")
    print(f"  Overload (>70%): {overload_hours} hours ({overload_hours/len(avg_loads)*100:.1f}%)")
    print(f"  Normal (30-70%): {normal_hours} hours ({normal_hours/len(avg_loads)*100:.1f}%)")
    print(f"  Underload (<30%): {underload_hours} hours ({underload_hours/len(avg_loads)*100:.1f}%)")
    
    # Create visualization
    create_traffic_visualization(hours, avg_loads, max_loads, min_loads, 
                                 active_controllers, topology)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")

def create_traffic_visualization(hours, avg_loads, max_loads, min_loads, 
                                 active_controllers, topology):
    """Create traffic pattern visualization"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Load patterns
    ax1.plot(hours, avg_loads, 'b-', linewidth=2, label='Average Load')
    ax1.fill_between(hours, min_loads, max_loads, alpha=0.3, label='Min-Max Range')
    
    # Add threshold lines
    ax1.axhline(y=0.70, color='r', linestyle='--', linewidth=1.5, label='Overload Threshold (70%)')
    ax1.axhline(y=0.30, color='g', linestyle='--', linewidth=1.5, label='Underload Threshold (30%)')
    
    # Shade time periods
    for hour in hours:
        if 9 <= hour <= 17:
            ax1.axvspan(hour-0.5, hour+0.5, alpha=0.1, color='yellow')
        elif hour < 9 or hour > 22:
            ax1.axvspan(hour-0.5, hour+0.5, alpha=0.1, color='blue')
    
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Controller Load', fontsize=12)
    ax1.set_title(f'Traffic Pattern Analysis - {topology.upper()}\n(Yellow=Peak, Blue=Night)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Active controllers
    ax2.plot(hours, active_controllers, 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Active Controllers', fontsize=12)
    ax2.set_title('Controller Activity Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 4)
    
    plt.tight_layout()
    
    # Save
    output_dir = '../results/traffic_analysis/'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}traffic_pattern_{topology}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    print(f"\n📊 Visualization saved: {filename}")
    
    # Try to display
    try:
        plt.show()
    except:
        print("   (Display not available, but image saved)")
    
    plt.close()

def compare_topologies():
    """Compare traffic patterns across all topologies"""
    
    topologies = ['gridnet', 'bellcanada', 'os3e', 'interoute', 'cogentco']
    
    print("\n" + "="*70)
    print("COMPARING TRAFFIC PATTERNS ACROSS TOPOLOGIES")
    print("="*70 + "\n")
    
    results = {}
    
    for topo in topologies:
        print(f"\nAnalyzing {topo}...")
        
        env = ThresholdBasedProactiveSDN(topology_name=topo)
        obs, _ = env.reset()
        
        loads = []
        for step in range(1000):
            obs, _, terminated, truncated, _ = env.step(0)
            
            if step % 50 == 0:
                active_loads = [env.controller_loads[c] for c in env.active_slaves]
                if active_loads:
                    loads.append(np.mean(active_loads))
            
            if terminated or truncated:
                break
        
        env.close()
        
        if loads:
            results[topo] = {
                'mean': np.mean(loads),
                'std': np.std(loads),
                'min': np.min(loads),
                'max': np.max(loads),
                'cv': np.std(loads) / np.mean(loads) if np.mean(loads) > 0 else 0
            }
    
    # Print comparison
    print("\n" + "="*70)
    print(f"{'Topology':<15} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10} {'CV':<10}")
    print("-"*70)
    
    for topo, stats in results.items():
        print(f"{topo:<15} {stats['mean']:>9.2%} {stats['std']:>9.2%} "
              f"{stats['min']:>9.2%} {stats['max']:>9.2%} {stats['cv']:>9.2f}")
    
    print("="*70)
    print("\nCV = Coefficient of Variation (higher = more variable traffic)")
    print("="*70 + "\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze traffic patterns')
    parser.add_argument('--topology', default='gridnet',
                       choices=['gridnet', 'bellcanada', 'os3e', 'interoute', 'cogentco'])
    parser.add_argument('--steps', type=int, default=2400,
                       help='Number of steps to simulate (2400 = ~1 day)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all topologies')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_topologies()
    else:
        analyze_traffic_pattern(args.topology, args.steps)

if __name__ == "__main__":
    main()
