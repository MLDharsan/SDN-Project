#!/usr/bin/env python3
"""
Proactive SDN Environment with REAL DATA Integration
Uses real topologies (Internet Topology Zoo) and traffic (CIC-Bell-DNS 2021)
ENHANCED: Better traffic patterns to encourage proactive behavior
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.topology_loader import TopologyLoader
    from utils.traffic_loader import TrafficLoader
except ImportError:
    print("⚠️  Could not import loaders, using abstract simulation")
    TopologyLoader = None
    TrafficLoader = None


class ProactiveSDNEnvReal(gym.Env):
    """
    SDN Environment with Real Network Topologies and Enhanced Traffic Patterns
    Compatible with MOOO-RDQN paper datasets
    ENHANCED: Clear day/night traffic variation to encourage proactive behavior
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 topology_name='os3e',  # gridnet, bellcanada, or os3e
                 num_parked_controllers=2,
                 history_length=5,
                 use_real_data=True,
                 enhance_traffic=True):  # NEW: Option to enhance traffic patterns
        super().__init__()
        
        self.use_real_data = use_real_data and (TopologyLoader is not None)
        self.enhance_traffic = enhance_traffic  # NEW: Enhanced traffic flag
        
        if self.use_real_data:
            # Load real topology
            print(f"🌐 Loading real topology: {topology_name}")
            self.topology_loader = TopologyLoader()
            self.topology_info = self.topology_loader.get_topology_info(topology_name)
            
            # Use real node count as switches
            self.num_switches = self.topology_info['num_nodes']
            
            # Use appropriate number of controllers based on network size
            if self.num_switches <= 10:
                self.num_active = 2
            elif self.num_switches <= 50:
                self.num_active = 3
            else:
                self.num_active = 5
            
            # Load real traffic data
            print(f"📊 Loading real traffic data...")
            self.traffic_loader = TrafficLoader()
            self.traffic_data = self.traffic_loader.load_cic_dns_2021('FirstDayBenign')
            self.traffic_loader.print_pattern_summary()
            
            # Show enhancement status
            if self.enhance_traffic:
                print(f"✨ Traffic pattern ENHANCED for proactive learning")
                print(f"   Peak (09-17): 100% | Night (00-06, 22-24): 30% | Variation: 70%")
            
            print(f"✅ Real data environment initialized:")
            print(f"   Topology: {topology_name} ({self.num_switches} switches)")
            print(f"   Controllers: {self.num_active} active + {num_parked_controllers} parked")
        else:
            # Fallback to abstract simulation
            print(f"📊 Using abstract simulation (real data unavailable)")
            self.num_switches = 34  # Same as OS3E
            self.num_active = 3
            self.topology_loader = None
            self.traffic_loader = None
            self.enhance_traffic = True  # Always enhance in abstract mode
        
        self.num_parked = num_parked_controllers
        self.total_controllers = self.num_active + self.num_parked
        self.history_length = history_length
        
        # Action space
        self.action_space = spaces.Discrete(
            4 + self.num_switches * self.num_active
        )
        
        # Observation space
        self.single_state_dim = (self.num_active + self.num_switches + 
                                 self.num_parked + 2)
        self.obs_dim = self.single_state_dim * (history_length + 1)
        
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )
        
        # History buffer
        self.state_history = deque(maxlen=history_length)
        
        # Energy constants
        self.ACTIVE_POWER = 150
        self.PARKED_POWER = 10
        
        # Simulation parameters
        self.time_step = 0
        self.max_steps = 1000
        self.traffic_phase = 0

        # ========================= NEW: base loads separate from traffic loads =========================
        self.base_controller_loads = np.zeros(self.num_active, dtype=float)
        self.current_traffic_multiplier = 1.0
        
        # Initialize
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize controller loads (will be overwritten by base+traffic below)
        self.controller_loads = np.random.uniform(0.3, 0.7, self.num_active)
        
        # Initialize switch mappings
        self.switch_mappings = np.random.randint(0, self.num_active, self.num_switches)

        # Compute base loads from mappings
        self._update_loads()
        
        # Random starting hour (kept as original variable; time_step logic unchanged)
        self.traffic_phase = np.random.randint(0, 24)
        
        # Initial parked status
        if np.random.rand() > 0.3:
            self.parked_status = np.array([1] + [0] * (self.num_parked - 1))
        else:
            self.parked_status = np.zeros(self.num_parked, dtype=int)

        # Apply traffic once so latency/energy reflect current traffic
        self._simulate_traffic_variation()
        
        self.current_latency = self._calculate_latency()
        self.current_energy = self._calculate_energy()
        
        self.state_history.clear()
        current_state = self._get_current_state()
        for _ in range(self.history_length):
            self.state_history.append(current_state)
        
        self.time_step = 0
        
        return self._get_observation(), {}
    
    def _get_enhanced_traffic_multiplier(self, hour):
        """
        Enhanced traffic pattern with CLEAR day/night variation
        This makes parking economically viable for the agent
        
        Pattern:
        - Night (00-06, 22-24): 30% load (LOW)
        - Morning ramp (06-09): 60% load
        - Peak hours (09-17): 100% load (HIGH)
        - Evening (17-22): 70% load
        
        Total variation: 70% (0.30 to 1.00)
        """
        if 0 <= hour < 6:
            return 0.30 + np.random.uniform(-0.05, 0.05)
        elif 6 <= hour < 9:
            progress = (hour - 6) / 3
            return 0.30 + (0.30 * progress) + np.random.uniform(-0.05, 0.05)
        elif 9 <= hour < 17:
            return 1.00 + np.random.uniform(-0.1, 0.1)
        elif 17 <= hour < 22:
            progress = (hour - 17) / 5
            return 1.00 - (0.30 * progress) + np.random.uniform(-0.05, 0.05)
        else:
            return 0.30 + np.random.uniform(-0.05, 0.05)
    
    def _simulate_traffic_variation(self):
        """
        Use ENHANCED traffic patterns or real data
        UPDATED: traffic is applied on top of base loads (not overwritten by _update_loads)
        """
        hour = (self.time_step // 40) % 24
        
        if self.enhance_traffic:
            traffic_multiplier = self._get_enhanced_traffic_multiplier(hour)
        elif self.use_real_data and self.traffic_loader:
            traffic_multiplier = self.traffic_loader.get_traffic_multiplier(hour)
        else:
            if 8 <= hour < 18:
                traffic_multiplier = 1.6
            elif 0 <= hour < 6 or 22 <= hour < 24:
                traffic_multiplier = 0.3
            else:
                traffic_multiplier = 1.0

        self.current_traffic_multiplier = traffic_multiplier
        
        # Apply to controller loads (base loads * traffic * noise)
        noise = np.random.uniform(0.95, 1.05, self.num_active)
        self.controller_loads = self.base_controller_loads * traffic_multiplier * noise
        
        # Occasional spikes (2% chance)
        if np.random.random() < 0.02:
            spike_controller = np.random.randint(0, self.num_active)
            spike_magnitude = 1.4 if 9 <= hour < 17 else 1.2
            self.controller_loads[spike_controller] *= spike_magnitude
        
        self.controller_loads = np.clip(self.controller_loads, 0.05, 0.95)
    
    def _get_current_state(self):
        """Get current state"""
        padded_mappings = np.zeros(self.num_switches)
        padded_mappings[:len(self.switch_mappings)] = self.switch_mappings
        
        return np.concatenate([
            self.controller_loads,
            padded_mappings / max(self.num_active, 1),
            self.parked_status,
            [self.current_latency / 100],
            [self.current_energy / 1000]
        ])
    
    def _get_observation(self):
        """Get full observation including history"""
        current = self._get_current_state()
        obs = np.concatenate([current] + list(self.state_history))
        return obs.astype(np.float32)
    
    def step(self, action):
        """Execute action"""
        action_type, entity_id, target_id = self._decode_action(action)
        action_success = self._execute_action(action_type, entity_id, target_id)
        
        self.time_step += 1
        
        # Apply traffic variation
        if self.time_step % 40 == 0:
            self._simulate_traffic_variation()
        
        self.current_latency = self._calculate_latency()
        self.current_energy = self._calculate_energy()
        
        current_state = self._get_current_state()
        self.state_history.append(current_state)
        
        reward = self._calculate_reward(action_success, action_type)
        
        done = self.time_step >= self.max_steps
        terminated = done
        truncated = False
        
        info = {
            'latency': self.current_latency,
            'energy': self.current_energy,
            'load_variance': np.var(self.controller_loads),
            'active_controllers': self.num_active + np.sum(self.parked_status),
            'action_type': action_type,
            'action_success': action_success,
            'hour_of_day': (self.time_step // 40) % 24,
            'topology': self.topology_info['name'] if self.use_real_data else 'abstract'
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _decode_action(self, action):
        """Decode action"""
        if action == 0:
            return 0, 0, 0
        elif action < 2 + self.num_parked:
            return 2, action - 1, 0  # Evoke controller
        elif action == 2 + self.num_parked:
            return 3, 0, 0  # Park
        else:
            migration_action = action - (3 + self.num_parked)
            switch_id = migration_action // self.num_active
            target_controller = migration_action % self.num_active
            
            if switch_id >= self.num_switches:
                return 0, 0, 0
            
            return 1, switch_id, target_controller
    
    def _execute_action(self, action_type, entity_id, target_id):
        """Execute action"""
        if action_type == 0:
            return True
        elif action_type == 1:
            return self._migrate_switch(entity_id, target_id)
        elif action_type == 2:
            return self._evoke_controller(entity_id)
        elif action_type == 3:
            return self._park_controller()
        return False
    
    def _migrate_switch(self, switch_id, target_controller):
        """Migrate switch"""
        if switch_id >= self.num_switches or target_controller >= self.num_active:
            return False
        
        old_controller = self.switch_mappings[switch_id]
        if old_controller == target_controller:
            return False
        
        self.switch_mappings[switch_id] = target_controller

        # Update base loads from new mappings (traffic applied later in step())
        self._update_loads()
        return True
    
    def _evoke_controller(self, parked_id):
        """Evoke controller"""
        if parked_id >= self.num_parked or self.parked_status[parked_id] == 1:
            return False
        
        self.parked_status[parked_id] = 1
        
        hour = (self.time_step // 40) % 24
        max_load = np.max(self.controller_loads)
        
        if (8 <= hour < 18) and max_load > 0.6:
            print(f"  ⚡ EVOKED controller {parked_id} [Day proactive | Hour: {hour:02d}, Max: {max_load:.2f}]")
        elif max_load > 0.8:
            print(f"  🚨 EMERGENCY EVOKE controller {parked_id} [SPIKE | Hour: {hour:02d}, Load: {max_load:.2f}]")
        else:
            print(f"  ⚡ EVOKED controller {parked_id} [Hour: {hour:02d}, Max: {max_load:.2f}]")
        
        return True
    
    def _park_controller(self):
        """Park controller"""
        active_parked = np.where(self.parked_status == 1)[0]
        if len(active_parked) == 0:
            return False
        
        park_id = active_parked[0]
        self.parked_status[park_id] = 0
        
        hour = (self.time_step // 40) % 24
        avg_load = np.mean(self.controller_loads)
        max_load = np.max(self.controller_loads)
        
        if (0 <= hour < 6 or 22 <= hour < 24) and avg_load < 0.5:
            print(f"  💤 PARKED controller {park_id} [Night | Hour: {hour:02d}, Avg: {avg_load:.2f}]")
        elif (6 <= hour < 22) and avg_load < 0.35 and max_load < 0.5:
            print(f"  💤 PARKED controller {park_id} [Day LOW traffic | Hour: {hour:02d}, Avg: {avg_load:.2f}]")
        else:
            print(f"  💤 PARKED controller {park_id} [Hour: {hour:02d}, Avg: {avg_load:.2f}]")
        
        # Update base loads after parking (traffic applied later in step())
        self._update_loads()
        return True
    
    def _update_loads(self):
        """
        Update BASE loads only from switch mapping.
        UPDATED: does NOT overwrite traffic loads anymore.
        """
        base = np.zeros(self.num_active, dtype=float)
        for controller_id in range(self.num_active):
            num_sw = np.sum(self.switch_mappings == controller_id)
            base[controller_id] = num_sw / max(self.num_switches, 1)
        self.base_controller_loads = base
    
    def _calculate_latency(self):
        """Calculate latency"""
        base_latency = 10
        load_variance = np.var(self.controller_loads)
        imbalance_penalty = load_variance * 50
        overload_penalty = np.sum(np.maximum(0, self.controller_loads - 0.8)) * 30
        return base_latency + imbalance_penalty + overload_penalty
    
    def _calculate_energy(self):
        """Calculate energy"""
        num_active = self.num_active + np.sum(self.parked_status)
        active_energy = num_active * self.ACTIVE_POWER
        num_parked = self.num_parked - np.sum(self.parked_status)
        parked_energy = num_parked * self.PARKED_POWER
        return active_energy + parked_energy
    
    def _calculate_reward(self, action_success, action_type):
        """
        Calculate reward - ENHANCED to strongly incentivize proactive behavior
        """
        hour = (self.time_step // 40) % 24
        active_count = self.num_active + np.sum(self.parked_status)
        parked_count = self.total_controllers - active_count
        avg_load = np.mean(self.controller_loads)
        max_load = np.max(self.controller_loads)
        
        reward = 100.0
        
        # ==================== PENALTIES ====================
        
        latency_penalty = (self.current_latency - 10) * 2.0
        reward -= max(0, latency_penalty)
        
        load_imbalance = max_load - np.min(self.controller_loads)
        if load_imbalance > 0.3:
            reward -= load_imbalance * 100
        
        if max_load > 0.8:
            overload_severity = (max_load - 0.8) * 250
            reward -= overload_severity
        
        min_needed_energy = self.num_active * self.ACTIVE_POWER + 20
        energy_waste = self.current_energy - min_needed_energy
        energy_penalty = max(0, energy_waste * 0.2)
        reward -= energy_penalty
        
        # ==================== PROACTIVE BONUSES ====================
        
        if (0 <= hour < 6 or 22 <= hour < 24):
            if parked_count > 0:
                night_parking_bonus = parked_count * 200.0
                if avg_load < 0.5:
                    night_parking_bonus += parked_count * 150.0
                reward += night_parking_bonus
                
                if action_type == 3 and action_success:
                    reward += 120.0
            
            elif parked_count == 0 and avg_load < 0.3:
                reward -= 150.0
        
        elif (9 <= hour < 17):
            if parked_count == 0:
                if max_load > 0.5:
                    reward += 80.0
            elif parked_count > 0:
                if max_load > 0.7:
                    reward -= 200.0
                elif max_load < 0.35 and avg_load < 0.3:
                    day_parking_bonus = parked_count * 100.0
                    reward += day_parking_bonus
                    if action_type == 3 and action_success:
                        reward += 70.0
        
        else:
            if parked_count > 0 and avg_load < 0.5:
                reward += parked_count * 50.0
        
        # ==================== EMERGENCY RESPONSE ====================
        
        if max_load > 0.8:
            if parked_count == 0:
                reward += 120.0
                if action_type == 2 and action_success:
                    reward += 90.0
            else:
                reward -= 180.0
        
        # ==================== MIGRATION BONUS ====================
        
        if action_type == 1 and action_success and load_imbalance < 0.3:
            reward += 8.0
        
        # ==================== ACTION PENALTIES/BONUSES ====================
        
        if not action_success and action_type != 0:
            reward -= 3.0
        
        if action_success and action_type != 0:
            reward += 5.0
        
        return reward
    
    def render(self, mode='human'):
        """Render state"""
        hour = (self.time_step // 40) % 24
        print(f"\n{'='*60}")
        print(f"Time: Step {self.time_step} | Hour {hour:02d}:00")
        if self.use_real_data:
            print(f"Topology: {self.topology_info['name']} ({self.num_switches} switches)")
        print(f"{'='*60}")
        print(f"Loads: {self.controller_loads}")
        print(f"Parked: {self.parked_status} | Active: {self.num_active + int(np.sum(self.parked_status))}")
        print(f"Latency: {self.current_latency:.2f}ms | Energy: {self.current_energy:.2f}W")
        print(f"{'='*60}\n")
    
    def close(self):
        pass


def test_environment():
    """Test the enhanced environment"""
    print("Testing Enhanced ProactiveSDNEnvReal...\n")
    
    # Test with different topologies
    for topo in ['gridnet', 'os3e']:
        print(f"\n{'='*60}")
        print(f"Testing with {topo.upper()} topology")
        print(f"{'='*60}\n")
        
        try:
            env = ProactiveSDNEnvReal(
                topology_name=topo, 
                use_real_data=True,
                enhance_traffic=True
            )
            obs, info = env.reset()
        
            print(f"Observation shape: {obs.shape}")
            print(f"Action space: {env.action_space}")
            print(f"Initial energy: {env.current_energy}W")
            print(f"Switches: {env.num_switches}")
            
            # Simulate a day cycle
            print("\n📊 Simulating 24-hour cycle:")
            for hour in range(24):
                env.time_step = hour * 40
                multiplier = env._get_enhanced_traffic_multiplier(hour)
                status = "🌙 Night" if multiplier < 0.5 else "☀️  Day"
                print(f"{status} Hour {hour:02d}: Traffic = {multiplier:.2f}x")
            
            print(f"\n✅ {topo.upper()} test passed!\n")
            env.close()
            
        except Exception as e:
            print(f"❌ Error testing {topo}: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_environment()
