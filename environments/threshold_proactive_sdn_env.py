"""
Threshold-Based Proactive Load Balancing Environment
Based on research proposal: Master-Slave-Parked Controller Architecture
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx

from utils.topology_loader import TopologyLoader

class ThresholdBasedProactiveSDN(gym.Env):
    """
    Proactive Load Balancing with Master-Slave-Parked Architecture
    
    Key Features:
    1. Multi-level thresholds (UNDERLOAD < 30%, OVERLOAD > 70%)
    2. Proactive actions: Migrate, Evoke, Park
    3. Master controller + Slave controllers + Parked pool
    4. Energy-efficient operation
    """
    
    def __init__(
        self,
        topology_name: str = 'os3e',
        num_slave_controllers: int = 3,
        num_parked_controllers: int = 2,
        underload_threshold: float = 0.30,
        overload_threshold: float = 0.70,
        use_real_data: bool = True
    ):
        super().__init__()

        # Architecture
        self.topology_name = self._normalize_topology_name(topology_name)
        self.num_slave_controllers = num_slave_controllers
        self.num_parked_controllers = num_parked_controllers
        self.total_controllers = 1 + num_slave_controllers + num_parked_controllers
        
        # Thresholds
        self.underload_threshold = underload_threshold
        self.overload_threshold = overload_threshold
        
        # Load topology
        self.topology_loader = TopologyLoader()
        self.controller_host_nodes = []
        self.topology_info = None
        self._load_topology()
        
        # Controller IDs
        self.master_id = 0
        self.slave_ids = list(range(1, 1 + num_slave_controllers))
        self.parked_ids = list(range(1 + num_slave_controllers, self.total_controllers))
        
        # Active/Parked sets
        self.active_slaves = set(self.slave_ids)
        self.parked_slaves = set()
        
        # Mappings
        self.switch_to_controller = {}
        self._initialize_mappings()
        
        # Loads (normalized 0-1)
        self.controller_loads = {i: 0.0 for i in range(self.total_controllers)}
        
        # Energy (Watts)
        self.energy_active = 100.0
        self.energy_parked = 10.0
        self.energy_master = 50.0
        
        # Traffic
        self.use_real_data = use_real_data
        self.time_step = 0
        self.traffic_phase = 0
        
        # Spaces
        self._setup_spaces()

    @staticmethod
    def _normalize_topology_name(topology_name: str) -> str:
        """Normalize topology names so train/test scripts can pass mixed casing safely."""
        if not topology_name:
            return 'os3e'
        return topology_name.strip().lower()
        
    def _load_topology(self):
        """Load network topology"""
        try:
            topology_info, delay_matrix, controller_hosts = (
                self.topology_loader.build_switch_to_controller_delay_matrix(
                    self.topology_name,
                    self.total_controllers,
                )
            )
            self.topology_info = topology_info
            self.topology = topology_info.graph.copy()
            self.num_switches = self.topology.number_of_nodes()
            self.distance_matrix = delay_matrix
            self.controller_host_nodes = controller_hosts
        except Exception:
            topology_sizes = {
                'gridnet': 9, 'bellcanada': 48, 'os3e': 34,
                'interoute': 110, 'cogentco': 197
            }
            self.num_switches = topology_sizes.get(self.topology_name, 34)
            self.topology = nx.star_graph(self.num_switches - 1)

            np.random.seed(42)
            self.distance_matrix = np.random.uniform(
                1,
                20,
                (self.num_switches, self.total_controllers),
            )
            self.controller_host_nodes = list(range(self.total_controllers))
    
    def _initialize_mappings(self):
        """Initialize switch-controller mappings"""
        if not self.active_slaves:
            # Fallback: activate first slave if none active
            self.active_slaves.add(self.slave_ids[0])
        
        for switch_id in range(self.num_switches):
            active_controllers = sorted(self.active_slaves)
            controller_id = min(
                active_controllers,
                key=lambda candidate: self.distance_matrix[switch_id, candidate]
            )
            self.switch_to_controller[switch_id] = controller_id
    
    def _setup_spaces(self):
        """Setup spaces"""
        obs_size = self.total_controllers * 2 + self.num_switches
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        
        max_actions = 1 + self.num_switches + self.num_parked_controllers + self.num_slave_controllers
        self.action_space = spaces.Discrete(max_actions)
    
    def _get_observation(self) -> np.ndarray:
        """Get observation"""
        loads = np.array([self.controller_loads[i] for i in range(self.total_controllers)])
        
        active_status = np.zeros(self.total_controllers)
        active_status[self.master_id] = 1
        for slave_id in self.active_slaves:
            active_status[slave_id] = 1
        
        assignments = np.array([
            self.switch_to_controller[i] / self.total_controllers
            for i in range(self.num_switches)
        ])
        
        return np.concatenate([loads, active_status, assignments]).astype(np.float32)
    
    def _simulate_traffic_variation(self):
        """Simulate day/night traffic cycle"""
        hour = self.traffic_phase % 24
        
        if 9 <= hour <= 17:
            base_load = np.random.uniform(0.7, 1.0)
        elif 18 <= hour <= 22:
            base_load = np.random.uniform(0.4, 0.7)
        else:
            base_load = np.random.uniform(0.1, 0.4)
        
        # Reset loads
        self.controller_loads = {i: 0.0 for i in range(self.total_controllers)}
        
        for switch_id in range(self.num_switches):
            controller_id = self.switch_to_controller[switch_id]
            if controller_id in self.parked_slaves:
                continue
            
            switch_load = base_load * np.random.uniform(0.8, 1.2)
            switch_load = np.clip(switch_load, 0, 1)
            self.controller_loads[controller_id] += switch_load
        
        for controller_id in self.active_slaves:
            num_assigned = sum(1 for s, c in self.switch_to_controller.items() if c == controller_id)
            if num_assigned > 0:
                self.controller_loads[controller_id] /= num_assigned
                self.controller_loads[controller_id] = np.clip(self.controller_loads[controller_id], 0, 1)
    
    def _check_thresholds(self) -> Dict[str, List[int]]:
        """Check threshold violations"""
        overloaded, underloaded, normal = [], [], []
        
        for controller_id in self.active_slaves:
            load = self.controller_loads[controller_id]
            if load > self.overload_threshold:
                overloaded.append(controller_id)
            elif load < self.underload_threshold:
                underloaded.append(controller_id)
            else:
                normal.append(controller_id)
        
        return {'overloaded': overloaded, 'underloaded': underloaded, 'normal': normal}

    def get_valid_actions(self) -> List[int]:
        """Return currently valid action IDs for masking or debugging."""
        valid_actions = [0]
        thresholds = self._check_thresholds()

        if thresholds['overloaded']:
            overloaded_set = set(thresholds['overloaded'])
            for switch_id in range(self.num_switches):
                source_controller = self.switch_to_controller.get(switch_id)
                if source_controller in overloaded_set:
                    valid_actions.append(1 + switch_id)

        parked_controllers = sorted(self.parked_slaves)
        for parked_idx, _ in enumerate(parked_controllers):
            valid_actions.append(1 + self.num_switches + parked_idx)

        if thresholds['underloaded']:
            base_idx = 1 + self.num_switches + self.num_parked_controllers
            valid_actions.extend(base_idx + i for i in range(self.num_slave_controllers))

        return sorted(set(a for a in valid_actions if 0 <= a < self.action_space.n))

    def describe_action(self, action: int) -> str:
        """Return a human-readable label for an action ID in the current state."""
        if action == 0:
            return 'noop'

        if 1 <= action <= self.num_switches:
            switch_id = action - 1
            controller_id = self.switch_to_controller.get(switch_id)
            return f'migrate_switch_{switch_id}_from_ctrl_{controller_id}'

        evoke_base = 1 + self.num_switches
        if evoke_base <= action <= self.num_switches + self.num_parked_controllers:
            parked_idx = action - evoke_base
            parked_controllers = sorted(self.parked_slaves)
            if parked_idx < len(parked_controllers):
                return f'evoke_slot_{parked_idx}_ctrl_{parked_controllers[parked_idx]}'
            return f'evoke_slot_{parked_idx}_unavailable'

        park_base = 1 + self.num_switches + self.num_parked_controllers
        park_idx = action - park_base
        if 0 <= park_idx < self.num_slave_controllers:
            underloaded = self._check_thresholds()['underloaded']
            if underloaded:
                target_idx = min(park_idx, len(underloaded) - 1)
                return f'park_slot_{park_idx}_ctrl_{underloaded[target_idx]}'
            return f'park_slot_{park_idx}_unavailable'

        return f'unknown_action_{action}'
    
    def _migrate_switch(self, switch_id: int, target_controller: int) -> bool:
        """Migrate switch to target controller"""
        if switch_id not in self.switch_to_controller:
            return False
        if target_controller not in self.active_slaves:
            return False

        current_controller = self.switch_to_controller[switch_id]
        if current_controller == target_controller:
            return False

        current_load = self.controller_loads[target_controller]
        additional_load = 1.0 / self.num_switches
        
        if current_load + additional_load > self.overload_threshold:
            return False
        
        self.switch_to_controller[switch_id] = target_controller
        return True
    
    def _evoke_controller(self, parked_id: int) -> bool:
        """Wake parked controller"""
        if parked_id not in self.parked_slaves:
            return False
        
        self.parked_slaves.remove(parked_id)
        self.active_slaves.add(parked_id)
        self.controller_loads[parked_id] = 0.0
        return True
    
    def _park_controller(self, controller_id: int) -> bool:
        """
        Park an underloaded controller
        
        🔥 FIXED: Properly checks for other active controllers before parking
        """
        # Basic validation
        if controller_id not in self.active_slaves:
            return False  # Already parked or invalid
        
        if controller_id == self.master_id:
            return False  # Can't park master
        
        # 🔥 CRITICAL FIX: Must have at least one other active controller
        other_active = [c for c in self.active_slaves if c != controller_id]
        if not other_active:
            return False  # Can't park the last active controller!
        
        # Check if underloaded enough to park
        if self.controller_loads[controller_id] >= self.underload_threshold:
            return False  # Not underloaded enough
        
        # Find switches that need to be migrated
        switches_to_migrate = [s for s, c in self.switch_to_controller.items() 
                               if c == controller_id]
        
        # If no switches, safe to park immediately
        if not switches_to_migrate:
            self.active_slaves.remove(controller_id)
            self.parked_slaves.add(controller_id)
            self.controller_loads[controller_id] = 0.0
            return True
        
        # Try to migrate all switches to other active controllers
        for switch_id in switches_to_migrate:
            # Find best target: least loaded among other active controllers
            target = min(other_active, key=lambda c: self.controller_loads[c])
            
            if not self._migrate_switch(switch_id, target):
                return False  # Migration failed, can't park
        
        # All switches successfully migrated, now park the controller
        self.active_slaves.remove(controller_id)
        self.parked_slaves.add(controller_id)
        self.controller_loads[controller_id] = 0.0
        
        return True
    
    def step(self, action: int):
        """Execute step"""
        self.time_step += 1
        if self.time_step % 50 == 0:
            self.traffic_phase = (self.traffic_phase + 1) % 24
            self._simulate_traffic_variation()
        
        action_type, success = self._execute_action(action)
        reward, info = self._calculate_reward(action_type, success)
        
        terminated = False
        truncated = self.time_step >= 1000
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> Tuple[str, bool]:
        """Execute action"""
        thresholds = self._check_thresholds()
        
        if action == 0:
            return 'noop', True
        elif action <= self.num_switches:
            switch_id = action - 1
            if not self.active_slaves:
                return 'migrate_failed', False
            if not thresholds['overloaded']:
                return 'migrate_failed', False
            source_controller = self.switch_to_controller.get(switch_id)
            if source_controller not in thresholds['overloaded']:
                return 'migrate_failed', False
            target = min(self.active_slaves, key=lambda c: self.controller_loads[c])
            success = self._migrate_switch(switch_id, target)
            return 'migrate' if success else 'migrate_failed', success
        elif action <= self.num_switches + self.num_parked_controllers:
            parked_idx = action - self.num_switches - 1
            parked_controllers = sorted(self.parked_slaves)
            if parked_idx < len(parked_controllers):
                parked_id = parked_controllers[parked_idx]
                success = self._evoke_controller(parked_id)
                return 'evoke' if success else 'evoke_failed', success
            return 'evoke_failed', False
        else:
            underloaded = thresholds['underloaded']
            if underloaded:
                controller_id = underloaded[0]
                success = self._park_controller(controller_id)
                return 'park' if success else 'park_failed', success
            return 'park_failed', False
    
    def _calculate_reward(self, action_type: str, success: bool) -> Tuple[float, Dict]:
        """Calculate multi-objective reward"""
        latency = self._calculate_latency()
        worst_case_latency = self._calculate_worst_case_latency()
        energy = self._calculate_energy()
        load_variance = self._calculate_load_variance()
        load_balance_index = self._calculate_load_balance_index()
        
        norm_latency = latency / 20.0
        norm_energy = energy / 1000.0
        norm_load_var = load_variance / 1.0
        
        reward = -0.3 * norm_latency - 0.5 * norm_energy - 0.2 * norm_load_var
        
        if action_type == 'park' and success:
            reward += 500.0
        elif action_type == 'evoke' and success:
            reward += 300.0
        elif action_type == 'migrate' and success:
            reward += 10.0
        
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
            'active_controllers': len(self.active_slaves),
            'parked_controllers': len(self.parked_slaves),
            'action_type': action_type,
            'action_success': success,
            'overloaded_count': len(self._check_thresholds()['overloaded']),
            'underloaded_count': len(self._check_thresholds()['underloaded'])
        }
        
        return reward, info
    
    def _calculate_latency(self) -> float:
        """Calculate controller-switch average latency in milliseconds."""
        total = 0.0
        for switch_id, controller_id in self.switch_to_controller.items():
            distance = self.distance_matrix[switch_id, controller_id]
            total += distance
        return total / self.num_switches

    def _calculate_worst_case_latency(self) -> float:
        """Calculate the worst controller-switch latency in milliseconds."""
        if not self.switch_to_controller:
            return 0.0
        return max(
            self.distance_matrix[switch_id, controller_id]
            for switch_id, controller_id in self.switch_to_controller.items()
        )
    
    def _calculate_energy(self) -> float:
        """Calculate total energy"""
        energy = self.energy_master
        energy += len(self.active_slaves) * self.energy_active
        energy += len(self.parked_slaves) * self.energy_parked
        return energy
    
    def _calculate_load_variance(self) -> float:
        """Calculate load variance"""
        if not self.active_slaves:
            return 0.0
        loads = [self.controller_loads[c] for c in self.active_slaves]
        return np.var(loads)

    def _calculate_load_balance_index(self) -> float:
        """
        Paper-style load-balance proxy.

        Inference: use max-load / mean-load across assigned active controllers.
        This is far more stable than max/min in this environment while still reflecting
        how concentrated the controller burden is.
        """
        if not self.active_slaves:
            return 0.0
        assigned_loads = []
        for controller_id in self.active_slaves:
            num_assigned = sum(
                1 for assigned_controller in self.switch_to_controller.values()
                if assigned_controller == controller_id
            )
            if num_assigned > 0:
                assigned_loads.append(self.controller_loads[controller_id] * num_assigned)

        if len(assigned_loads) < 2:
            return 1.0

        max_load = max(assigned_loads)
        mean_load = float(np.mean(assigned_loads))
        if max_load <= 0:
            return 1.0
        return max_load / max(mean_load, 1e-3)
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # 🔥 CRITICAL FIX: Initialize with parked controllers properly
        self.active_slaves = set(self.slave_ids)
        self.parked_slaves = set(self.parked_ids)  # ← FIXED! Was set()
        self._initialize_mappings()
        self.controller_loads = {i: 0.0 for i in range(self.total_controllers)}
        self.time_step = 0
        self.traffic_phase = np.random.randint(0, 24)
        self._simulate_traffic_variation()
        
        return self._get_observation(), {}
    
    def render(self):
        """Render state"""
        thresholds = self._check_thresholds()
        print(f"\nStep {self.time_step} | Hour {self.traffic_phase}")
        print(f"Active: {len(self.active_slaves)} | Parked: {len(self.parked_slaves)}")
        print(f"Overloaded: {thresholds['overloaded']}")
        print(f"Underloaded: {thresholds['underloaded']}")
