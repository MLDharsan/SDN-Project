"""Regression tests for the shared training and fair-evaluation core."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from environments.rainbow_dqn_model import RainbowDQN
from environments.threshold_proactive_sdn_env import ThresholdBasedProactiveSDN
from experiments.evaluate_fair_baselines import select_threshold_action
from experiments.preprocessing.split_traffic_temporal import split_temporally


class ConstantPredictor:
    history_length = 4
    forecast_horizon = 2

    def __init__(self, num_switches, value):
        self.config = SimpleNamespace(num_switches=num_switches)
        self.value = value

    def predict(self, history):
        return np.full(history.shape[1], self.value, dtype=np.float32)


class EnvironmentTests(unittest.TestCase):
    def make_env(self, value=0.1, rows=20):
        trace = np.full((rows, 9), value, dtype=np.float32)
        return ThresholdBasedProactiveSDN(
            topology_name="gridnet", traffic_data=trace, max_steps=10
        )

    def test_trace_advances_once_per_step(self):
        env = self.make_env()
        env.reset(seed=42, options={"traffic_start_index": 3})
        self.assertEqual(env.traffic_index, 4)
        env.step(0)
        self.assertEqual(env.traffic_index, 5)

    def test_action_slots_are_stable_and_threshold_policy_is_valid(self):
        env = self.make_env(value=0.1)
        env.reset(seed=42)
        action = select_threshold_action(env)
        self.assertIn(action, env.get_valid_actions())
        self.assertTrue(env.describe_action(action).startswith("park_ctrl_"))

    def test_reward_is_inside_declared_range(self):
        env = self.make_env(value=0.1)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(select_threshold_action(env))
        self.assertGreaterEqual(reward, -2.0)
        self.assertLessEqual(reward, 1.0)

    def test_predictive_mode_expands_state_without_changing_current_mode(self):
        trace = np.full((20, 9), 0.5, dtype=np.float32)
        current = ThresholdBasedProactiveSDN("gridnet", traffic_data=trace, max_steps=10)
        predictive = ThresholdBasedProactiveSDN(
            "gridnet",
            traffic_data=trace,
            max_steps=10,
            observation_mode="predictive",
            traffic_predictor=ConstantPredictor(9, 0.9),
        )
        current_state, _ = current.reset(seed=42)
        predictive_state, _ = predictive.reset(seed=42)
        self.assertEqual(current_state.shape, (21,))
        self.assertEqual(predictive_state.shape, (36,))

    def test_forecasted_overload_allows_early_evoke(self):
        trace = np.full((20, 9), 0.5, dtype=np.float32)
        env = ThresholdBasedProactiveSDN(
            "gridnet",
            traffic_data=trace,
            max_steps=10,
            observation_mode="predictive",
            traffic_predictor=ConstantPredictor(9, 0.9),
        )
        env.reset(seed=42)
        self.assertEqual(env._check_thresholds()["overloaded"], [])
        evoke_action = env.action_for_evoke(min(env.parked_slaves))
        self.assertIn(evoke_action, env.get_valid_actions())
        _state, _reward, _terminated, _truncated, info = env.step(evoke_action)
        self.assertEqual(info["action_type"], "evoke")
        self.assertTrue(info["action_success"])

    def test_temporal_split_preserves_order_and_boundaries(self):
        data = np.arange(100, dtype=np.float32).reshape(50, 2)
        train, validation, test, metadata = split_temporally(data)
        self.assertTrue(np.array_equal(train[-1], data[34]))
        self.assertTrue(np.array_equal(validation[0], data[35]))
        self.assertTrue(np.array_equal(test[0], data[42]))
        self.assertEqual(metadata["train_range"], [0, 35])


class RainbowTests(unittest.TestCase):
    def make_agent(self, n_step=3):
        return RainbowDQN(
            state_dim=4,
            action_dim=2,
            hidden_dim=8,
            n_step=n_step,
            buffer_size=20,
            batch_size=2,
            device="cpu",
        )

    def test_c51_projection_preserves_probability_mass_on_exact_atoms(self):
        agent = self.make_agent(n_step=1)
        rewards = torch.tensor([0.0])
        dones = torch.tensor([1.0])
        distribution = torch.full((1, 51), 1 / 51)
        projected = agent._project_distribution(rewards, dones, distribution)
        self.assertTrue(torch.allclose(projected.sum(dim=1), torch.ones(1), atol=1e-6))

    def test_terminal_transition_flushes_n_step_prefixes(self):
        agent = self.make_agent(n_step=3)
        state = np.zeros(4, dtype=np.float32)
        agent.push_n_step(state, 0, 1.0, state, False)
        agent.push_n_step(state, 0, 1.0, state, False)
        agent.push_n_step(state, 0, 1.0, state, True)
        self.assertEqual(len(agent.memory), 3)
        self.assertEqual(len(agent.n_step_buffer), 0)


if __name__ == "__main__":
    unittest.main()
