#!/usr/bin/env python3
"""Train the proactive Rainbow DQN on a processed traffic trace."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environments.rainbow_dqn_model import RainbowDQN  # noqa: E402
from environments.threshold_proactive_sdn_env import ThresholdBasedProactiveSDN  # noqa: E402
from forecasting.gru_traffic_predictor import GRUTrafficForecaster  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


TOPOLOGY_SIZES = {
    "gridnet": 9,
    "bellcanada": 48,
    "os3e": 34,
    "interoute": 110,
    "cogentco": 197,
}


def topology_hyperparameters(topology: str) -> dict:
    if topology in {"interoute", "cogentco"}:
        return {"hidden_dim": 512, "lr": 5e-5, "n_step": 5, "buffer_size": 100_000, "batch_size": 256}
    if topology in {"bellcanada", "os3e"}:
        return {"hidden_dim": 256, "lr": 1e-4, "n_step": 3, "buffer_size": 50_000, "batch_size": 128}
    return {"hidden_dim": 128, "lr": 2e-4, "n_step": 3, "buffer_size": 30_000, "batch_size": 64}


def checkpoint_payload(agent: RainbowDQN, state_dim: int, action_dim: int, config: dict) -> dict:
    return {
        "online_net": agent.online_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hyperparameters": config,
        "reward_version": "normalized_multi_objective_v2",
        "action_schema_version": "stable_controller_slots_v2",
    }


def train_rainbow_fixed_rewards(
    topology: str = "gridnet",
    timesteps: int = 20_000,
    mode: str = "proactive",
    train_freq: int = 4,
    mask_invalid_actions: bool = True,
    traffic_path: str | None = None,
    seed: int = 42,
    v_min: float = -10.0,
    v_max: float = 10.0,
    epsilon_start: float = 0.10,
    epsilon_end: float = 0.01,
    observation_mode: str = "current",
    gru_model_path: str | None = None,
    output_dir: str | None = None,
) -> tuple[RainbowDQN, str]:
    topology = topology.strip().lower()
    if topology not in TOPOLOGY_SIZES:
        raise ValueError(f"Unsupported topology: {topology}")
    if v_min >= v_max:
        raise ValueError("v_min must be lower than v_max")
    if observation_mode not in {"current", "predictive"}:
        raise ValueError("observation_mode must be 'current' or 'predictive'")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    trace_path = Path(traffic_path) if traffic_path else (
        PROJECT_ROOT / "data" / "traffic" / "processed" / topology /
        f"{topology}_synthetic_traffic.npy"
    )
    if not trace_path.is_absolute():
        trace_path = PROJECT_ROOT / trace_path
    if not trace_path.exists():
        raise FileNotFoundError(f"Traffic trace not found: {trace_path}")
    traffic_data = np.load(trace_path)

    predictor = None
    resolved_gru_path = None
    if observation_mode == "predictive":
        resolved_gru_path = Path(gru_model_path) if gru_model_path else (
            PROJECT_ROOT / "models" / "forecasting" / f"LATEST_gru_{topology}.pth"
        )
        if not resolved_gru_path.is_absolute():
            resolved_gru_path = PROJECT_ROOT / resolved_gru_path
        predictor = GRUTrafficForecaster(resolved_gru_path, device="cpu")

    env = ThresholdBasedProactiveSDN(
        topology_name=topology,
        num_slave_controllers=3,
        num_parked_controllers=2,
        traffic_data=traffic_data,
        traffic_update_interval=1,
        max_steps=1000,
        observation_mode=observation_mode,
        traffic_predictor=predictor,
    )
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    hp = topology_hyperparameters(topology)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        **hp,
        "gamma": 0.99,
        "n_atoms": 51,
        "v_min": v_min,
        "v_max": v_max,
        "seed": seed,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "traffic_update_interval": 1,
        "mask_invalid_actions": mask_invalid_actions,
        "observation_mode": observation_mode,
        "gru_model_path": str(resolved_gru_path.resolve()) if resolved_gru_path else None,
        "forecast_history_length": predictor.history_length if predictor else None,
        "forecast_horizon": predictor.forecast_horizon if predictor else None,
    }
    agent = RainbowDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hp["hidden_dim"],
        lr=hp["lr"],
        gamma=0.99,
        n_step=hp["n_step"],
        buffer_size=hp["buffer_size"],
        batch_size=hp["batch_size"],
        n_atoms=51,
        v_min=v_min,
        v_max=v_max,
        device=device,
    )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = None
    if SummaryWriter is not None:
        log_dir = PROJECT_ROOT / "logs" / "proactive_dqn_tensorboard" / f"rainbow_{observation_mode}_{mode}_{topology}_{run_timestamp}"
        writer = SummaryWriter(log_dir=str(log_dir))

    def reset_episode(episode: int):
        start = (episode * env.max_steps) % len(traffic_data)
        agent.reset_n_step()
        return env.reset(seed=seed + episode, options={"traffic_start_index": start})[0]

    def greedy_valid_action(state: np.ndarray) -> int:
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            q_values = agent.online_net.get_q_values(tensor).squeeze(0)
            valid = env.get_valid_actions()
            masked = torch.full_like(q_values, -torch.inf)
            masked[valid] = q_values[valid]
            return int(masked.argmax().item())

    def epsilon_at(step: int) -> float:
        decay = max(1.0, timesteps * 0.6)
        return float(epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / decay))

    print("=" * 72)
    print("RAINBOW DQN TRAINING - CONSISTENT ENVIRONMENT V3")
    print("=" * 72)
    print(f"Topology: {topology} ({TOPOLOGY_SIZES[topology]} switches)")
    print(f"Traffic: {trace_path} | shape={traffic_data.shape} | one row/environment step")
    print(f"Observation mode: {observation_mode}")
    if predictor:
        print(
            f"GRU: {resolved_gru_path} | history={predictor.history_length} | "
            f"horizon={predictor.forecast_horizon}"
        )
    print(f"Reward range: [-2, 1] | C51 support: [{v_min:g}, {v_max:g}] | seed={seed}")
    print(f"State/action dimensions: {state_dim}/{action_dim} | device={device}")

    started = time.time()
    state = reset_episode(0)
    episode = 0
    episode_reward = 0.0
    episode_rewards: list[float] = []
    losses: list[float] = []
    latencies: list[float] = []
    energies: list[float] = []
    load_balances: list[float] = []
    actions: Counter[str] = Counter()

    for step in range(timesteps):
        epsilon = epsilon_at(step)
        if rng.random() < epsilon:
            candidates = env.get_valid_actions() if mask_invalid_actions else list(range(action_dim))
            action = int(rng.choice(candidates))
        elif mask_invalid_actions:
            action = greedy_valid_action(state)
        else:
            action = agent.select_action(state, training=True)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.push_n_step(state, action, reward, next_state, done)

        if step >= 1000 and step % max(1, train_freq) == 0 and len(agent.memory) >= agent.batch_size:
            loss = agent.train_step()
            if loss is not None:
                losses.append(float(loss))
                if writer:
                    writer.add_scalar("train/loss", loss, step)
        if step > 0 and step % 1000 == 0:
            agent.update_target_network()

        episode_reward += float(reward)
        latencies.append(float(info["cs_avg_latency"]))
        energies.append(float(info["energy"]))
        load_balances.append(float(info["load_balance_index"]))
        if info["action_success"]:
            actions[info["action_type"]] += 1
        if writer:
            writer.add_scalar("train/reward", reward, step)
            writer.add_scalar("train/latency", info["cs_avg_latency"], step)
            writer.add_scalar("train/energy", info["energy"], step)

        state = next_state
        if done:
            episode_rewards.append(episode_reward)
            if writer:
                writer.add_scalar("episode/reward", episode_reward, episode)
            episode += 1
            episode_reward = 0.0
            state = reset_episode(episode)

        if (step + 1) % max(1, timesteps // 10) == 0:
            recent = np.mean(episode_rewards[-10:]) if episode_rewards else episode_reward
            print(f"Step {step + 1:>8,}/{timesteps:,} | episodes={episode:>3} | recent reward={recent:8.2f} | epsilon={epsilon:.3f}")

    if episode_reward or not episode_rewards:
        episode_rewards.append(episode_reward)

    models_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "models"
    if not models_dir.is_absolute():
        models_dir = PROJECT_ROOT / models_dir
    filename_prefix = "rainbow"
    latest_prefix = "LATEST_rainbow"
    if observation_mode == "predictive" and output_dir is None:
        models_dir = models_dir / "predictive"
    if observation_mode == "predictive":
        filename_prefix = "gru_rainbow"
        latest_prefix = "LATEST_gru_rainbow"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{filename_prefix}_{mode}_{topology}_{timesteps}steps_{run_timestamp}.pth"
    latest_path = models_dir / f"{latest_prefix}_{mode}_{topology}.pth"
    payload = checkpoint_payload(agent, state_dim, action_dim, config)
    torch.save(payload, model_path)
    # Keep one physical checkpoint. LATEST is a relative symlink so large
    # topologies do not consume hundreds of extra megabytes per run.
    latest_path.unlink(missing_ok=True)
    latest_path.symlink_to(model_path.name)

    metadata = {
        "topology": topology,
        "timesteps": timesteps,
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "traffic_path": str(trace_path),
        "traffic_shape": list(traffic_data.shape),
        "observation_mode": observation_mode,
        "gru_model_path": str(resolved_gru_path.resolve()) if resolved_gru_path else None,
        "reward_function": "normalized_multi_objective_v2",
        "action_schema_version": "stable_controller_slots_v2",
        "episodes": len(episode_rewards),
        "avg_episode_reward": float(np.mean(episode_rewards)),
        "avg_latency": float(np.mean(latencies)),
        "avg_energy": float(np.mean(energies)),
        "paper_load_balance_std": float(np.mean(load_balances)),
        "avg_loss": float(np.mean(losses)) if losses else None,
        "successful_actions": dict(actions),
        "training_time_seconds": time.time() - started,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hyperparameters": config,
    }
    metadata_path = model_path.with_name(model_path.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    if writer:
        writer.close()
    env.close()

    print("\nTRAINING COMPLETE")
    print(f"Model: {model_path}")
    print(f"Latest: {latest_path}")
    print(f"Metadata: {metadata_path}")
    return agent, str(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train proactive Rainbow DQN consistently.")
    parser.add_argument("--topology", default="gridnet", choices=list(TOPOLOGY_SIZES))
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--mode", default="proactive", choices=["proactive", "reactive"])
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--traffic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=0.10)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument(
        "--observation-mode", choices=["current", "predictive"], default="current"
    )
    parser.add_argument("--gru-model", help="GRU checkpoint used in predictive mode")
    parser.add_argument("--output-dir", help="Override checkpoint output directory")
    parser.add_argument("--no-mask-invalid-actions", action="store_true")
    args = parser.parse_args()
    train_rainbow_fixed_rewards(
        topology=args.topology,
        timesteps=args.timesteps,
        mode=args.mode,
        train_freq=args.train_freq,
        mask_invalid_actions=not args.no_mask_invalid_actions,
        traffic_path=args.traffic,
        seed=args.seed,
        v_min=args.v_min,
        v_max=args.v_max,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        observation_mode=args.observation_mode,
        gru_model_path=args.gru_model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
