import json
from datetime import datetime
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "traffic" / "processed"


TOPOLOGY_CONFIGS = {
    "gridnet": {
        "switches": 9,
        "base_scale": 0.85,
        "peak_scale": 1.10,
        "burst_chance": 0.030,
        "burst_strength": (1.20, 1.45),
        "hotspot_fraction": 0.22,
        "hotspot_boost": (1.20, 1.45),
        "noise_scale": 0.10,
        "trend_scale": 0.05,
        "memory": 0.82,
        "event_interval": 420,
    },
    "bellcanada": {
        "switches": 48,
        "base_scale": 0.95,
        "peak_scale": 1.18,
        "burst_chance": 0.045,
        "burst_strength": (1.25, 1.60),
        "hotspot_fraction": 0.18,
        "hotspot_boost": (1.15, 1.50),
        "noise_scale": 0.12,
        "trend_scale": 0.06,
        "memory": 0.85,
        "event_interval": 300,
    },
    "os3e": {
        "switches": 34,
        "base_scale": 1.00,
        "peak_scale": 1.25,
        "burst_chance": 0.055,
        "burst_strength": (1.25, 1.70),
        "hotspot_fraction": 0.20,
        "hotspot_boost": (1.20, 1.55),
        "noise_scale": 0.14,
        "trend_scale": 0.07,
        "memory": 0.84,
        "event_interval": 260,
    },
    "interoute": {
        "switches": 110,
        "base_scale": 1.08,
        "peak_scale": 1.32,
        "burst_chance": 0.065,
        "burst_strength": (1.30, 1.80),
        "hotspot_fraction": 0.14,
        "hotspot_boost": (1.20, 1.65),
        "noise_scale": 0.16,
        "trend_scale": 0.08,
        "memory": 0.86,
        "event_interval": 220,
    },
    "cogentco": {
        "switches": 197,
        "base_scale": 1.12,
        "peak_scale": 1.38,
        "burst_chance": 0.080,
        "burst_strength": (1.35, 1.95),
        "hotspot_fraction": 0.12,
        "hotspot_boost": (1.25, 1.75),
        "noise_scale": 0.18,
        "trend_scale": 0.09,
        "memory": 0.88,
        "event_interval": 180,
    },
}


def get_time_profile(hour):
    """Approximate day-night cycle with stronger peaks and quieter valleys."""
    if 7 <= hour <= 9:
        return 0.72
    if 10 <= hour <= 16:
        return 0.92
    if 17 <= hour <= 20:
        return 0.80
    if 21 <= hour <= 23:
        return 0.48
    return 0.20


def choose_hotspots(num_switches, fraction, rng):
    hotspot_count = max(1, int(num_switches * fraction))
    return rng.choice(num_switches, size=hotspot_count, replace=False)


def create_traffic(num_timesteps=20000, topology="os3e", seed=42):
    topology = topology.lower()
    config = TOPOLOGY_CONFIGS[topology]
    num_switches = config["switches"]
    rng = np.random.default_rng(seed)

    traffic = np.zeros((num_timesteps, num_switches), dtype=np.float32)
    switch_bias = rng.uniform(0.85, 1.15, size=num_switches)
    switch_phase = rng.uniform(0, 2 * np.pi, size=num_switches)
    hotspot_switches = choose_hotspots(num_switches, config["hotspot_fraction"], rng)

    burst_remaining = 0
    burst_multiplier = 1.0
    rebalancing_remaining = 0
    rebalance_targets = np.array([], dtype=int)
    cooldown_remaining = 0

    for t in range(num_timesteps):
        hour = (t // 60) % 24
        base = get_time_profile(hour) * config["base_scale"]

        # Multi-timescale structure so the trace is not just white noise.
        long_wave = 1.0 + config["trend_scale"] * np.sin(2 * np.pi * t / 1440.0)
        short_wave = 1.0 + (config["trend_scale"] * 0.6) * np.sin(
            2 * np.pi * t / 240.0 + switch_phase
        )

        if burst_remaining <= 0 and rng.random() < config["burst_chance"]:
            burst_remaining = int(rng.integers(12, 48))
            burst_multiplier = float(rng.uniform(*config["burst_strength"]))
        if burst_remaining > 0:
            burst_remaining -= 1
        else:
            burst_multiplier = 1.0

        # Periodic calmer windows help create underload opportunities for parking.
        if cooldown_remaining <= 0 and (t % (config["event_interval"] * 2) == 0) and t > 0:
            cooldown_remaining = int(rng.integers(20, 60))
        cooldown_multiplier = 0.65 if cooldown_remaining > 0 else 1.0
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Periodic rebalancing pressure on a subset of switches encourages threshold crossings.
        if rebalancing_remaining <= 0 and (t % config["event_interval"] == 0) and t > 0:
            target_count = max(1, int(num_switches * 0.18))
            rebalancing_remaining = int(rng.integers(25, 90))
            rebalance_targets = rng.choice(num_switches, size=target_count, replace=False)
        hotspot_multiplier = np.ones(num_switches, dtype=np.float32)
        hotspot_multiplier[hotspot_switches] *= rng.uniform(*config["hotspot_boost"])
        if rebalancing_remaining > 0:
            hotspot_multiplier[rebalance_targets] *= rng.uniform(1.25, 1.75)
            rebalancing_remaining -= 1

        raw_load = (
            base
            * long_wave
            * short_wave
            * burst_multiplier
            * cooldown_multiplier
            * switch_bias
            * hotspot_multiplier
        )
        raw_load += rng.normal(0.0, config["noise_scale"], size=num_switches)
        raw_load = np.clip(raw_load, 0.01, 1.0)

        if t > 0:
            traffic[t] = (
                config["memory"] * traffic[t - 1] + (1.0 - config["memory"]) * raw_load
            )
        else:
            traffic[t] = raw_load

        traffic[t] = np.clip(traffic[t], 0.0, 1.0)

    output = DATA_DIR / topology / f"{topology}_synthetic_traffic.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, traffic)

    metadata = {
        "topology": topology,
        "timesteps": num_timesteps,
        "switches": num_switches,
        "mean": float(traffic.mean()),
        "std": float(traffic.std()),
        "p95": float(np.percentile(traffic, 95)),
        "max": float(traffic.max()),
        "seed": seed,
        "config": config,
        "created": datetime.now().isoformat(),
    }
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"✅ {topology.upper():<12} | Shape: {traffic.shape} | "
        f"Mean: {traffic.mean():.3f} | Std: {traffic.std():.3f} | P95: {np.percentile(traffic, 95):.3f}"
    )


print("\n🌐 Creating synthetic traffic...\n")
for topo_name in TOPOLOGY_CONFIGS:
    create_traffic(topology=topo_name, seed=42 + TOPOLOGY_CONFIGS[topo_name]["switches"])
print("\n✅ All done!")
