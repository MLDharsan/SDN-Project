"""Load and validate the Mininet experiment configuration."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from mininet_integration._paths import DEFAULT_CONFIG, resolve


@dataclass
class ExperimentConfig:
    raw: Dict[str, Any]
    profile_name: str
    path: Path

    @property
    def topology(self) -> str:
        return self.raw["topology"]

    @property
    def num_slave_controllers(self) -> int:
        return int(self.raw["num_slave_controllers"])

    @property
    def num_parked_controllers(self) -> int:
        return int(self.raw["num_parked_controllers"])

    @property
    def total_controllers(self) -> int:
        return 1 + self.num_slave_controllers + self.num_parked_controllers

    @property
    def profile(self) -> Dict[str, Any]:
        return self.raw["profiles"][self.profile_name]

    @property
    def traffic_path(self) -> Path:
        return resolve(self.profile["traffic_path"])

    def model_path(self, observation_mode: str) -> Path:
        key = "predictive_model" if observation_mode == "predictive" else "current_model"
        return resolve(self.profile[key])

    @property
    def gru_model_path(self) -> Path:
        return resolve(self.profile["gru_model"])

    def section(self, name: str) -> Dict[str, Any]:
        return self.raw[name]

    def to_json(self) -> Dict[str, Any]:
        data = copy.deepcopy(self.raw)
        data["_selected_profile"] = self.profile_name
        data["_config_path"] = str(self.path)
        return data


def load_config(path: str | Path | None = None, profile: str | None = None) -> ExperimentConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG
    config_path = resolve(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    profile_name = profile or raw.get("default_profile", "synthetic")
    if profile_name not in raw.get("profiles", {}):
        raise ValueError(
            f"Unknown profile '{profile_name}'. Available: {sorted(raw.get('profiles', {}))}"
        )
    for section in ("network", "traffic", "timing", "controllers", "probes", "run"):
        if section not in raw:
            raise ValueError(f"Config is missing section '{section}'")

    timing = raw["timing"]
    if int(timing["interval_s"]) != timing["interval_s"] or timing["interval_s"] < 2:
        raise ValueError("timing.interval_s must be an integer >= 2 (iperf3 -t is integral)")
    if timing["measure_warmup_s"] + timing["measure_margin_s"] >= timing["interval_s"]:
        raise ValueError("measure_warmup_s + measure_margin_s must be below interval_s")
    if raw["traffic"]["max_rate_mbps"] <= 0:
        raise ValueError("traffic.max_rate_mbps must be positive")
    return ExperimentConfig(raw=raw, profile_name=profile_name, path=config_path)
