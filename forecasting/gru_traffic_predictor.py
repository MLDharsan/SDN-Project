"""GRU traffic forecasting model and inference wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class GRUTrafficNetwork(nn.Module):
    """Predict per-switch traffic at one configurable future horizon."""

    def __init__(
        self,
        num_switches: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_switches = num_switches
        self.gru = nn.GRU(
            input_size=num_switches,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_switches),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        sequence, _hidden = self.gru(history)
        return self.output(sequence[:, -1])


@dataclass(frozen=True)
class ForecastConfig:
    num_switches: int
    history_length: int
    forecast_horizon: int
    hidden_dim: int
    num_layers: int
    dropout: float


class GRUTrafficForecaster:
    """Frozen NumPy-facing wrapper around a trained GRU checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"GRU checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        self.config = ForecastConfig(
            num_switches=int(config["num_switches"]),
            history_length=int(config["history_length"]),
            forecast_horizon=int(config["forecast_horizon"]),
            hidden_dim=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
        )
        self.device = torch.device(device)
        self.mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
        self.std = np.maximum(self.std, 1e-6)
        self.model = GRUTrafficNetwork(
            num_switches=self.config.num_switches,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.validation_metrics = checkpoint.get("validation_metrics", {})

    @property
    def history_length(self) -> int:
        return self.config.history_length

    @property
    def forecast_horizon(self) -> int:
        return self.config.forecast_horizon

    @torch.no_grad()
    def predict(self, history: np.ndarray) -> np.ndarray:
        values = np.asarray(history, dtype=np.float32)
        expected = (self.history_length, self.config.num_switches)
        if values.shape != expected:
            raise ValueError(f"Expected traffic history shape {expected}, got {values.shape}")
        normalized = (values - self.mean) / self.std
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        prediction = self.model(tensor).squeeze(0).cpu().numpy()
        prediction = prediction * self.std + self.mean
        return np.clip(prediction, 0.0, 1.0).astype(np.float32)
