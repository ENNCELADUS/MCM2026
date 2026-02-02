"""Random generators for stochastic simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RiskParams:
    """Container for risk simulation parameters."""

    # Elevator efficiency: eta ~ Beta(alpha, beta)
    eta_alpha: float = 17.0
    eta_beta: float = 3.0

    # Rocket reliability: X ~ Bernoulli(success_rate)
    rocket_success_rate: float = 0.985

    # Monte Carlo runs
    n_runs: int = 1000

    @classmethod
    def from_config(cls, constants: dict[str, Any]) -> "RiskParams":
        """Load from constants.yaml risk section."""
        risk = constants.get("risk", {})
        return cls(
            eta_alpha=float(risk.get("elevator", {}).get("eta_alpha", 17.0)),
            eta_beta=float(risk.get("elevator", {}).get("eta_beta", 3.0)),
            rocket_success_rate=float(
                risk.get("rocket", {}).get("success_rate", 0.985)
            ),
            n_runs=int(risk.get("monte_carlo_runs", 1000)),
        )


class RiskSampler:
    """Generates stochastic samples for simulation."""

    def __init__(self, params: RiskParams, seed: int | None = None):
        self.params = params
        self.rng = np.random.default_rng(seed)

    def sample_elevator_efficiency(self) -> float:
        """Sample elevator efficiency factor from Beta distribution.

        Returns:
            eta in [0, 1], expected value ~0.85.
        """
        return self.rng.beta(self.params.eta_alpha, self.params.eta_beta)

    def sample_rocket_success(self, n_launches: int) -> int:
        """Sample number of successful rocket launches from Binomial.

        Args:
            n_launches: Number of planned launches.

        Returns:
            Number of successful launches.
        """
        if n_launches <= 0:
            return 0
        return self.rng.binomial(n_launches, self.params.rocket_success_rate)

    def sample_rocket_failures(self, n_launches: int) -> int:
        """Get number of failed launches."""
        return n_launches - self.sample_rocket_success(n_launches)
