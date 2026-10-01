"""Generative model for A/B conversion experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExperimentParams:
    """Known data-generating process for the lab."""

    p_control: float
    p_treatment: float
    n_control: int
    n_treatment: int

    @property
    def absolute_uplift(self) -> float:
        return self.p_treatment - self.p_control

    @property
    def relative_uplift(self) -> float:
        if self.p_control <= 0:
            raise ValueError("p_control must be positive for relative uplift")
        return (self.p_treatment - self.p_control) / self.p_control

    @property
    def n_total(self) -> int:
        return self.n_control + self.n_treatment


@dataclass(frozen=True)
class ExperimentResult:
    """One realized A/B experiment (binary outcomes)."""

    control: np.ndarray
    treatment: np.ndarray

    @property
    def rate_control(self) -> float:
        return float(np.mean(self.control))

    @property
    def rate_treatment(self) -> float:
        return float(np.mean(self.treatment))

    @property
    def absolute_uplift(self) -> float:
        return self.rate_treatment - self.rate_control

    @property
    def relative_uplift(self) -> float:
        rc = self.rate_control
        if rc <= 0:
            return float("nan")
        return (self.rate_treatment - rc) / rc


def simulate_experiment(params: ExperimentParams, rng: np.random.Generator) -> ExperimentResult:
    """Draw one Bernoulli A/B experiment under unequal allocation."""
    control = rng.binomial(1, params.p_control, size=params.n_control).astype(np.float64)
    treatment = rng.binomial(1, params.p_treatment, size=params.n_treatment).astype(np.float64)
    return ExperimentResult(control=control, treatment=treatment)


def simulate_many(
    params: ExperimentParams,
    n_experiments: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Monte Carlo sampling distribution of the relative-uplift estimator.

    Returns shape (n_experiments,) of relative uplifts. NaNs are possible if a
    simulated control arm has zero conversions (rare at these rates).
    """
    if n_experiments < 1:
        raise ValueError("n_experiments must be >= 1")
    # Vectorized: each row is one experiment's successes via Binomial counts
    succ_c = rng.binomial(params.n_control, params.p_control, size=n_experiments)
    succ_t = rng.binomial(params.n_treatment, params.p_treatment, size=n_experiments)
    rate_c = succ_c / params.n_control
    rate_t = succ_t / params.n_treatment
    with np.errstate(divide="ignore", invalid="ignore"):
        uplift = (rate_t - rate_c) / rate_c
    return uplift.astype(np.float64)


def scale_params(params: ExperimentParams, factor: float) -> ExperimentParams:
    """Scale both arm sizes by ``factor`` (at least 1 user per arm)."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    return ExperimentParams(
        p_control=params.p_control,
        p_treatment=params.p_treatment,
        n_control=max(1, int(round(params.n_control * factor))),
        n_treatment=max(1, int(round(params.n_treatment * factor))),
    )
