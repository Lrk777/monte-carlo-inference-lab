"""Reproducibility and unit tests for the Monte Carlo lab."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.estimate import (
    analytic_relative_uplift_se,
    bootstrap_relative_uplift,
    normal_approx_ci,
    params_from_config,
    relative_uplift_from_arrays,
    run_pipeline,
    summarize_sampling_distribution,
)
from src.simulate import ExperimentParams, scale_params, simulate_experiment, simulate_many

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"
SNAPSHOT = ROOT / "reports" / "snapshots" / "expected_metrics.json"


def test_relative_uplift_truth():
    p = ExperimentParams(0.08, 0.095, 5000, 2000)
    assert p.absolute_uplift == pytest.approx(0.015)
    assert p.relative_uplift == pytest.approx(0.015 / 0.08)


def test_simulate_experiment_shapes_and_seed():
    params = ExperimentParams(0.1, 0.12, 100, 80)
    a = simulate_experiment(params, np.random.default_rng(7))
    b = simulate_experiment(params, np.random.default_rng(7))
    assert a.control.shape == (100,)
    assert a.treatment.shape == (80,)
    np.testing.assert_array_equal(a.control, b.control)
    np.testing.assert_array_equal(a.treatment, b.treatment)


def test_simulate_many_reproducible():
    params = ExperimentParams(0.08, 0.095, 500, 200)
    x = simulate_many(params, 200, np.random.default_rng(42))
    y = simulate_many(params, 200, np.random.default_rng(42))
    np.testing.assert_allclose(x, y)
    assert np.isfinite(x).mean() > 0.95


def test_bootstrap_contains_point_estimate():
    params = ExperimentParams(0.08, 0.095, 2000, 1000)
    exp = simulate_experiment(params, np.random.default_rng(1))
    ci, samples = bootstrap_relative_uplift(exp, 500, np.random.default_rng(2), 0.95)
    assert ci.lower <= ci.point <= ci.upper
    assert np.isfinite(samples).sum() > 100


def test_normal_ci_symmetric_around_point():
    params = ExperimentParams(0.08, 0.095, 2000, 1000)
    exp = simulate_experiment(params, np.random.default_rng(3))
    ci = normal_approx_ci(exp, 0.95)
    mid = 0.5 * (ci.lower + ci.upper)
    assert mid == pytest.approx(ci.point, abs=1e-12)


def test_delta_method_se_positive():
    se = analytic_relative_uplift_se(0.08, 0.095, 5000, 2000)
    assert se > 0


def test_scale_params():
    base = ExperimentParams(0.1, 0.2, 100, 50)
    doubled = scale_params(base, 2.0)
    assert doubled.n_control == 200
    assert doubled.n_treatment == 100


def test_relative_uplift_nan_when_control_zero():
    assert np.isnan(relative_uplift_from_arrays(np.zeros(10), np.ones(10)))


def test_summarize_bias_zero_when_centered():
    truth = 0.2
    samples = np.full(100, truth)
    s = summarize_sampling_distribution(samples, truth)
    assert s["bias"] == pytest.approx(0.0)
    assert s["rmse"] == pytest.approx(0.0)


@pytest.mark.slow
def test_pipeline_matches_snapshot():
    """Full seeded run must match checked-in snapshot within tolerance."""
    assert SNAPSHOT.exists(), "run with --write-snapshot once to create snapshot"
    metrics = run_pipeline(CONFIG, write_snapshot=False)
    expected = json.loads(SNAPSHOT.read_text())

    assert metrics["seed"] == expected["seed"]
    assert metrics["process"]["true_relative_uplift"] == pytest.approx(
        expected["true_relative_uplift"], abs=1e-9
    )
    assert metrics["monte_carlo"]["mean"] == pytest.approx(
        expected["monte_carlo_mean"], abs=5e-3
    )
    assert metrics["monte_carlo"]["std"] == pytest.approx(
        expected["monte_carlo_std"], abs=5e-3
    )
    assert metrics["bootstrap_ci"]["point"] == pytest.approx(
        expected["bootstrap_point"], abs=5e-3
    )
    assert metrics["bootstrap_ci"]["lower"] == pytest.approx(
        expected["bootstrap_lower"], abs=1e-2
    )
    assert metrics["bootstrap_ci"]["upper"] == pytest.approx(
        expected["bootstrap_upper"], abs=1e-2
    )
    assert metrics["coverage"]["bootstrap_coverage"] == pytest.approx(
        expected["bootstrap_coverage"], abs=0.08
    )
    assert metrics["coverage"]["normal_coverage"] == pytest.approx(
        expected["normal_coverage"], abs=0.08
    )


def test_params_from_config():
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    p = params_from_config(cfg)
    assert p.n_control == 5000
    assert p.p_treatment == 0.095
