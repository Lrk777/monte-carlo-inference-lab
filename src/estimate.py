"""Estimation: bootstrap CIs, coverage, and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.plot import (
    plot_bootstrap_histogram,
    plot_ci_convergence,
    plot_sampling_distribution,
)
from src.simulate import (
    ExperimentParams,
    ExperimentResult,
    scale_params,
    simulate_experiment,
    simulate_many,
)


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    level: float
    method: str
    n_resamples: int
    n_valid: int


def relative_uplift_from_arrays(control: np.ndarray, treatment: np.ndarray) -> float:
    rc = float(np.mean(control))
    rt = float(np.mean(treatment))
    if rc <= 0:
        return float("nan")
    return (rt - rc) / rc


def bootstrap_relative_uplift(
    experiment: ExperimentResult,
    n_resamples: int,
    rng: np.random.Generator,
    ci_level: float = 0.95,
) -> tuple[BootstrapCI, np.ndarray]:
    """Percentile bootstrap CI for relative uplift (resampling within arms)."""
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    if n_resamples < 10:
        raise ValueError("n_resamples too small")

    n_c = len(experiment.control)
    n_t = len(experiment.treatment)
    point = experiment.relative_uplift

    idx_c = rng.integers(0, n_c, size=(n_resamples, n_c))
    idx_t = rng.integers(0, n_t, size=(n_resamples, n_t))
    rate_c = experiment.control[idx_c].mean(axis=1)
    rate_t = experiment.treatment[idx_t].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = (rate_t - rate_c) / rate_c

    valid = samples[np.isfinite(samples)]
    alpha = 1.0 - ci_level
    if valid.size < max(20, n_resamples // 50):
        raise RuntimeError("too few finite bootstrap samples for a stable CI")

    lower = float(np.quantile(valid, alpha / 2))
    upper = float(np.quantile(valid, 1.0 - alpha / 2))
    ci = BootstrapCI(
        point=float(point) if math.isfinite(point) else float("nan"),
        lower=lower,
        upper=upper,
        level=ci_level,
        method="percentile",
        n_resamples=n_resamples,
        n_valid=int(valid.size),
    )
    return ci, samples


def analytic_relative_uplift_se(rate_c: float, rate_t: float, n_c: int, n_t: int) -> float:
    """
    Delta-method SE for relative uplift r = (p_t - p_c) / p_c.

    Var(r) ≈ (1/p_c^2) Var(p_t) + (p_t^2 / p_c^4) Var(p_c)
    with Var(p) = p(1-p)/n.
    """
    if rate_c <= 0:
        return float("nan")
    var_c = rate_c * (1.0 - rate_c) / n_c
    var_t = rate_t * (1.0 - rate_t) / n_t
    var_r = (1.0 / rate_c**2) * var_t + (rate_t**2 / rate_c**4) * var_c
    return float(math.sqrt(max(var_r, 0.0)))


def normal_approx_ci(
    experiment: ExperimentResult,
    ci_level: float = 0.95,
) -> BootstrapCI:
    """Wald/normal CI using the delta-method SE (often optimistic under skew)."""
    z = float(_normal_ppf(0.5 + ci_level / 2.0))
    point = experiment.relative_uplift
    se = analytic_relative_uplift_se(
        experiment.rate_control,
        experiment.rate_treatment,
        len(experiment.control),
        len(experiment.treatment),
    )
    return BootstrapCI(
        point=float(point),
        lower=float(point - z * se),
        upper=float(point + z * se),
        level=ci_level,
        method="normal_delta",
        n_resamples=0,
        n_valid=0,
    )


def _normal_ppf(p: float) -> float:
    """Approximate inverse CDF of N(0,1) (Acklam / Beasley-Springer style)."""
    # Rational approximation good enough for CI z-scores
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614436e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879557428e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def summarize_sampling_distribution(samples: np.ndarray, truth: float) -> dict[str, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        raise RuntimeError("no finite Monte Carlo samples")
    mean = float(np.mean(finite))
    return {
        "n_experiments": int(samples.size),
        "n_finite": int(finite.size),
        "mean": mean,
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
        "bias": mean - truth,
        "rmse": float(math.sqrt(np.mean((finite - truth) ** 2))),
        "truth": truth,
    }


def estimate_coverage(
    params: ExperimentParams,
    n_trials: int,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Empirical coverage of percentile bootstrap vs normal-approx CIs."""
    truth = params.relative_uplift
    boot_hits = 0
    normal_hits = 0
    boot_widths: list[float] = []
    normal_widths: list[float] = []

    for _ in range(n_trials):
        exp = simulate_experiment(params, rng)
        boot_ci, _ = bootstrap_relative_uplift(exp, n_bootstrap, rng, ci_level)
        norm_ci = normal_approx_ci(exp, ci_level)
        if boot_ci.lower <= truth <= boot_ci.upper:
            boot_hits += 1
        if norm_ci.lower <= truth <= norm_ci.upper:
            normal_hits += 1
        boot_widths.append(boot_ci.upper - boot_ci.lower)
        normal_widths.append(norm_ci.upper - norm_ci.lower)

    return {
        "n_trials": float(n_trials),
        "ci_level": ci_level,
        "bootstrap_coverage": boot_hits / n_trials,
        "normal_coverage": normal_hits / n_trials,
        "bootstrap_mean_width": float(np.mean(boot_widths)),
        "normal_mean_width": float(np.mean(normal_widths)),
        "truth": truth,
    }


def ci_convergence(
    params: ExperimentParams,
    scale_factors: list[float],
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """CI half-width vs scaled sample size for one seeded experiment per scale."""
    rows: list[dict[str, float]] = []
    for factor in scale_factors:
        scaled = scale_params(params, factor)
        exp = simulate_experiment(scaled, rng)
        ci, _ = bootstrap_relative_uplift(exp, n_bootstrap, rng, ci_level)
        half = 0.5 * (ci.upper - ci.lower)
        rows.append(
            {
                "scale": float(factor),
                "n_control": float(scaled.n_control),
                "n_treatment": float(scaled.n_treatment),
                "n_total": float(scaled.n_total),
                "point": ci.point,
                "half_width": half,
                "lower": ci.lower,
                "upper": ci.upper,
            }
        )
    return rows


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def params_from_config(cfg: dict[str, Any]) -> ExperimentParams:
    p = cfg["process"]
    return ExperimentParams(
        p_control=float(p["p_control"]),
        p_treatment=float(p["p_treatment"]),
        n_control=int(p["n_control"]),
        n_treatment=int(p["n_treatment"]),
    )


def _ci_to_dict(ci: BootstrapCI) -> dict[str, Any]:
    return asdict(ci)


def run_pipeline(config_path: Path, write_snapshot: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    params = params_from_config(cfg)
    truth = params.relative_uplift

    # 1) Monte Carlo sampling distribution
    mc_samples = simulate_many(params, int(cfg["monte_carlo"]["n_experiments"]), rng)
    mc_summary = summarize_sampling_distribution(mc_samples, truth)

    # 2) One observed experiment + bootstrap + normal CI
    observed = simulate_experiment(params, rng)
    boot_ci, boot_samples = bootstrap_relative_uplift(
        observed,
        int(cfg["bootstrap"]["n_resamples"]),
        rng,
        float(cfg["bootstrap"]["ci_level"]),
    )
    norm_ci = normal_approx_ci(observed, float(cfg["bootstrap"]["ci_level"]))

    # 3) Coverage study
    coverage = estimate_coverage(
        params,
        int(cfg["coverage"]["n_trials"]),
        int(cfg["coverage"]["n_bootstrap"]),
        float(cfg["bootstrap"]["ci_level"]),
        rng,
    )

    # 4) Convergence grid
    conv = ci_convergence(
        params,
        [float(x) for x in cfg["convergence"]["scale_factors"]],
        int(cfg["convergence"]["n_bootstrap"]),
        float(cfg["bootstrap"]["ci_level"]),
        rng,
    )

    metrics: dict[str, Any] = {
        "seed": seed,
        "process": {
            "p_control": params.p_control,
            "p_treatment": params.p_treatment,
            "n_control": params.n_control,
            "n_treatment": params.n_treatment,
            "true_absolute_uplift": params.absolute_uplift,
            "true_relative_uplift": truth,
        },
        "observed_experiment": {
            "rate_control": observed.rate_control,
            "rate_treatment": observed.rate_treatment,
            "absolute_uplift": observed.absolute_uplift,
            "relative_uplift": observed.relative_uplift,
        },
        "monte_carlo": mc_summary,
        "bootstrap_ci": _ci_to_dict(boot_ci),
        "normal_ci": _ci_to_dict(norm_ci),
        "coverage": coverage,
        "convergence": conv,
    }

    root = config_path.resolve().parents[1] if config_path.name.endswith(".yaml") else Path.cwd()
    # Prefer project root = parent of configs/
    if config_path.parent.name == "configs":
        root = config_path.parent.parent
    else:
        root = Path.cwd()

    metrics_path = root / cfg["paths"]["metrics"]
    figures_dir = root / cfg["paths"]["figures"]
    snapshot_path = root / cfg["paths"]["snapshot"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    plot_sampling_distribution(
        mc_samples,
        truth,
        figures_dir / "sampling_distribution.png",
    )
    plot_bootstrap_histogram(
        boot_samples,
        boot_ci,
        truth,
        figures_dir / "bootstrap_ci.png",
    )
    plot_ci_convergence(conv, figures_dir / "ci_convergence.png")

    if write_snapshot:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        # Compact snapshot for regression tests (stable keys only)
        snapshot = {
            "seed": seed,
            "true_relative_uplift": round(truth, 10),
            "monte_carlo_mean": round(mc_summary["mean"], 6),
            "monte_carlo_std": round(mc_summary["std"], 6),
            "bootstrap_point": round(boot_ci.point, 6),
            "bootstrap_lower": round(boot_ci.lower, 6),
            "bootstrap_upper": round(boot_ci.upper, 6),
            "bootstrap_coverage": round(coverage["bootstrap_coverage"], 4),
            "normal_coverage": round(coverage["normal_coverage"], 4),
        }
        with snapshot_path.open("w") as f:
            json.dump(snapshot, f, indent=2)
            f.write("\n")

    return metrics


def _print_summary(metrics: dict[str, Any]) -> None:
    p = metrics["process"]
    mc = metrics["monte_carlo"]
    b = metrics["bootstrap_ci"]
    n = metrics["normal_ci"]
    c = metrics["coverage"]
    print("monte-carlo-inference-lab")
    print(
        f"  truth relative uplift: {p['true_relative_uplift']:.4f} "
        f"({p['p_control']:.3f} → {p['p_treatment']:.3f})"
    )
    print(
        f"  MC mean±std (n={int(mc['n_finite'])}): "
        f"{mc['mean']:.4f} ± {mc['std']:.4f}  bias={mc['bias']:+.4f}"
    )
    print(
        f"  observed relative uplift: {metrics['observed_experiment']['relative_uplift']:.4f}"
    )
    print(
        f"  bootstrap {b['level']:.0%} CI [{b['lower']:.4f}, {b['upper']:.4f}]  "
        f"point={b['point']:.4f}"
    )
    print(
        f"  normal    {n['level']:.0%} CI [{n['lower']:.4f}, {n['upper']:.4f}]  "
        f"point={n['point']:.4f}"
    )
    print(
        f"  coverage @ {c['ci_level']:.0%}: bootstrap={c['bootstrap_coverage']:.3f}  "
        f"normal={c['normal_coverage']:.3f}  (n_trials={int(c['n_trials'])})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monte Carlo + bootstrap inference for A/B relative uplift"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML config path",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override Monte Carlo n_experiments",
    )
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help="Write reports/snapshots/expected_metrics.json for tests",
    )
    args = parser.parse_args(argv)

    cfg_path = args.config
    if args.seed is not None or args.n is not None:
        cfg = load_config(cfg_path)
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.n is not None:
            cfg["monte_carlo"]["n_experiments"] = args.n
        tmp = Path("/tmp/monte-carlo-inference-run.yaml")
        with tmp.open("w") as f:
            yaml.safe_dump(cfg, f)
        cfg_path = tmp

    metrics = run_pipeline(cfg_path, write_snapshot=args.write_snapshot)
    _print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
