"""Figures for sampling distribution, bootstrap CI, and CI convergence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.estimate import BootstrapCI


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
        }
    )


def plot_sampling_distribution(
    samples: np.ndarray,
    truth: float,
    out_path: Path,
) -> None:
    _style()
    finite = samples[np.isfinite(samples)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(finite, bins=40, color="#2c5f6e", alpha=0.85, edgecolor="white", density=True)
    ax.axvline(truth, color="#c45c26", linewidth=2, label=f"truth = {truth:.3f}")
    ax.axvline(float(np.mean(finite)), color="#1a1a1a", linestyle="--", linewidth=1.5, label="MC mean")
    ax.set_xlabel("Relative uplift estimator")
    ax.set_ylabel("Density")
    ax.set_title("Monte Carlo sampling distribution of relative uplift")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_bootstrap_histogram(
    samples: np.ndarray,
    ci: BootstrapCI,
    truth: float,
    out_path: Path,
) -> None:
    _style()
    finite = samples[np.isfinite(samples)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(finite, bins=40, color="#3d6b4f", alpha=0.85, edgecolor="white", density=True)
    ax.axvline(ci.point, color="#1a1a1a", linestyle="--", linewidth=1.5, label=f"point = {ci.point:.3f}")
    ax.axvline(ci.lower, color="#2c5f6e", linewidth=2, label=f"{ci.level:.0%} CI")
    ax.axvline(ci.upper, color="#2c5f6e", linewidth=2)
    ax.axvline(truth, color="#c45c26", linewidth=2, label=f"truth = {truth:.3f}")
    ax.set_xlabel("Bootstrap relative uplift")
    ax.set_ylabel("Density")
    ax.set_title("Percentile bootstrap CI (one observed experiment)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_ci_convergence(rows: list[dict[str, Any]], out_path: Path) -> None:
    _style()
    n_total = [r["n_total"] for r in rows]
    half = [r["half_width"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(n_total, half, marker="o", color="#2c5f6e", linewidth=2, markersize=7)
    ax.set_xlabel("Total sample size (control + treatment)")
    ax.set_ylabel("Bootstrap CI half-width")
    ax.set_title("CI precision vs sample size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
