# monte-carlo-inference

Monte Carlo sampling distributions and bootstrap confidence intervals for **A/B relative conversion uplift** under unequal traffic.

Intended public repo name: `monte-carlo-inference-lab` · author **Lrk777**.

## Problem

In an A/B test with unequal allocation, how uncertain is the **relative** conversion uplift \((p_t - p_c)/p_c\)? Absolute lift is easier; stakeholders usually ask for percent lift, which is nonlinear and more skewed — especially when the control rate is small.

## Demo

```text
$ python -m src.estimate --config configs/default.yaml

monte-carlo-inference-lab
  truth relative uplift: 0.1875 (0.080 → 0.095)
  MC mean±std (n=5000): 0.1854 ± 0.1014  bias=-0.0021
  observed relative uplift: 0.1573
  bootstrap 95% CI [-0.0261, 0.3649]  point=0.1573
  normal    95% CI [-0.0376, 0.3522]  point=0.1573
  coverage @ 95%: bootstrap=0.945  normal=0.948  (n_trials=400)
```

Figures (checked in under `reports/figures/`):

| Figure | What it shows |
|--------|----------------|
| `sampling_distribution.png` | Oracle MC distribution of \(\hat r\) vs truth |
| `bootstrap_ci.png` | Percentile bootstrap CI on one observed experiment |
| `ci_convergence.png` | CI half-width vs total \(n\) (log–log) |

Longer write-up: [`methods.md`](./methods.md).

## Method

1. **Generative model** — Independent Bernoulli conversions: control \(p_c=0.08\) (\(n_c=5000\)), treatment \(p_t=0.095\) (\(n_t=2000\)). True relative uplift \(r = 0.1875\).
2. **Monte Carlo** — Simulate 5000 synthetic experiments; summarize bias / variance / RMSE of \(\hat r\).
3. **One observed experiment** — Draw a single seeded A/B realization; estimate \(\hat r\).
4. **Percentile bootstrap** — Resample within arms (4000 resamples) → 95% CI.
5. **Normal / delta-method CI** — Analytic SE for relative uplift (comparison baseline).
6. **Coverage study** — 400 repeated experiments; fraction of CIs that contain true \(r\).
7. **Convergence** — Scale sample sizes by \(\{0.25,0.5,1,2,4\}\) and plot CI half-width vs \(n\).

All knobs live in `configs/default.yaml` (seed `42`).

## Results

Checked-in run (seed **42**):

| Quantity | Value |
|----------|-------|
| True relative uplift | **0.1875** |
| MC mean ± std (5000 expts) | 0.1854 ± 0.1014 |
| MC bias | −0.0021 |
| Observed \(\hat r\) | 0.1573 |
| Bootstrap 95% CI | **[−0.026, 0.365]** |
| Normal 95% CI | [−0.038, 0.352] |
| Bootstrap coverage (400 trials) | **0.945** |
| Normal coverage (400 trials) | 0.948 |

**Takeaways**

- The relative-uplift estimator is nearly unbiased here (bias ≈ −0.002) but **noisy** (std ≈ 0.10) at these sample sizes — a single experiment’s point estimate can sit well below truth.
- On the observed draw, the 95% bootstrap interval **includes 0**, so you would not call a significant relative lift despite a true +18.75% effect — power/uncertainty matter more than the point estimate.
- Empirical coverage for both CI methods is near nominal (~95%). CI half-width shrinks roughly like \(1/\sqrt{n}\) as total traffic grows (see convergence figure).

## Stack

- Python 3.11+
- NumPy, matplotlib, PyYAML, pytest

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m src.estimate --config configs/default.yaml
pytest -q
```

Optional overrides:

```bash
python -m src.estimate --seed 42 --n 10000
python -m src.estimate --write-snapshot   # refresh reports/snapshots/expected_metrics.json
```

## Limitations

- Lab data only — rates and allocation are **assumed**, not estimated from a product.
- Independent Bernoulli users; no session dependence, bots, or network effects.
- Percentile bootstrap is not an exact finite-sample guarantee; coverage is empirical.
- Relative uplift blows up when \(\hat p_c \approx 0\); unsuitable for very rare events without a different estimand.
- No sequential testing, peeking correction, CUPED, or multi-metric FDR.
- What would invalidate claims: non-stationarity mid-test, optional stopping, or treating users as independent when they are not. See [`methods.md`](./methods.md).

## License

MIT — see [`LICENSE`](./LICENSE).
