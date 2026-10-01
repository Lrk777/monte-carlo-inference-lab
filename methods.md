# Methods — Monte Carlo inference lab

## Quantity of interest

Relative conversion uplift

\[
r = \frac{p_t - p_c}{p_c}
\]

under unequal traffic (\(n_c \neq n_t\)). Absolute uplift \(p_t - p_c\) is easier statistically; product and analytics stakeholders usually ask for **relative** lift, which is a nonlinear transform and more skewed.

Default generative truth: \(p_c = 0.08\), \(p_t = 0.095\) → \(r = 0.1875\), with \(n_c = 5000\), \(n_t = 2000\).

## Generative model

Independent Bernoulli outcomes:

- Control: \(X_i \sim \mathrm{Bernoulli}(p_c)\), \(i = 1,\ldots,n_c\)
- Treatment: \(Y_j \sim \mathrm{Bernoulli}(p_t)\), \(j = 1,\ldots,n_t\)

Estimator on one experiment:

\[
\hat r = \frac{\hat p_t - \hat p_c}{\hat p_c}, \quad \hat p_c = \bar X,\ \hat p_t = \bar Y
\]

Undefined (or unstable) when \(\hat p_c = 0\); at these rates that is rare but handled as non-finite in summaries.

## Monte Carlo sampling distribution

Draw \(M\) independent experiments from the known process and collect \(\{\hat r_m\}\). Report mean, std, 5th/95th percentiles, bias, and RMSE vs true \(r\). This is the **oracle** sampling distribution — useful for teaching bias/variance, not available in a live A/B test.

## Bootstrap CI (percentile)

On one observed experiment, resample **within arms** with replacement, recompute \(\hat r^*\), and take the \(\alpha/2\) and \(1-\alpha/2\) quantiles. Default: 95% CI, 4000 resamples.

## Normal / delta-method CI

For comparison, a Wald interval uses the delta-method SE of \(r = (p_t - p_c)/p_c\):

\[
\widehat{\mathrm{Var}}(\hat r)
\approx \frac{1}{\hat p_c^2}\widehat{\mathrm{Var}}(\hat p_t)
+ \frac{\hat p_t^2}{\hat p_c^4}\widehat{\mathrm{Var}}(\hat p_c)
\]

with \(\widehat{\mathrm{Var}}(\hat p) = \hat p(1-\hat p)/n\). This can undercover when \(\hat r\) is skewed (small \(p_c\), unequal \(n\)).

## Coverage study

Repeat: simulate experiment → build both CIs → check whether true \(r\) falls inside. Empirical coverage near the nominal level (e.g. 0.95) is the honesty check.

## Convergence

Scale \((n_c, n_t)\) by factors \(\{0.25, 0.5, 1, 2, 4\}\) and plot bootstrap CI half-width vs total \(n\). Expect roughly \(1/\sqrt{n}\) decay on a log-log plot.

## What would invalidate the claims

- Outcomes not independent (network effects, repeated users counted as unique)
- Conversion probability changing mid-experiment (non-stationarity)
- Peeking / optional stopping without adjustment
- Using relative uplift when control rate is near zero
- Treating bootstrap percentile intervals as exact finite-sample guarantees

## Out of scope

Full Bayesian hierarchical models, sequential testing boundaries, CUPED / covariate adjustment, and multi-metric false-discovery control.
