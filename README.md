# Computational Peto’s Paradox — Results Summary

This repository summarizes a sequence of analyses comparing hazard models for AI task success, with extensive checks for confounding (task mix), alternative fits (logistic vs Weibull vs exponential), and Bayesian model comparison.

## Data Sources
- Primary dataset: `eval-analysis-public/data/external/all_runs.jsonl`
- Benchmark snapshot: `benchmark_results.yaml`

## Runtime Configuration (Portable Paths)
Scripts now read these environment variables (defaults shown):
- `PETO_DATA_FILE` (default: `eval-analysis-public/data/external/all_runs.jsonl`)
- `PETO_RESULTS_ROOT` (default: `results_mle`)
- `PETO_OUTPUT_DIR` (used by MLE fits, default: `results_mle`)
- `PETO_ADDITIONAL_DIR` (default: `results_mle/additional`)
- `PETO_WEIBULL_CSV` (default: `results_mle/weibull_params_with_bootstrap_ci.csv` if present, else `results/weibull_params_with_bootstrap_ci.csv`)
- `PETO_MIN_N`, `BOOTSTRAP_N`, `BURNIN_MINUTES`, etc. (see script headers)

## Core Models
We compared three model families (all with per‑model parameters in the Bayesian runs):
- **Logistic (METR)**: threshold model in log‑time.
- **Exponential (Ord)**: constant hazard.
- **Weibull (Peto test)**: varying hazard, shape parameter `k`.

## Key Outputs (Files)
### Frequentist / Curve Fit (MLE-based)
- Per‑model fits and BIC (MLE): `results_mle/model_fit_summary.csv`
- Bootstrap CI table (Weibull MLE): `results_mle/weibull_params_with_bootstrap_ci.csv`
- CI plots: `results_mle/weibull_k_by_model_ci.png`, `results_mle/weibull_lambda_by_model_ci.png`

### Bayesian Model Comparison (AI Only)
- WAIC: `results_mle/bayes_compare/bayesian_model_comparison_waic.csv`
- Forest plot (Weibull k): `results_mle/bayes_compare/peto_coefficient_forest.png`

### Bayesian Hierarchical (AI Only)
- Posterior summary: `results_mle/hierarchical/posterior_summary.csv`

### Stratified Analyses
- Task group (SWAA / HCAST / RE‑Bench):
  - Summary: `results_mle/stratified_task_groups/task_group_posterior_summary.csv`
  - Plot: `results_mle/stratified_task_groups/task_group_k_forest.png`
- Task family (repo):
  - Summary: `results_mle/stratified/stratified_posterior_summary.csv`
  - Plot: `results_mle/stratified/stratified_k_forest.png`

### Model Horizons & Comparisons
- p50/p80 logistic vs Weibull: `results_mle/additional/p50_p80_logistic_weibull.csv`
- p99: `results_mle/additional/p99_logistic_weibull.csv`
- p99.9: `results_mle/additional/p999_logistic_weibull.csv`
- p1: `results_mle/additional/p1_logistic_weibull.csv`
- Scatter plots: `results_mle/additional/p50_logistic_vs_weibull.png`, `results_mle/additional/p80_logistic_vs_weibull.png`, `results_mle/additional/p99_logistic_vs_weibull.png`, `results_mle/additional/p999_logistic_vs_weibull.png`, `results_mle/additional/p1_logistic_vs_weibull.png`

### Additional Analyses
- Peto curve (log lambda vs capability proxy):
  - Data: `results_mle/additional/peto_curve_data.csv`
  - Plot: `results_mle/additional/peto_curve_log_lambda_vs_capability.png`
- Burn‑in test (30 min truncation):
  - Data: `results_mle/additional/burnin_weibull_comparison.csv`
  - Plot: `results_mle/additional/burnin_k_shift.png`
- Task‑group lambda by model:
  - Data: `results_mle/additional/task_group_model_weibull.csv`
  - Plot: `results_mle/additional/task_group_lambda_by_model.png`
- Capability covariate (Bayesian):
  - Beta summary: `results_mle/additional/bayes_capability_beta_summary.csv`
  - Posterior plot: `results_mle/additional/bayes_capability_beta_posterior.png`
  - P(beta<0): `results_mle/additional/bayes_capability_beta_pneg.txt`
- Metric correlations:
  - Data: `results_mle/additional/model_level_metrics.csv`
  - Plot: `results_mle/additional/metrics_correlation_heatmap.png`
  - Lambda vs accuracy plot: `results_mle/additional/lambda_vs_accuracy.png`

### k‑Fold CV (Bayesian)
- With ceiling `c`:
  - Summary: `results_mle/bayes_compare_kfold/kfold_elpd_summary.csv`
  - By fold: `results_mle/bayes_compare_kfold/kfold_elpd_by_fold.csv`
  - By alias: `results_mle/bayes_compare_kfold/kfold_elpd_by_alias_summary.csv`
- Without ceiling `c`:
  - Summary: `results_mle/bayes_compare_kfold_noc/kfold_elpd_summary.csv`
  - By alias: `results_mle/bayes_compare_kfold_noc/kfold_elpd_by_alias_summary.csv`

## Key Findings (Short)
- **Logistic** wins by BIC for ~half the models and wins WAIC; Weibull is often close.
- **Weibull k < 1** frequently appears, consistent with “infant mortality” / decreasing hazard, but the effect varies by task group.
- **Stratified task groups:** HCAST shows strong k<1; RE‑Bench and SWAA overlap k≈1.
- **Model comparison via 5‑fold CV:** Logistic > Weibull > Exponential, with ~14 elpd gap between Logistic and Weibull.
- **Per‑alias CV:** Logistic wins broadly; a few models slightly prefer Weibull.
- **Peto curve (log lambda vs capability):** strong negative correlation using capability proxy (mean success rate).

## Notes on Fit & Warnings
- WAIC/LOO produced warnings for Weibull/Exponential in some runs (high variance / Pareto‑k).
- k‑fold CV is more stable and is the primary comparison used.
- Some Bayesian runs still show minor R‑hat warnings; interpretations should focus on robust comparisons (CV, consistency across analyses).

## Scripts Used
- `code/generate_model_fits.py` — per‑model curve fits + plots
- `code/hierarchical_bayes.py` — hierarchical Weibull with ceiling
- `code/bayesian_model_comparison.py` — WAIC/LOO comparison
- `code/stratified_weibull.py` — stratified by task family
- `code/stratified_task_groups.py` — stratified by task source
- `code/bayes_posteriors.py` — posterior plots (AI only)
- `code/additional_analyses.py` — peto curve, burn‑in, task group checks, covariate model
- `code/additional_metrics.py` — lambda vs accuracy + correlations
- `code/bayes_compare_kfold.py` — k‑fold CV with ceiling `c`
- `code/bayes_compare_kfold_noc.py` — k‑fold CV without ceiling `c`
- `code/mle_utils.py` — Bernoulli MLE helpers for logistic/exponential/Weibull
- `code/bootstrap_weibull_params_mle.py` — MLE bootstrap CI regeneration

## Conversation Summary (Why These Analyses)
- Compared logistic vs exponential vs Weibull fits to test constant vs increasing vs decreasing hazard.
- Added hierarchical Bayesian models with ceiling `c` to probe “infant mortality.”
- Stratified by task groups and task families to rule out Simpson’s Paradox.
- Added burn‑in truncation to test early‑failure effects.
- Compared horizons (p50/p80/p99/p99.9/p1) across logistic and Weibull.
- Ran k‑fold CV for robust model comparison and per‑alias attribution.

## High‑Level Interpretation
- The best‑supported global fit is logistic, but Weibull remains competitive and often implies decreasing hazard (k<1).
- The “improving horizon” can be explained by either threshold shifts (logistic) or early‑failure dynamics (Weibull); stratification suggests heterogeneity across task types.

---
If you want a shorter or more narrative README for publication, tell me the target audience and I’ll rewrite it.
