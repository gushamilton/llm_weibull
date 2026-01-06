# Computational Peto’s Paradox — Results Summary

This repository summarizes a sequence of analyses comparing hazard models for AI task success, with extensive checks for confounding (task mix), alternative fits (logistic vs Weibull vs exponential), and Bayesian model comparison.

## Data Sources
- Primary dataset: `eval-analysis-public/data/external/all_runs.jsonl`
- Benchmark snapshot: `benchmark_results.yaml`

## Core Models
We compared three model families (all with per‑model parameters in the Bayesian runs):
- **Logistic (METR)**: threshold model in log‑time.
- **Exponential (Ord)**: constant hazard.
- **Weibull (Peto test)**: varying hazard, shape parameter `k`.

## Key Outputs (Files)
### Frequentist / Curve Fit
- Per‑model fits and BIC: `results/model_fit_summary.csv`
- Bootstrap CI table (Weibull): `results/weibull_params_with_bootstrap_ci.csv`
- CI plots: `results/weibull_k_by_model_ci.png`, `results/weibull_lambda_by_model_ci.png`

### Bayesian Model Comparison (AI Only)
- WAIC: `results/bayes_compare/bayesian_model_comparison_waic.csv`
- Forest plot (Weibull k): `results/bayes_compare/peto_coefficient_forest.png`

### Bayesian Hierarchical (AI Only)
- Posterior summary: `results/hierarchical/posterior_summary.csv`

### Stratified Analyses
- Task group (SWAA / HCAST / RE‑Bench):
  - Summary: `results/stratified_task_groups/task_group_posterior_summary.csv`
  - Plot: `results/stratified_task_groups/task_group_k_forest.png`
- Task family (repo):
  - Summary: `results/stratified/stratified_posterior_summary.csv`
  - Plot: `results/stratified/stratified_k_forest.png`

### Model Horizons & Comparisons
- p50/p80 logistic vs Weibull: `results/additional/p50_p80_logistic_weibull.csv`
- p99: `results/additional/p99_logistic_weibull.csv`
- p99.9: `results/additional/p999_logistic_weibull.csv`
- p1: `results/additional/p1_logistic_weibull.csv`
- Scatter plots: `results/additional/p50_logistic_vs_weibull.png`, `results/additional/p80_logistic_vs_weibull.png`, `results/additional/p99_logistic_vs_weibull.png`, `results/additional/p999_logistic_vs_weibull.png`, `results/additional/p1_logistic_vs_weibull.png`

### Additional Analyses
- Peto curve (log lambda vs capability proxy):
  - Data: `results/additional/peto_curve_data.csv`
  - Plot: `results/additional/peto_curve_log_lambda_vs_capability.png`
- Burn‑in test (30 min truncation):
  - Data: `results/additional/burnin_weibull_comparison.csv`
  - Plot: `results/additional/burnin_k_shift.png`
- Task‑group lambda by model:
  - Data: `results/additional/task_group_model_weibull.csv`
  - Plot: `results/additional/task_group_lambda_by_model.png`
- Capability covariate (Bayesian):
  - Beta summary: `results/additional/bayes_capability_beta_summary.csv`
  - Posterior plot: `results/additional/bayes_capability_beta_posterior.png`
  - P(beta<0): `results/additional/bayes_capability_beta_pneg.txt`
- Metric correlations:
  - Data: `results/additional/model_level_metrics.csv`
  - Plot: `results/additional/metrics_correlation_heatmap.png`
  - Lambda vs accuracy plot: `results/additional/lambda_vs_accuracy.png`

### k‑Fold CV (Bayesian)
- With ceiling `c`:
  - Summary: `results/bayes_compare_kfold/kfold_elpd_summary.csv`
  - By fold: `results/bayes_compare_kfold/kfold_elpd_by_fold.csv`
  - By alias: `results/bayes_compare_kfold/kfold_elpd_by_alias_summary.csv`
- Without ceiling `c`:
  - Summary: `results/bayes_compare_kfold_noc/kfold_elpd_summary.csv`
  - By alias: `results/bayes_compare_kfold_noc/kfold_elpd_by_alias_summary.csv`

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
- `results/generate_model_fits.py` — per‑model curve fits + plots
- `results/hierarchical_bayes.py` — hierarchical Weibull with ceiling
- `results/bayesian_model_comparison.py` — WAIC/LOO comparison
- `results/stratified_weibull.py` — stratified by task family
- `results/stratified_task_groups.py` — stratified by task source
- `results/bayes_posteriors.py` — posterior plots (AI only)
- `results/additional_analyses.py` — peto curve, burn‑in, task group checks, covariate model
- `results/additional_metrics.py` — lambda vs accuracy + correlations
- `results/bayes_compare_kfold.py` — k‑fold CV with ceiling `c`
- `results/bayes_compare_kfold_noc.py` — k‑fold CV without ceiling `c`

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
