# Computational Peto's Paradox (METR Agentic Tasks)

This repo analyzes METR's public task-run data to compare hazard models for AI task
success (logistic vs Weibull vs exponential) and test for decreasing hazard
(Peto-style "infant mortality"). It includes frequentist curve fits, Bayesian
model comparison, stratified analyses, and bootstrap checks.

## Repository Layout
- `code/` — analysis scripts (MLE fits, Bayesian models, bootstraps, stratified runs).
- `results_mle/` — primary outputs for the current analysis pass.
- `docs/` — written summaries and reports.
- `eval-analysis-public/` — upstream METR repo and data (vendored copy).
- `benchmark_results.yaml` — benchmark snapshot used in some summaries.

## Data Inputs
- Primary dataset: `eval-analysis-public/data/external/all_runs.jsonl`.
- If you store the dataset elsewhere, set `PETO_DATA_FILE` to its path.

## Quickstart
All scripts default to reading from `eval-analysis-public` and writing to
`results_mle/`.

```bash
python code/generate_model_fits.py
```

## Environment
- For dependencies, see `eval-analysis-public/README.md` (Poetry + DVC setup).
- Any Python environment with the required packages is fine.

## Configuration
You can override paths and thresholds via environment variables:
- `PETO_DATA_FILE` (default: `eval-analysis-public/data/external/all_runs.jsonl`)
- `PETO_RESULTS_ROOT` (default: `results_mle`)
- `PETO_OUTPUT_DIR` (default: `results_mle`)
- `PETO_ADDITIONAL_DIR` (default: `results_mle/additional`)
- `PETO_WEIBULL_CSV` (default: `results_mle/weibull_params_with_bootstrap_ci.csv`)
- `PETO_MIN_N`, `BOOTSTRAP_N`, `BURNIN_MINUTES`, etc. (see script headers)

## Key Outputs
- MLE fit summary: `results_mle/model_fit_summary.csv`
- Weibull parameter CIs: `results_mle/weibull_params_with_bootstrap_ci.csv`
- Bayesian comparison: `results_mle/bayes_compare/bayesian_model_comparison_waic.csv`
- Stratified summaries: `results_mle/stratified/` and `results_mle/stratified_task_groups/`

## Key Results (In Order)
1) Frequentist fits: logistic wins BIC for about half the models, Weibull is close behind, exponential rarely wins.
2) Weibull k estimates: k < 1 is common, consistent with decreasing hazard / "infant mortality".
3) Bayesian model comparison (AI-only): WAIC prefers logistic overall, Weibull second.
4) Hierarchical Bayes: posterior mass for k > 1 is near zero in these runs.
5) Stratified analyses: HCAST shows strong k < 1; RE-Bench and SWAA overlap k ≈ 1.
6) Horizon comparisons: logistic vs Weibull horizons are similar but diverge at extremes for some models.
7) k-fold CV: logistic > Weibull > exponential; this is the most robust comparison.

## Reports
- `docs/peto_report.md` — short report on the initial fit and hierarchical model.
- `docs/key_results.md` — ordered, concise walkthrough of the findings.

## Notes
- `results_mle/` is the authoritative output set for current runs.
- Some Bayesian runs emit diagnostics warnings; the k-fold CV results are the most stable.
