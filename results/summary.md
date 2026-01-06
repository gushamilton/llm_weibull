# Peto Analysis Summary (All Results)

## Data & Scope
- Dataset: `eval-analysis-public/data/external/all_runs.jsonl`
- Outcome: `score_binarized` (success=1)
- Time: `human_minutes` (converted to hours for Bayesian models)

## Frequentist Fits (per-model, BIC/AIC)
- Summary table: `/Users/fh6520/R/peto/results/model_fit_summary.csv`
- Result: Logistic wins BIC for 7/14 models, Weibull for 5/14, Exponential for 2/14.
- Interpretation: Logistic (threshold) generally fits best by BIC, but Weibull is competitive for a sizeable minority.

## Weibull Parameters (Bootstrap CIs)
- Table: `/Users/fh6520/R/peto/results/weibull_params_with_bootstrap_ci.csv`
- Plot (k with 95% CI): `/Users/fh6520/R/peto/results/weibull_k_by_model_ci.png`
- Plot (lambda with 95% CI, log scale): `/Users/fh6520/R/peto/results/weibull_lambda_by_model_ci.png`
- Note: `gpt-3.5-turbo-instruct` was excluded from these plots/tables.

## Bayesian Hierarchical (Model-Level, AI Only)
- Script: `/Users/fh6520/R/peto/results/hierarchical_bayes.py`
- Posterior summary: `/Users/fh6520/R/peto/results/hierarchical/posterior_summary.csv`
- Key finding: Posterior k values are all < 1 in this run; P(k>1) is 0 across models.
- Caveat: Some sampling diagnostics flagged in earlier shorter runs; this was a fast run with 2 chains.

## Bayesian Model Comparison (AI Only)
- WAIC comparison: `/Users/fh6520/R/peto/results/bayes_compare/bayesian_model_comparison_waic.csv`
- Result (WAIC): Logistic ranked best (weight ~0.80), Weibull second (weight ~0.20), Exponential last.
- Plot: `/Users/fh6520/R/peto/results/bayes_compare/peto_coefficient_forest.png`
- Caveat: WAIC emitted warnings for Weibull/Exponential; Logistic was clean.

## Stratified by Task Group (SWAA, HCAST, RE-Bench)
- Summary: `/Users/fh6520/R/peto/results/stratified_task_groups/task_group_posterior_summary.csv`
- Plot: `/Users/fh6520/R/peto/results/stratified_task_groups/task_group_k_forest.png`
- k (mean, 95% HDI):
  - HCAST: 0.583 (0.544–0.624)
  - RE-Bench: 0.864 (0.270–1.757)
  - SWAA: 1.002 (0.568–1.560)
- Interpretation: HCAST shows strong k<1; RE-Bench and SWAA overlap 1.0.

## Overall Impression
- Logistic is the strongest overall baseline by BIC and WAIC.
- Weibull often fits well but does not dominate; hazard increase (k>1) is not robust in these runs.
- Stratifying by task groups shows heterogeneity: one group (HCAST) supports k<1, while others are ambiguous.
- Next step for a definitive call: increase chains/draws and consider k-fold CV for model comparison.
