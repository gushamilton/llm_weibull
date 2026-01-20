# Key Results (Ordered)

1) Frequentist model fits (MLE)
- Logistic wins BIC for about half the models; Weibull is close behind; exponential rarely wins.
- Output: `results_mle/model_fit_summary.csv`

2) Weibull parameter estimates
- Weibull shape parameter k is often < 1, consistent with decreasing hazard ("infant mortality").
- Outputs: `results_mle/weibull_params_with_bootstrap_ci.csv`, `results_mle/weibull_k_by_model_ci.png`

3) Bayesian model comparison (AI-only)
- WAIC ranks logistic best overall, Weibull second, exponential last (with some diagnostics warnings).
- Output: `results_mle/bayes_compare/bayesian_model_comparison_waic.csv`

4) Bayesian hierarchical model (AI-only)
- Posterior mass for k > 1 is near zero in these runs.
- Output: `results_mle/hierarchical/posterior_summary.csv`

5) Stratified analyses
- Task group stratification: HCAST supports k < 1; RE-Bench and SWAA overlap k ≈ 1.
- Outputs: `results_mle/stratified_task_groups/`, `results_mle/stratified/`

6) Horizon comparisons and additional checks
- Logistic vs Weibull horizons (p50/p80/p99/p99.9/p1) are similar but diverge at extremes for some models.
- Outputs: `results_mle/additional/`

7) k-fold cross-validation (most robust)
- CV ranks logistic > Weibull > exponential; this is the most stable comparison.
- Outputs: `results_mle/bayes_compare_kfold/`, `results_mle/bayes_compare_kfold_noc/`
