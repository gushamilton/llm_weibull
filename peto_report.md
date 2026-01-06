# Computational Peto's Paradox: Model Fit Summary

## Code
- Script: `/Users/fh6520/R/peto/results/generate_model_fits.py`
- Run command:
```bash
. /Users/fh6520/R/peto/eval-analysis-public/.venv/bin/activate && python /Users/fh6520/R/peto/results/generate_model_fits.py
```
- Outputs:
  - `/Users/fh6520/R/peto/results/model_fit_summary.csv`
  - `/Users/fh6520/R/peto/results/model_fits/*.png`
  - `/Users/fh6520/R/peto/results/weibull_k_by_model_ci.png`
  - `/Users/fh6520/R/peto/results/weibull_lambda_by_model_ci.png`
  - `/Users/fh6520/R/peto/results/weibull_params_with_bootstrap_ci.csv`

## Results
- Best BIC counts (out of 14 models, excluding `gpt-3.5-turbo-instruct`): logistic 7, weibull 5, exponential 2.
- Weibull k summary across models: mean 0.605, median 0.579, max 0.951.
- All fitted k values are < 1, implying decreasing hazard under the Weibull model for every model in this filtered set.
- Per-model summary and parameters: `/Users/fh6520/R/peto/results/model_fit_summary.csv`.

## Overall Impression
- The data favor a threshold-style logistic fit for about half the models, but Weibull competes closely and wins for a sizable minority.
- Exponential (constant hazard) is rarely best by BIC.
- The Weibull shape parameter consistently falls below 1, indicating a decreasing hazard pattern with task duration in this dataset, not an increasing one.

## Bayesian Hierarchical Model
## Code
- Script: `/Users/fh6520/R/peto/results/hierarchical_bayes.py`
- Run command (fast run used here):
```bash
. /Users/fh6520/R/peto/eval-analysis-public/.venv/bin/activate && PM_DRAWS=500 PM_TUNE=500 python /Users/fh6520/R/peto/results/hierarchical_bayes.py
```
- Outputs:
  - `/Users/fh6520/R/peto/results/hierarchical/posterior_summary.csv`
  - `/Users/fh6520/R/peto/results/hierarchical/p_k_gt_1_by_model.csv`
  - `/Users/fh6520/R/peto/results/hierarchical/p_k_gt_1_global.txt`
  - `/Users/fh6520/R/peto/results/hierarchical/hierarchical_k_forest.png`
  - `/Users/fh6520/R/peto/results/hierarchical/hierarchical_c_forest.png`

## Results
- Posterior P(k > 1) is 0.0 for every model in the hierarchy; global P(k > 1) is 0.0.
- k posteriors are all below 1.0 at 95% HDI, consistent with decreasing hazard even after adding a ceiling parameter.

## Sampling Notes
- The fast run produced divergences and some R-hat > 1.01; re-run with more draws and higher target_accept for final results.
