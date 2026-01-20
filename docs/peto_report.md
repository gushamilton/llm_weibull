# Computational Peto's Paradox: Model Fit Summary

## Code
- Script: `code/generate_model_fits.py`
- Run command:
```bash
source eval-analysis-public/.venv/bin/activate
python code/generate_model_fits.py
```
- Outputs:
  - `results/model_fit_summary.csv`
  - `results/model_fits/*.png`
  - `results/weibull_k_by_model_ci.png`
  - `results/weibull_lambda_by_model_ci.png`
  - `results/weibull_params_with_bootstrap_ci.csv`

## Results
- Best BIC counts (out of 14 models, excluding `gpt-3.5-turbo-instruct`): logistic 7, weibull 5, exponential 2.
- Weibull k summary across models: mean 0.605, median 0.579, max 0.951.
- All fitted k values are < 1, implying decreasing hazard under the Weibull model for every model in this filtered set.
- Per-model summary and parameters: `results/model_fit_summary.csv`.

## Overall Impression
- The data favor a threshold-style logistic fit for about half the models, but Weibull competes closely and wins for a sizable minority.
- Exponential (constant hazard) is rarely best by BIC.
- The Weibull shape parameter consistently falls below 1, indicating a decreasing hazard pattern with task duration in this dataset, not an increasing one.

## Bayesian Hierarchical Model
## Code
- Script: `code/hierarchical_bayes.py`
- Run command (fast run used here):
```bash
source eval-analysis-public/.venv/bin/activate
PM_DRAWS=500 PM_TUNE=500 python code/hierarchical_bayes.py
```
- Outputs:
  - `results/hierarchical/posterior_summary.csv`
  - `results/hierarchical/p_k_gt_1_by_model.csv`
  - `results/hierarchical/p_k_gt_1_global.txt`
  - `results/hierarchical/hierarchical_k_forest.png`
  - `results/hierarchical/hierarchical_c_forest.png`

## Results
- Posterior P(k > 1) is 0.0 for every model in the hierarchy; global P(k > 1) is 0.0.
- k posteriors are all below 1.0 at 95% HDI, consistent with decreasing hazard even after adding a ceiling parameter.

## Sampling Notes
- The fast run produced divergences and some R-hat > 1.01; re-run with more draws and higher target_accept for final results.
