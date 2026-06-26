# Regression Region Validation

This report validates `InsideForestRegressor` as an interpretable region extractor for continuous targets.
`predict(X)` is evaluated as region labels; the internal random forest is evaluated separately with R2/RMSE.

## Reproduction

```bash
python experiments/validate_regression_regions.py --profile quick
```

## Decision Checks

- Median RF test R2 across all runs: `0.6277`.
- Median test rule coverage: `0.9008`.
- Median test known-region coverage: `0.7460`.
- Median test target spread reduction inside regions: `0.5450`.
- Median region-mean RMSE lift vs train-mean baseline: `0.1537`.
- Classification-style target warnings observed: `0`.

## Dataset Summary

| Dataset | Runs | RF R2 | RF RMSE | Mean RMSE | Region RMSE | Region Lift | Rule Coverage | Known Coverage | Std Reduction | Regions | Clusters | Fit s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| diabetes | 2 | 0.4173 | 56.1308 | 75.0765 | 64.9787 | 0.1342 | 0.9802 | 0.8452 | 0.5283 | 154.5000 | 61.5000 | 0.9619 |
| friedman1 | 2 | 0.7170 | 2.5284 | 4.7960 | 3.5953 | 0.2501 | 0.9881 | 0.8849 | 0.5275 | 148.0000 | 55.5000 | 0.7539 |
| linear_sparse | 2 | 0.7876 | 55.6702 | 119.0026 | 104.1888 | 0.1496 | 0.7143 | 0.6389 | 0.5599 | 108.0000 | 45.0000 | 1.1383 |
| nonlinear_signal | 2 | 0.5801 | 1.5776 | 2.4551 | 2.2907 | 0.0694 | 0.7937 | 0.6825 | 0.5634 | 126.0000 | 52.5000 | 1.0175 |

## Interpretation

- Positive `region_rmse_lift_vs_mean` means region labels produce better region-mean estimates than a train-mean baseline.
- Positive target spread reduction means covered observations are grouped into regions with tighter target values than the overall target spread.
- `test_rule_coverage` counts observations assigned to any learned rule.
- `test_known_region_coverage` counts observations assigned to a region label observed during training; unknown labels use the train mean in the all-row RMSE.
- The absence of classification-style target warnings confirms that regression feature selection is not routed through class-based scoring.

## Raw Outputs

- `metrics.csv`: one row per dataset and seed.
- `summary.csv`: median metrics per dataset.
