# Regression Region Validation

This report validates `InsideForestContinuousRegionClusterer` as supervised clustering for continuous targets.
`predict(X)` returns region IDs and `score(X, y)` is eta squared including `-1`; forest R2/RMSE is diagnostic only.

## Reproduction

```bash
python experiments/validate_regression_regions.py --profile quick
```

## Decision Checks

- Median RF test R2 across all runs: `0.6277`.
- Median test eta squared: `0.6459`.
- Median assignment stability ARI: `0.2875`.
- Median selected-feature Jaccard: `1.0000`.
- Median test rule coverage: `1.0000`.
- Median test known-region coverage: `0.9365`.
- Median test target spread reduction inside regions: `0.4810`.
- Median region-mean RMSE lift vs train-mean baseline: `0.2225`.
- Classification-style target warnings observed: `0`.

## Dataset Summary

| Dataset | Runs | Eta2 | Stability ARI | Feature Jaccard | Coverage | Std Reduction | Compression | Regions | RF R2 | Fit s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| diabetes | 2 | 0.6045 | 0.3571 | 0.8889 | 1.0000 | 0.4418 | 0.7991 | 293.5000 | 0.4173 | 0.4705 |
| friedman1 | 2 | 0.6680 | 0.2152 | 1.0000 | 1.0000 | 0.4928 | 0.7925 | 314.0000 | 0.7170 | 0.3648 |
| linear_sparse | 2 | 0.6722 | 0.2535 | 1.0000 | 1.0000 | 0.4983 | 0.7924 | 321.0000 | 0.7876 | 0.3744 |
| nonlinear_signal | 2 | 0.6632 | 0.4318 | 1.0000 | 1.0000 | 0.4993 | 0.7998 | 299.5000 | 0.5801 | 0.8036 |

## Interpretation

- Positive `region_rmse_lift_vs_mean` means region labels produce better region-mean estimates than a train-mean baseline.
- `test_eta_squared` is the canonical score: variance explained by all assigned clusters, including `-1`.
- `assignment_stability_ari` compares holdout assignments from two forest seeds without assuming matching numeric cluster IDs.
- Positive target spread reduction means covered observations are grouped into regions with tighter target values than the overall target spread.
- `test_rule_coverage` counts observations assigned to any learned rule.
- `test_known_region_coverage` counts observations assigned to a region label observed during training; unknown labels use the train mean in the all-row RMSE.
- The absence of classification-style target warnings confirms that regression feature selection is not routed through class-based scoring.

## Raw Outputs

- `metrics.csv`: one row per dataset and seed.
- `summary.csv`: median metrics per dataset.
