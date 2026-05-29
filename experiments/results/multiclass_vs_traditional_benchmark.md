# Multiclass vs Traditional Benchmark

Median over 3 runs. Both methods use the same RandomForest parameters: `n_estimators=30`, `max_depth=6`, `random_state=42`, `n_jobs=1`.

| Dataset | Method | Fit s | Assign/Predict s | RF acc | Assignment acc | Rules/Regions | Fallback/Unmatched |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iris | InsideForestClassifier_traditional | 0.3584 | 0.0532 | 1.0000 | 0.7067 | 80 | 0.5933 |
| iris | InsideForestMulticlassClassifier | 0.0727 | 0.1855 | 1.0000 | 0.9800 | 76 | 0.0000 |
| wine | InsideForestClassifier_traditional | 0.8534 | 0.1142 | 1.0000 | 0.6854 | 79 | 0.6236 |
| wine | InsideForestMulticlassClassifier | 0.0899 | 0.2188 | 1.0000 | 0.9888 | 82 | 0.0000 |
| synthetic_3class | InsideForestClassifier_traditional | 2.9232 | 0.5384 | 0.9800 | 0.7844 | 314 | 0.2689 |
| synthetic_3class | InsideForestMulticlassClassifier | 0.2008 | 0.5438 | 0.9800 | 0.9511 | 222 | 0.0000 |

## Ratios

- iris: multiclass fit is 4.93x the speed of traditional; multiclass assignment takes 3.49x traditional predict time.
- wine: multiclass fit is 9.49x the speed of traditional; multiclass assignment takes 1.92x traditional predict time.
- synthetic_3class: multiclass fit is 14.56x the speed of traditional; multiclass assignment takes 1.01x traditional predict time.
