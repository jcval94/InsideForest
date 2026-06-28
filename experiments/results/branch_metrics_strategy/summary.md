# Branch metric strategy A/B test

## Metric audit

- `n_sample` is empirical support on the DataFrame passed to InsideForest; it is not geometric hyperrectangle volume.
- `ef_sample` is `mean(y)` for covered rows. For binary labels encoded as 0/1 this is the positive-class rate; it is not a label-invariant multiclass purity metric.
- sklearn node statistics summarize the tree training sample. With bootstrap they are not equivalent to rescoring the full analysis DataFrame.
- Lowest observed native-node support match rate in bootstrap scenarios: 0.288.

## Decision

- Candidate: `shared_prefix`.
- Public-output equivalence: `True`.
- Median speedup: 1.59x.
- Worst observed speedup: 1.08x.
- Adopted in production: `True`.

The native `apply(X)` path is retained only as experimental evidence because full-precision sklearn thresholds can disagree with InsideForest's historical six-decimal exported rules at boundary values.

## Scenario medians

| Scenario | Strategy | Seconds | Speedup | Summary equal | Regions equal | Labels equal |
| --- | --- | ---: | ---: | --- | --- | --- |
| binary_bootstrap | native_apply | 0.4177 | 2.78x | True | True | True |
| binary_bootstrap | shared_prefix | 0.4641 | 3.30x | True | True | True |
| binary_no_bootstrap | native_apply | 0.8121 | 1.79x | True | True | True |
| binary_no_bootstrap | shared_prefix | 0.7235 | 2.36x | True | True | True |
| regression_mixed_bootstrap | native_apply | 0.2621 | 1.50x | True | True | True |
| regression_mixed_bootstrap | shared_prefix | 0.2692 | 1.46x | True | True | True |
| regression_negative_no_bootstrap | native_apply | 0.1352 | 1.72x | True | True | True |
| regression_negative_no_bootstrap | shared_prefix | 0.1593 | 1.46x | True | True | True |
| threshold_boundary | native_apply | 0.1192 | 2.70x | False | False | True |
| threshold_boundary | shared_prefix | 0.1360 | 2.65x | True | True | True |
