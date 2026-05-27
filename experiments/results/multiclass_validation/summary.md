# Multiclass Real Gain Validation

Profile: `quick`. Splits: 2, repeats: 1.

## Decision Checks

- Efficiency: multiclass fit was at least 2x faster in 100.0% of dataset/fold pairs (median ratio 10.18x).
- Predictive regions-only delta vs traditional cluster: median balanced-accuracy delta +0.5006.
- Fallback effect: median balanced-accuracy gain from fallback +0.0000.
- RF baseline comparison: median RF minus multiclass-with-fallback delta +0.0277.
- Negative label permutation median balanced accuracy: multiclass_regions_only=0.3222, multiclass_with_fallback=0.3222, rf_baseline=0.3178, traditional_cluster=0.2333.

## How To Read This

- Treat `traditional_cluster` as supervised clustering, not direct classification.
- Treat `multiclass_regions_only` as the cleanest test of interpretable rules.
- Treat `multiclass_with_fallback` as rules plus RandomForest backstop.
- If `rf_baseline` matches or beats `multiclass_with_fallback`, predictive lift is mostly the forest, not the rule layer.

## Summary Table

| Dataset | Kind | Balanced Acc | Macro F1 | Fit s | Coverage | Fallback | Rules |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| breast_cancer | multiclass_regions_only | 0.9436 | 0.9407 | 0.2391 | 0.9978 | 0.0022 | 38.0000 |
| breast_cancer | multiclass_with_fallback | 0.9407 | 0.9385 | 0.2391 | 1.0000 | 0.0022 | 38.0000 |
| breast_cancer | rf_baseline | 0.9376 | 0.9402 | 0.1315 | 1.0000 | 0.0000 |  |
| breast_cancer | traditional_cluster | 0.4771 | 0.5147 | 3.8801 | 0.7956 | 0.2044 | 126.5000 |
| digits | multiclass_regions_only | 0.8772 | 0.8759 | 1.8729 | 1.0000 | 0.0000 | 651.0000 |
| digits | multiclass_with_fallback | 0.8772 | 0.8759 | 1.8729 | 1.0000 | 0.0000 | 651.0000 |
| digits | rf_baseline | 0.9110 | 0.9118 | 0.1592 | 1.0000 | 0.0000 |  |
| digits | traditional_cluster | 0.2902 | 0.3669 | 5.2280 | 0.8444 | 0.1556 | 115.5000 |
| iris | multiclass_regions_only | 0.9533 | 0.9533 | 0.2214 | 1.0000 | 0.0000 | 50.5000 |
| iris | multiclass_with_fallback | 0.9533 | 0.9533 | 0.2214 | 1.0000 | 0.0000 | 50.5000 |
| iris | rf_baseline | 0.9600 | 0.9600 | 0.1560 | 1.0000 | 0.0000 |  |
| iris | traditional_cluster | 0.3400 | 0.3624 | 0.8626 | 0.4600 | 0.5400 | 40.0000 |
| negative_label_permuted | multiclass_regions_only | 0.3222 | 0.3145 | 0.8004 | 1.0000 | 0.0000 | 172.5000 |
| negative_label_permuted | multiclass_with_fallback | 0.3222 | 0.3145 | 0.8004 | 1.0000 | 0.0000 | 172.5000 |
| negative_label_permuted | rf_baseline | 0.3178 | 0.3157 | 0.2144 | 1.0000 | 0.0000 |  |
| negative_label_permuted | traditional_cluster | 0.2333 | 0.2771 | 11.0763 | 0.9978 | 0.0022 | 222.0000 |
| negative_permuted_columns | multiclass_regions_only | 0.8045 | 0.8045 | 0.6939 | 1.0000 | 0.0000 | 143.5000 |
| negative_permuted_columns | multiclass_with_fallback | 0.8045 | 0.8045 | 0.6939 | 1.0000 | 0.0000 | 143.5000 |
| negative_permuted_columns | rf_baseline | 0.8379 | 0.8385 | 0.2238 | 1.0000 | 0.0000 |  |
| negative_permuted_columns | traditional_cluster | 0.3136 | 0.3993 | 7.1902 | 0.9133 | 0.0867 | 158.5000 |
| negative_remapped_ids | multiclass_regions_only | 0.8089 | 0.8091 | 0.6525 | 1.0000 | 0.0000 | 132.0000 |
| negative_remapped_ids | multiclass_with_fallback | 0.8089 | 0.8091 | 0.6525 | 1.0000 | 0.0000 | 132.0000 |
| negative_remapped_ids | rf_baseline | 0.8422 | 0.8434 | 0.2238 | 1.0000 | 0.0000 |  |
| negative_remapped_ids | traditional_cluster | 0.2556 | 0.3353 | 7.4835 | 0.8844 | 0.1156 | 158.0000 |
| synthetic_10class | multiclass_regions_only | 0.5579 | 0.5445 | 1.4289 | 1.0000 | 0.0000 | 710.0000 |
| synthetic_10class | multiclass_with_fallback | 0.5579 | 0.5445 | 1.4289 | 1.0000 | 0.0000 | 710.0000 |
| synthetic_10class | rf_baseline | 0.6062 | 0.5998 | 0.1460 | 1.0000 | 0.0000 |  |
| synthetic_10class | traditional_cluster | 0.1949 | 0.2295 | 4.1173 | 0.9422 | 0.0578 | 118.0000 |
| synthetic_3class | multiclass_regions_only | 0.7977 | 0.7978 | 0.7802 | 1.0000 | 0.0000 | 147.0000 |
| synthetic_3class | multiclass_with_fallback | 0.7977 | 0.7978 | 0.7802 | 1.0000 | 0.0000 | 147.0000 |
| synthetic_3class | rf_baseline | 0.8178 | 0.8192 | 0.2284 | 1.0000 | 0.0000 |  |
| synthetic_3class | traditional_cluster | 0.2865 | 0.3739 | 8.7236 | 0.9400 | 0.0600 | 175.5000 |
| synthetic_5class_overlap | multiclass_regions_only | 0.4847 | 0.4836 | 1.1869 | 1.0000 | 0.0000 | 336.5000 |
| synthetic_5class_overlap | multiclass_with_fallback | 0.4847 | 0.4836 | 1.1869 | 1.0000 | 0.0000 | 336.5000 |
| synthetic_5class_overlap | rf_baseline | 0.5342 | 0.5343 | 0.1993 | 1.0000 | 0.0000 |  |
| synthetic_5class_overlap | traditional_cluster | 0.1741 | 0.2346 | 7.2382 | 0.9666 | 0.0334 | 182.0000 |
| synthetic_high_cardinality_onehot | multiclass_regions_only | 0.2737 | 0.2030 | 0.5068 | 1.0000 | 0.0000 | 92.5000 |
| synthetic_high_cardinality_onehot | multiclass_with_fallback | 0.2737 | 0.2030 | 0.5068 | 1.0000 | 0.0000 | 92.5000 |
| synthetic_high_cardinality_onehot | rf_baseline | 0.2924 | 0.2554 | 0.2023 | 1.0000 | 0.0000 |  |
| synthetic_high_cardinality_onehot | traditional_cluster | 0.2637 | 0.1990 | 5.0742 | 0.9978 | 0.0022 | 97.5000 |
| synthetic_imbalance_noise | multiclass_regions_only | 0.4578 | 0.4493 | 0.6774 | 1.0000 | 0.0000 | 186.5000 |
| synthetic_imbalance_noise | multiclass_with_fallback | 0.4578 | 0.4493 | 0.6774 | 1.0000 | 0.0000 | 186.5000 |
| synthetic_imbalance_noise | rf_baseline | 0.4690 | 0.4696 | 0.2482 | 1.0000 | 0.0000 |  |
| synthetic_imbalance_noise | traditional_cluster | 0.2229 | 0.3104 | 5.5228 | 0.8400 | 0.1600 | 120.5000 |
| synthetic_irrelevant_features | multiclass_regions_only | 0.7779 | 0.7780 | 0.7010 | 1.0000 | 0.0000 | 140.0000 |
| synthetic_irrelevant_features | multiclass_with_fallback | 0.7779 | 0.7780 | 0.7010 | 1.0000 | 0.0000 | 140.0000 |
| synthetic_irrelevant_features | rf_baseline | 0.8090 | 0.8102 | 0.2312 | 1.0000 | 0.0000 |  |
| synthetic_irrelevant_features | traditional_cluster | 0.3022 | 0.3545 | 8.4679 | 0.9533 | 0.0467 | 171.5000 |
| titanic_onehot | multiclass_regions_only | 0.7695 | 0.7763 | 0.6388 | 0.9956 | 0.0044 | 111.0000 |
| titanic_onehot | multiclass_with_fallback | 0.7683 | 0.7748 | 0.6388 | 1.0000 | 0.0044 | 111.0000 |
| titanic_onehot | rf_baseline | 0.7630 | 0.7676 | 0.2211 | 1.0000 | 0.0000 |  |
| titanic_onehot | traditional_cluster | 0.2327 | 0.3385 | 13.8175 | 0.6622 | 0.3378 | 328.0000 |
| wine | multiclass_regions_only | 0.9441 | 0.9396 | 0.2427 | 1.0000 | 0.0000 | 48.5000 |
| wine | multiclass_with_fallback | 0.9441 | 0.9396 | 0.2427 | 1.0000 | 0.0000 | 48.5000 |
| wine | rf_baseline | 0.9745 | 0.9728 | 0.1398 | 1.0000 | 0.0000 |  |
| wine | traditional_cluster | 0.3789 | 0.4079 | 1.7249 | 0.5899 | 0.4101 | 55.0000 |
