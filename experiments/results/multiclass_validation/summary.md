# Archived pre-clusterer validation

The files in this directory were generated under the removed class-prediction and RandomForest-fallback contract. They are historical raw evidence only; balanced accuracy, macro F1, predicted-class fields, and fallback rates do not describe `InsideForestClassRegionClusterer`.

For the current supervised region-clustering contract, run:

```bash
python experiments/validate_class_region_clusters.py --profile quick
```

The current decision criteria are coverage, unmatched rate, AMI/NMI/ARI, homogeneity, completeness, region purity/lift/entropy, class coverage, geometry and assignment stability, compression, runtime, and memory.
