# Archived pre-clusterer benchmark

This report belongs to the former classifier contract and is retained only to explain the adjacent raw CSV artifact. Its assignment accuracy, RandomForest accuracy, predicted classes, and fallback measurements must not be used to evaluate the current API.

The canonical benchmark is:

```bash
python experiments/validate_class_region_clusters.py --profile quick
```

It compares `InsideForestRegionClusterer` and `InsideForestClassRegionClusterer` using coverage, unmatched rate, AMI, NMI, ARI, homogeneity, completeness, purity, lift, entropy, region count, stability, runtime, and memory. Cluster `-1` is included and no forest fallback is applied.
