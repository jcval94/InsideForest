![InsideForest](./data/inside_f1_1.jpeg)

# InsideForest

Current release: **0.4.1**

InsideForest discovers interpretable regions in tabular data through **supervised clustering**. A random forest is used only to generate candidate leaves; the public estimators select useful regions, describe them, and assign observations to region-cluster IDs.

The canonical estimators are:

- `InsideForestRegionClusterer` for general categorical targets.
- `InsideForestClassRegionClusterer` when each region must retain its complete class distribution and class-specific diagnostics.
- `InsideForestContinuousRegionClusterer` for continuous targets.

`predict(X)` always returns cluster IDs. An observation outside every selected region receives `-1`; there is no fallback to `RandomForest.predict` or `predict_proba`.

## Install

```bash
python -m pip install InsideForest==0.4.1
```

For development:

```bash
git clone https://github.com/jcval94/InsideForest.git
cd InsideForest
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

[OPEN THE COMPLETE USE-CASE NOTEBOOK DIRECTLY IN COLAB](https://colab.research.google.com/github/jcval94/InsideForest/blob/master/InsideForest/examples/InsideForest_Caso_de_Uso.ipynb)

## Class-guided regions

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from InsideForest import InsideForestClassRegionClusterer

X, y = load_wine(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = InsideForestClassRegionClusterer(
    rf_params={"n_estimators": 50, "random_state": 42},
    min_support=2,
    leaf_percentile=90,
    branch_aggregation="none",
)
cluster_ids = model.fit_predict(X_train, y_train)
test_cluster_ids = model.predict(X_test)

assignments = model.assign_regions(X_test)
region_scores = model.transform(X_test)
regions = model.explain_regions(top_n=10)
quality = model.region_quality_report(X_test, y_test)
class_regions = model.regions_for_class(model.classes_[0], top_n=5)
ambiguous = model.ambiguous_regions(top_n=10)
```

Each physical leaf can produce at most one final region. Its `region_target_class` is the class that maximizes the configured purity-lift-coverage objective. Overlaps are resolved by region score, entropy, support, and stable cluster ID.

For categorical targets, `score(X, y)` is adjusted mutual information (AMI), including cluster `-1`. Quality reports also include coverage, unmatched rate, NMI, ARI, homogeneity, completeness, purity, lift, entropy, and class-level diagnostics.

## Continuous-target regions

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from InsideForest import InsideForestContinuousRegionClusterer

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = InsideForestContinuousRegionClusterer(
    rf_params={"n_estimators": 50, "random_state": 42},
    min_support=3,
    leaf_percentile=90,
    branch_aggregation="none",
)
model.fit(X_train, y_train)

cluster_ids = model.predict(X_test)
assignments = model.assign_regions(X_test)
quality = model.region_quality_report(X_test, y_test)
eta_squared = model.score(X_test, y_test)
```

Continuous regions store support, coverage, mean, median, standard deviation, IQR, range, target shift, dispersion reduction, separation, and region score. The canonical score is η²: the fraction of target variance explained by all returned clusters, including `-1`. Numeric target prediction is intentionally outside the clusterer contract.

## Shared API and fitted attributes

All canonical clusterers expose:

- `fit`, `fit_predict`, `predict`, and `transform`.
- `assign_regions`, `explain_regions`, and `region_quality_report`.
- `score`, `get_params`, `set_params`, `save`, and `load`.
- `forest_`, `raw_regions_`, `regions_`, `region_metrics_`, and training assignments in `labels_`.
- `feature_importances_` and `plot_importances`, which describe only the branch-generating forest.

`InsideForestClassRegionClusterer` additionally exposes `regions_for_class`, `ambiguous_regions`, `class_coverage_report`, and `classes_` metadata.

## Compatibility

`InsideForestClassifier`, `InsideForestMulticlassClassifier`, and `InsideForestRegressor` are deprecated migration aliases. They emit `FutureWarning`; the classifier/regressor aliases temporarily preserve their legacy forest-based `score` behavior. New code should use the canonical clusterers above.

Historical low-level helpers (`Trees`, `Regions`, `Labels`, metadata utilities, and description helpers) remain available, but they are not the canonical estimator contract.

## Validation

The class-region benchmark evaluates coverage, unmatched rate, clustering agreement, regional quality, stability, runtime, and memory:

```bash
python experiments/validate_class_region_clusters.py --profile quick
```

The continuous benchmark evaluates η², coverage, dispersion reduction, assignment stability, geometric stability, and branch compression. Forest R²/RMSE are reported separately as generator diagnostics:

```bash
python experiments/validate_regression_regions.py --profile quick
```

Run the complete test suite with:

```bash
python -m pytest tests -q
```

See the [documentation site](https://jcval94.github.io/InsideForest/), [quick API](docs/quick_api.html), [migration API pages](docs/api/index.html), and [v0.4.1 changelog](docs/changelog.html) for details.

## License

InsideForest is distributed under the [MIT License](LICENSE).
