![InsideForest](./data/inside_f1_1.jpeg)

# InsideForest

InsideForest is a **supervised clustering** technique built on decision forests to identify and describe categories within a dataset. It discovers relevant regions, assigns labels and produces interpretable descriptions.

*Supervised clustering* groups observations using the target variable to guide segmentation. Instead of letting the algorithm find groups on its own, existing labels steer the search for coherent patterns.

Whether you work with customer data, sales or any other source, the library helps you understand your information and make informed decisions.

## Example use cases

- Analyze customer behavior to identify profitable segments.
- Classify patients by medical history and symptoms.
- Evaluate marketing channels using website traffic.
- Build more accurate image-recognition systems.

## Benefits

Building and analyzing a random forest with InsideForest uncovers hidden trends and provides **insights** that support business decisions.

[USE CASE](https://colab.research.google.com/drive/11VGeB0V6PLMlQ8Uhba91fJ4UN1Bfbs90?usp=sharing)

## Installation

```bash
pip install InsideForest
```

## Main dependencies
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- openai

## Benchmark

Run the comparative benchmark against KMeans and DBSCAN:

```bash
python -m experiments.benchmark
```

This command prints tables for each dataset with purity, macro F1,
target-divergence and runtime. Datasets include Digits, Iris, Wine,
Titanic and a synthetic large classification set. The following blocks
show the results obtained in this environment (Titanic is omitted due to
dataset download restrictions):

```
=== Dataset: Digits ===
            algorithm  purity   f1  divergence  runtime
         InsideForest   0.216 0.00       0.279   46.973
         KMeans(k=10)   0.673 0.62       0.711    0.034
DBSCAN(eps=0.5,min=5)   0.102 0.00       0.000    0.009
```

```
=== Dataset: Iris ===
            algorithm  purity    f1  divergence  runtime
         InsideForest   0.439 0.000       0.128    1.644
          KMeans(k=3)   0.667 0.531       0.427    0.006
DBSCAN(eps=0.5,min=5)   0.680 0.000       0.402    0.002
```

```
=== Dataset: Wine ===
            algorithm  purity    f1  divergence  runtime
         InsideForest   0.397 0.000       0.029    3.542
          KMeans(k=3)   0.966 0.967       0.628    0.003
DBSCAN(eps=0.5,min=5)   0.399 0.000       0.000    0.002
```

```
=== Dataset: SyntheticLarge ===
            algorithm  purity    f1  divergence  runtime
         InsideForest   0.202 0.000       0.002  226.617
          KMeans(k=5)   0.408 0.405       0.277    0.025
DBSCAN(eps=0.5,min=5)   0.201 0.000       0.000    0.074
```

Titanic results require downloading the dataset; run the benchmark
locally with network access to reproduce them.

## Basic workflow
The typical order for applying InsideForest is:
1. Train a decision forest or `RandomForest` model.
2. Use `Trees.get_branches` to extract each tree's branches.
3. Apply `Regions.prio_ranges` to prioritize areas of interest.
4. Link each observation with `Regions.labels`.
5. Optionally interpret results with `generate_descriptions` and `categorize_conditions`.
6. Finally, use helpers such as `Models` and `Labels` for further analysis.

## InsideForestClassifier and InsideForestRegressor wrappers
For a simplified workflow you can use the `InsideForestClassifier` or
`InsideForestRegressor` classes, which combine the random forest training and
region labeling steps:

Note: InsideForest is typically run on a subset of the data, for example using 35% of the observations and reserving the remaining 65% for other purposes.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from InsideForest import InsideForestClassifier, InsideForestRegressor

iris = load_iris()
X, y = iris.data, iris.target

# Train on 35% of the data and keep the rest for later analysis
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, train_size=0.35, stratify=y, random_state=15
)

in_f = InsideForestClassifier(
    rf_params={"random_state": 15},
    tree_params={"mode": "py", "n_sample_multiplier": 0.05, "ef_sample_multiplier": 10},
)

in_f.fit(X_train, y_train)
pred_labels = in_f.predict(X_rest)  # cluster labels for the remaining data
training_labels = in_f.labels_  # labels for the training subset
```

After fitting, you can inspect the random forest's feature importances and
optionally visualize them:

```python
importances = in_f.feature_importances_
ax = in_f.plot_importances()
```

### Saving and loading models

Both `InsideForestClassifier` and `InsideForestRegressor` include
convenience methods to persist a fitted instance using `joblib`:

```python
in_f.save("model.joblib")
loaded = InsideForestClassifier.load("model.joblib")
```

The loaded model restores the underlying random forest and computed
attributes, allowing you to continue generating labels or predictions
without re-fitting.

## Use case (Iris)
The following summarizes the flow used in the [example notebook](https://colab.research.google.com/drive/11VGeB0V6PLMlQ8Uhba91fJ4UN1Bfbs90?usp=sharing).

### 1. Model preparation

```python
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder.appName('Iris').getOrCreate()

# Load data into Spark
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
```

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=df.columns[0], y=df.columns[1], hue='species', data=df,
                palette='coolwarm')
plt.show()
```

![Dataset](./data/iris_ds.png)

```python
from InsideForest import Trees, Regions, Labels
treesSP = Trees('pyspark', n_sample_multiplier=0.05, ef_sample_multiplier=10)
regions = Regions()
labels = Labels()
```

### 2. Obtaining branches and clusters

```python
pyspark_mod = treesSP.get_branches(df, 'species', model)
priority_ranges = regions.prio_ranges(pyspark_mod, df)
clusterized, descriptive = regions.labels(df, priority_ranges, False)
```

### 3. Visualization

```python
for range_df in priority_ranges[:3]:
    if len(range_df['linf'].columns) > 3:
        continue
    regions.plot_multi_dims(range_df, df, 'species')
```

![Plot 1](./data/plot_1.png)

![Plot 2](./data/plot_2.png)

The blue areas highlight the most relevant branches of the forest, revealing where the target variable concentrates.

### `Models`

```python
from InsideForest.models import Models

m = Models()
fp_rows, rest = m.get_knn_rows(df_train, 'target', criterio_fp=True)
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5]}
cv_model = m.get_cvRF(X_train, y_train, param_grid)
```

Provides methods for retrieving critical observations with KNN and tuning a random forest with cross-validation.

### `Labels`

```python
from InsideForest.labels import Labels

lb = Labels()
labels_out = lb.get_labels(priority_ranges, df, 'target', max_labels=5)
```

Generates descriptive labels for the branches and clusters obtained from the model.

### `plot_experiments`

```python
from InsideForest.regions import Regions
from sklearn.datasets import load_iris
import pandas as pd

# Example row from an experiments table
experiment = {
    "intersection": "[5.45 <= petal_length <= 8.9]",
    "only_cluster_a": "[-0.9 <= sepal_width <= 1.55, 4.75 <= sepal_length <= 6.0]",
    "only_cluster_b": "[1.0 <= petal_width <= 3.0, 1.7 <= sepal_width <= 3.3]",
    "variables_a": "['sepal_length', 'sepal_width']",
    "variables_b": "['petal_width', 'sepal_length', 'sepal_width']"
}

iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=[c.replace(' (cm)', '').replace(' ', '_') for c in iris.feature_names]
)

regions = Regions()
regions.plot_experiments(df, experiment, interactive=False)
```

Compares clusters A and B using the rules provided by a row from the experiments table.

## Experiments

The `experiments/benchmark.py` module runs supervised clustering
benchmarks on a medium sized dataset (`Digits`) and on a synthetically
generated large dataset. It compares `InsideForest` with traditional
baselines like KMeans and DBSCAN, reporting purity, macro F1-score and
runtime for each method. It also performs a basic sensitivity analysis
on key hyperparameters: `K` for KMeans and `eps`/`min_samples` for
DBSCAN.

Execute the script with:

```
python -m experiments.benchmark
```

## License

This project is distributed under the MIT license. See [LICENSE](LICENSE) for details.

## Using OpenAI for descriptions
`generate_descriptions` from `InsideForest.descrip` uses the `openai` library. An API key is required either through the `OPENAI_API_KEY` argument or the environment variable of the same name.

Using the **Iris** example conditions you can generate automatic descriptions:

```python
from InsideForest.descrip import generate_descriptions
import os

iris_conds = [
    "4.3 <= sepal length (cm) <= 5.8 and 1.0 <= petal width (cm) <= 1.8"
]
os.environ["OPENAI_API_KEY"] = "sk-your-key"
res = generate_descriptions(iris_conds, OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"))
```

### `categorize_conditions`

```python
from InsideForest.descrip import categorize_conditions
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = iris.target

categories = categorize_conditions(iris_conds, df, n_groups=3)
```

Generalizes numeric variable conditions into level-based categories.

### `categorize_conditions_generalized`

Offers the same generalization as `categorize_conditions` but accepts boolean columns.

```python
from InsideForest.descrip import categorize_conditions_generalized
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = iris.target
df['large_petal'] = df['petal length (cm)'] > 4

bool_conds = [
    "large_petal == True and 1.0 <= petal width (cm) <= 1.8"
]
categories_bool = categorize_conditions_generalized(bool_conds, df, n_groups=2)
```

### `build_conditions_table`

Builds a tidy table with categorized conditions and their metrics.

```python
from InsideForest.descrip import build_conditions_table

effectiveness = [0.75]
weights = [len(df)]

table = build_conditions_table(bool_conds, df, effectiveness, weights, n_groups=2)
```

This produces a summary `DataFrame` where each condition is tagged by group along with the provided effectiveness and weight.

## Tests

Latest test run:

```bash
pytest -q
```

```
41 passed
```

