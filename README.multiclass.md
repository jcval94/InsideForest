# InsideForest Multiclass

Esta version agrega una capa opt-in para interpretacion multiclase sin reemplazar el flujo tradicional de InsideForest. La API nueva vive en `InsideForest.multiclass` y conserva, por hoja del bosque, el vector completo de probabilidades por clase.

## Uso Rapido

```python
import pandas as pd
from sklearn.datasets import load_iris

from InsideForest.multiclass import InsideForestMulticlassClassifier

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

model = InsideForestMulticlassClassifier(
    rf_params={"n_estimators": 50, "max_depth": 6, "random_state": 42},
    percentil=95,
    min_support=2,
    random_state=42,
)

model.fit(X, y)

rules = model.explain(top_n=10)
assignments = model.assign_regions(X)
prototypes = model.prototype_regions(top_n=3)
conflicts = model.confusion_regions(top_n=10)
```

## Que Cambia

- `class_distribution`: cada regla guarda `P(y=k | leaf)` para todas las clases.
- `score`: se calcula con pureza, cobertura y lift; no usa el valor numerico del ID de clase.
- `target_class`: cada hoja se expande en vistas one-vs-rest para rankear reglas por clase.
- `assign_regions`: devuelve clase predicha, confianza, margen top-2 y fallback explicito al bosque.
- `get_multiclass_labels`: resume pureza, entropia, lift y distribucion, sin promediar targets nominales.

## API Publica

```python
from InsideForest.multiclass import (
    InsideForestMulticlassClassifier,
    extract_multiclass_leaf_rules,
    score_multiclass_rules,
    get_multiclass_labels,
)
```

### `InsideForestMulticlassClassifier`

```python
InsideForestMulticlassClassifier(
    rf_params=None,
    percentil=95,
    low_frac=0.05,
    min_support=1,
    max_rules_per_class=None,
    score="purity_lift_coverage",
    random_state=42,
    n_jobs=1,
    conflict_margin=0.15,
)
```

Metodos principales:

- `fit(X, y, rf=None)`: entrena o recibe un `RandomForestClassifier` ya ajustado.
- `explain(class_label=None, top_n=None)`: devuelve reglas rankeadas.
- `assign_regions(X)`: asigna cada fila a la mejor region o usa `model_fallback`.
- `prototype_regions(class_label=None, top_n=10)`: mejores regiones por clase.
- `confusion_regions(top_n=20)`: regiones con margen bajo entre las dos clases mas probables.

## Columnas Clave

`explain()` devuelve, entre otras:

- `region_id`
- `description`
- `class_counts`
- `class_distribution`
- `dominant_class`
- `dominant_probability`
- `target_class`
- `target_probability`
- `prior_probability`
- `lift`
- `entropy`
- `js_divergence`
- `score`

`assign_regions()` devuelve:

- `region_id`
- `predicted_class`
- `confidence`
- `score`
- `matched_region_count`
- `second_class`
- `margin`
- `is_conflict`
- `source`: `"region"` o `"model_fallback"`

## Ejemplo Con Etiquetas String

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from InsideForest.multiclass import InsideForestMulticlassClassifier

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = np.array([iris.target_names[i] for i in iris.target])

model = InsideForestMulticlassClassifier(
    rf_params={"n_estimators": 40, "random_state": 7},
    random_state=7,
)
model.fit(X, y)

setosa_rules = model.explain(class_label="setosa", top_n=5)
row_assignments = model.assign_regions(X.head())
```

## Ejemplo Con Funciones De Bajo Nivel

```python
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

from InsideForest.multiclass import (
    extract_multiclass_leaf_rules,
    get_multiclass_labels,
)

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=6,
    random_state=42,
).fit(X, y)

rules = extract_multiclass_leaf_rules(
    rf,
    X,
    y,
    percentil=95,
    min_support=2,
)

labels = get_multiclass_labels(rules, X, y, class_labels=rf.classes_)
```

## Benchmark

El benchmark reproducible esta en:

- `experiments/multiclass_vs_traditional_benchmark.py`
- `experiments/results/multiclass_vs_traditional_benchmark.csv`
- `experiments/results/multiclass_vs_traditional_benchmark.md`

Comando:

```bash
python experiments/multiclass_vs_traditional_benchmark.py
```

Resultados medidos en esta maquina, mediana de 3 corridas:

| Dataset | Metodo | Fit s | Assign/Predict s | RF acc | Assignment acc | Rules/Regions | Fallback/Unmatched |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iris | InsideForestClassifier tradicional | 0.3584 | 0.0532 | 1.0000 | 0.7067 | 80 | 0.5933 |
| iris | InsideForestMulticlassClassifier | 0.0727 | 0.1855 | 1.0000 | 0.9800 | 76 | 0.0000 |
| wine | InsideForestClassifier tradicional | 0.8534 | 0.1142 | 1.0000 | 0.6854 | 79 | 0.6236 |
| wine | InsideForestMulticlassClassifier | 0.0899 | 0.2188 | 1.0000 | 0.9888 | 82 | 0.0000 |
| synthetic_3class | InsideForestClassifier tradicional | 2.9232 | 0.5384 | 0.9800 | 0.7844 | 314 | 0.2689 |
| synthetic_3class | InsideForestMulticlassClassifier | 0.2008 | 0.5438 | 0.9800 | 0.9511 | 222 | 0.0000 |

Lectura rapida:

- El fit multiclase fue entre 4.93x y 14.56x mas rapido en estos datasets.
- La asignacion multiclase puede ser mas costosa que `predict` tradicional en datasets chicos, porque evalua regiones y calcula conflicto por clase.
- La accuracy de asignacion multiclase es directa contra la clase real; la tradicional usa labels de cluster y se reporta alineada por mayoria para poder compararla.

## Validacion De Si La Mejora Es Real

La mejora inicial no debe leerse como una prueba concluyente de "mejor modelo predictivo". Para separar eficiencia, calidad predictiva e interpretabilidad se agrego una validacion de holdout con `RepeatedStratifiedKFold`:

- Script: `experiments/validate_multiclass_real_gain.py`
- Resultados: `experiments/results/multiclass_validation/`
- Metricas por fold: `experiments/results/multiclass_validation/fold_metrics.csv`
- Resumen agregado: `experiments/results/multiclass_validation/summary.csv`
- Lectura ejecutiva: `experiments/results/multiclass_validation/summary.md`
- Graficas: `experiments/results/multiclass_validation/figures/`
- Matrices de confusion: `experiments/results/multiclass_validation/confusion_matrices/`

Comando rapido:

```bash
python experiments/validate_multiclass_real_gain.py --profile quick
```

Comando mas pesado:

```bash
python experiments/validate_multiclass_real_gain.py --profile full
```

Tambien se pueden acotar datasets y folds:

```bash
python experiments/validate_multiclass_real_gain.py --datasets iris wine titanic_onehot --n-splits 2 --n-repeats 1
```

### Que Compara

Cada fold entrena solo con train y evalua en test:

- `rf_baseline`: `RandomForestClassifier` puro.
- `traditional_cluster`: `InsideForestClassifier` tradicional tratado como clustering supervisado.
- `multiclass_regions_only`: solo filas cubiertas por regiones multiclase.
- `multiclass_with_fallback`: regiones multiclase mas fallback explicito a `rf.predict_proba`.

Las metricas predictivas incluyen accuracy, balanced accuracy, macro-F1 y weighted-F1. Las metricas interpretativas incluyen coverage, pureza ponderada, entropia, lift, numero de reglas y estabilidad aproximada de features entre folds. Para el flujo tradicional se reportan purity, NMI, AMI, ARI, homogeneity, completeness y accuracy alineada como metrica de cluster.

### Datasets Incluidos

El perfil rapido corre:

- Built-in: Iris, Wine, Breast Cancer, Digits.
- Seaborn/local: Titanic one-hot.
- Sinteticos: 3 clases, 5 clases solapadas, 10 clases, imbalance/ruido, features irrelevantes, high-cardinality one-hot.
- Pruebas negativas: labels permutados, IDs remapeados y columnas permutadas.

### Resultado De Referencia

Ultima corrida local del perfil `quick`: 14 datasets/pruebas, 2 folds, 4 comparaciones por fold, 112 filas de metricas.

| Check | Resultado |
| --- | ---: |
| Folds donde multiclase fue al menos 2x mas rapido que tradicional | 28/28 |
| Ratio mediano de fit tradicional / fit multiclase | 10.18x |
| Folds donde `multiclass_regions_only` supera a `traditional_cluster` en balanced accuracy | 27/28 |
| Delta mediano de balanced accuracy, regiones multiclase vs tradicional | +0.5006 |
| Ganancia mediana por fallback | +0.0000 |
| Folds donde RF puro iguala o supera a multiclase con fallback | 23/28 |
| Balanced accuracy mediana con labels permutados | 0.3222 |

Lectura: la mejora de eficiencia parece real bajo esta validacion. La mejora predictiva frente al tradicional tambien aparece en test, pero esa comparacion no es entre dos clasificadores equivalentes: el tradicional produce clusters y la nueva capa produce clases. El fallback no fue el factor principal en esta corrida, pero el RF puro suele igualar o superar a la version multiclase con fallback; por eso la claim correcta es interpretabilidad multiclase eficiente con performance cercana al bosque, no superioridad predictiva general sobre RandomForest.

### Criterios De Decision

- La eficiencia se considera real si multiclase mantiene una mediana al menos 2x mas rapida en 80% de datasets/folds.
- La mejora predictiva se considera real solo si `multiclass_regions_only` supera al tradicional en macro-F1/balanced accuracy en test.
- Si la mejora solo aparece en `multiclass_with_fallback`, se reporta como mejora explicada por RandomForest.
- Si `rf_baseline` iguala o supera a multiclase, se debe vender como interpretacion con reglas, no como mejora predictiva.
- Si labels permutados siguen dando metricas altas, hay leakage o metrica mal definida y se bloquea cualquier claim.

### Graficas Generadas

- `balanced_accuracy_distribution.png`
- `macro_f1_distribution.png`
- `fit_seconds_distribution.png`
- `coverage_distribution.png`
- `coverage_vs_accuracy.png`
- `fallback_rate_vs_accuracy.png`
- `runtime_vs_rules.png`

## Validacion

Suite completa:

```powershell
$bt = Join-Path (Resolve-Path .) ('.tmp\pytest-full-' + [guid]::NewGuid().ToString('N'))
pytest tests -q --basetemp $bt
```

Resultado de referencia:

```text
98 passed, 23 warnings
```

Tests especificos multiclase:

```bash
pytest tests/test_multiclass_metrics.py tests/test_multiclass_rules.py tests/test_multiclass_labels.py tests/test_multiclass_interpreter.py -q
```

## Limitaciones

- V1 soporta `sklearn.ensemble.RandomForestClassifier`.
- PySpark queda fuera de esta capa porque `toDebugString` no expone de forma confiable el vector completo por hoja.
- La API no se exporta desde `InsideForest.__init__`; se usa de forma explicita con `InsideForest.multiclass`.
- El flujo tradicional sigue existiendo y no se reemplaza.
