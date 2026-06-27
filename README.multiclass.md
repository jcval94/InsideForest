# InsideForest: regiones supervisadas por clase

`InsideForestClassRegionClusterer` extrae regiones enriquecidas para una variable objetivo categórica. El `RandomForestClassifier` interno no es el resultado predictivo: únicamente genera ramas candidatas.

## Contrato

- Cada hoja física conserva su distribución completa de clases.
- La hoja se asocia con la única clase que maximiza `purity_lift_coverage`.
- `predict(X)` devuelve IDs de cluster regional, nunca etiquetas de clase.
- Una fila no cubierta recibe `-1`; no se usa fallback al bosque.
- `labels_` contiene las asignaciones de entrenamiento y `region_metrics_` contiene las métricas de las regiones.
- `branch_aggregation="none"` es el único modo habilitado mientras la agregación de ramas no supere la ablación definida por el proyecto.

## Uso

```python
import pandas as pd
from sklearn.datasets import load_iris

from InsideForest import InsideForestClassRegionClusterer

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

model = InsideForestClassRegionClusterer(
    rf_params={"n_estimators": 50, "max_depth": 6},
    leaf_percentile=95,
    low_leaf_fraction=0.05,
    min_support=2,
    random_state=42,
)

cluster_ids = model.fit_predict(X, y)
assignments = model.assign_regions(X)
regions = model.explain_regions(top_n=10)
setosa_regions = model.regions_for_class(0, top_n=5)
ambiguous = model.ambiguous_regions(top_n=10)
quality = model.region_quality_report()
coverage_by_class = model.class_coverage_report()
```

`transform(X)` devuelve una matriz `n_samples × n_regions`: el score de una región cuando la muestra satisface sus límites y cero en caso contrario. Para filas cubiertas, el índice del máximo coincide con el `cluster_id` retornado por `predict`.

## Salidas

`regions_` incluye:

- `cluster_id` y `representative_region_id`;
- `lower_bounds`, `upper_bounds` y descripción;
- `region_target_class`;
- `class_distribution`, `target_probability`, `lift` y `entropy`;
- `region_score`, `class_margin`, soporte y cobertura;
- `source_region_ids` y `branch_aggregation`.

`assign_regions()` incluye:

- `cluster_id`;
- `representative_region_id`;
- `region_target_class` como metadato supervisado, no como predicción;
- `membership_score`, distribución, lift, entropía y margen;
- regiones coincidentes y `source`, limitado a `region` o `unmatched`.

## Métricas

`score(X, y)` devuelve adjusted mutual information e incluye `-1` como un cluster explícito. `region_quality_report()` añade cobertura, unmatched rate, compresión, pureza, lift, entropía, NMI, AMI, ARI, homogeneidad, completitud y acuerdo diagnóstico entre la clase real y `region_target_class`.

La accuracy del bosque no forma parte del contrato. El bosque ajustado queda disponible como `forest_` para inspección y `feature_importances_` describe exclusivamente ese generador de ramas.

## Compatibilidad

El nombre antiguo sigue disponible temporalmente:

```python
from InsideForest.multiclass import InsideForestMulticlassClassifier
```

Emite `FutureWarning` y traduce los parámetros anteriores:

| Anterior | Canónico |
| --- | --- |
| `percentil` | `leaf_percentile` |
| `low_frac` | `low_leaf_fraction` |
| `max_rules_per_class` | `max_regions_per_class` |
| `score` | `rule_score` |
| `conflict_margin` | `ambiguity_margin` |
| `rf_` | `forest_` |
| `rules_` | `regions_` |
| `explain` | `explain_regions` |
| `prototype_regions` | `regions_for_class` |
| `confusion_regions` | `ambiguous_regions` |

## Validación

Los benchmarks deben tratar el método como clustering supervisado. En cada fold deben reutilizar el mismo bosque y las mismas ramas crudas para comparar variantes y reportar:

- reducción de regiones crudas a finales;
- cobertura y unmatched rate;
- pureza, NMI, AMI, ARI, homogeneidad y completitud;
- lift, entropía y cobertura por clase;
- estabilidad de regiones y asignaciones entre remuestreos;
- tiempo y memoria.

No deben usar accuracy de clasificación como criterio principal ni rellenar regiones ausentes con predicciones del bosque.

Benchmark reproducible:

```bash
python experiments/validate_class_region_clusters.py --profile quick
python experiments/validate_class_region_clusters.py --profile full
```

Los resultados se escriben en `experiments/results/class_region_cluster_validation/`.

Pruebas específicas:

```bash
pytest tests/test_multiclass_rules.py \
       tests/test_multiclass_labels.py \
       tests/test_multiclass_interpreter.py \
       tests/test_region_clusterer_contract.py -q
```
