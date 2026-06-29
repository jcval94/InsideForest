![InsideForest](./data/inside_f1_1.jpeg)

# InsideForest

Versión actual: **0.4.1**

InsideForest descubre regiones interpretables en datos tabulares mediante **clustering supervisado**. El Random Forest se usa únicamente para generar hojas candidatas; los estimadores públicos seleccionan regiones útiles, las describen y asignan las observaciones a IDs de clusters regionales.

Los estimadores canónicos son:

- `InsideForestRegionClusterer` para objetivos categóricos generales.
- `InsideForestClassRegionClusterer` cuando cada región debe conservar la distribución completa de clases y diagnósticos específicos por clase.
- `InsideForestContinuousRegionClusterer` para objetivos continuos.

`predict(X)` siempre devuelve IDs de cluster. Una observación fuera de todas las regiones seleccionadas recibe `-1`; no existe fallback a `RandomForest.predict` ni `predict_proba`.

## Instalación

```bash
python -m pip install InsideForest==0.4.1
```

Para desarrollo:

```bash
git clone https://github.com/jcval94/InsideForest.git
cd InsideForest
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

[ABRIR EL NOTEBOOK COMPLETO DEL CASO DE USO DIRECTAMENTE EN COLAB](https://colab.research.google.com/github/jcval94/InsideForest/blob/master/InsideForest/examples/InsideForest_Caso_de_Uso.ipynb)

## Regiones guiadas por clase

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
clusters_train = model.fit_predict(X_train, y_train)
clusters_test = model.predict(X_test)

asignaciones = model.assign_regions(X_test)
scores_regionales = model.transform(X_test)
regiones = model.explain_regions(top_n=10)
calidad = model.region_quality_report(X_test, y_test)
regiones_clase = model.regions_for_class(model.classes_[0], top_n=5)
ambiguas = model.ambiguous_regions(top_n=10)
```

Cada hoja física puede producir como máximo una región final. Su `region_target_class` es la clase que maximiza el objetivo configurado de pureza, lift y cobertura. Los solapamientos se resuelven por score regional, entropía, soporte e ID estable del cluster.

Para objetivos categóricos, `score(X, y)` es información mutua ajustada (AMI), incluido el cluster `-1`. Los reportes también incluyen cobertura, tasa sin asignar, NMI, ARI, homogeneidad, completitud, pureza, lift, entropía y diagnósticos por clase.

## Regiones para objetivos continuos

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

clusters = model.predict(X_test)
asignaciones = model.assign_regions(X_test)
calidad = model.region_quality_report(X_test, y_test)
eta_cuadrada = model.score(X_test, y_test)
```

Las regiones continuas guardan soporte, cobertura, media, mediana, desviación estándar, IQR, rango, desplazamiento del objetivo, reducción de dispersión, separación y score regional. El score canónico es η²: la fracción de varianza del objetivo explicada por todos los clusters devueltos, incluido `-1`. La predicción numérica queda deliberadamente fuera del contrato.

## API compartida y atributos ajustados

Todos los clusterers canónicos exponen:

- `fit`, `fit_predict`, `predict` y `transform`.
- `assign_regions`, `explain_regions` y `region_quality_report`.
- `score`, `get_params`, `set_params`, `save` y `load`.
- `forest_`, `raw_regions_`, `regions_`, `region_metrics_` y asignaciones de entrenamiento en `labels_`.
- `feature_importances_` y `plot_importances`, que describen exclusivamente el bosque generador.

`InsideForestClassRegionClusterer` añade `regions_for_class`, `ambiguous_regions`, `class_coverage_report` y el metadato `classes_`.

## Compatibilidad

`InsideForestClassifier`, `InsideForestMulticlassClassifier` e `InsideForestRegressor` son aliases de migración deprecados. Emiten `FutureWarning`; los aliases de clasificador y regresor conservan temporalmente su comportamiento histórico de `score` basado en el bosque. El código nuevo debe usar los clusterers canónicos.

Los auxiliares históricos de bajo nivel (`Trees`, `Regions`, `Labels`, metadatos y generación de descripciones) siguen disponibles, pero no forman parte del contrato canónico de los estimadores.

## Validación

El benchmark categórico evalúa cobertura, tasa sin asignar, acuerdo de clustering, calidad regional, estabilidad, tiempo y memoria:

```bash
python experiments/validate_class_region_clusters.py --profile quick
```

El benchmark continuo evalúa η², cobertura, reducción de dispersión, estabilidad de asignaciones y geometría, y compresión de ramas. R²/RMSE del bosque se reportan por separado como diagnóstico del generador:

```bash
python experiments/validate_regression_regions.py --profile quick
```

Ejecuta todas las pruebas con:

```bash
python -m pytest tests -q
```

Consulta el [sitio de documentación](https://jcval94.github.io/InsideForest/), la [API rápida](docs/quick_api_es.html), las [páginas de migración](docs/api/index_es.html) y el [registro de v0.4.1](docs/changelog_es.html).

## Licencia

InsideForest se distribuye bajo la [Licencia MIT](LICENSE).
