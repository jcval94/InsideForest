![InsideForest](./data/inside_f1_1.jpeg)

# InsideForest

InsideForest es una técnica de **clustering supervisado** construida sobre bosques de decisión para identificar y describir categorías dentro de un conjunto de datos. Descubre regiones relevantes, asigna etiquetas y produce descripciones interpretables.

*El clustering supervisado* agrupa observaciones utilizando la variable objetivo para guiar la segmentación. En lugar de dejar que el algoritmo encuentre grupos por sí solo, las etiquetas existentes orientan la búsqueda de patrones coherentes.

Ya sea que trabajes con datos de clientes, ventas u otra fuente, la biblioteca te ayuda a comprender tu información y tomar decisiones informadas.

## Casos de uso de ejemplo

- Analizar el comportamiento de los clientes para identificar segmentos rentables.
- Clasificar pacientes por historial médico y síntomas.
- Evaluar canales de marketing usando el tráfico del sitio web.
- Construir sistemas de reconocimiento de imágenes más precisos.

## Beneficios

Construir y analizar un bosque aleatorio con InsideForest revela tendencias ocultas y proporciona **insights** que respaldan decisiones de negocio.

[CASO DE USO](https://colab.research.google.com/drive/11VGeB0V6PLMlQ8Uhba91fJ4UN1Bfbs90?usp=sharing)

## Instalación

```bash
pip install InsideForest
```

## Dependencias principales
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- openai

## Benchmark

Ejecuta el benchmark comparativo contra KMeans y DBSCAN:

```bash
python -m experiments.benchmark
```

Este comando imprime tablas por cada conjunto de datos con `purity`,
diversas variantes de F1, exactitud, métricas de información y tiempo de
ejecución. Se incluyen los datasets Digits, Iris, Wine y un conjunto
sintético grande. La siguiente tabla muestra los resultados obtenidos en
este entorno (Titanic se omite por restricciones de descarga de datos):

| Dataset | Algorithm | Purity | Macro F1 | Accuracy | Nmi | Ami | Ari | Bcubed F1 | Divergence | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Digits | InsideForest | 0.216 | 0.113 | 0.216 | 0.195 | 0.191 | 0.064 | 0.266 | 0.279 | 71.623 |
| Digits | KMeans(k=10) | 0.673 | 0.620 | 0.666 | 0.672 | 0.669 | 0.531 | 0.633 | 0.711 | 0.049 |
| Digits | DBSCAN(eps=0.5,min=5) | 0.102 | 0.018 | 0.102 | 0.000 | 0.000 | 0.000 | 0.182 | 0.000 | 0.013 |
| Iris | InsideForest | 0.439 | 0.365 | 0.439 | 0.152 | 0.125 | 0.027 | 0.534 | 0.128 | 2.269 |
| Iris | KMeans(k=3) | 0.667 | 0.531 | 0.580 | 0.590 | 0.584 | 0.433 | 0.710 | 0.427 | 0.003 |
| Iris | DBSCAN(eps=0.5,min=5) | 0.680 | 0.674 | 0.680 | 0.511 | 0.505 | 0.442 | 0.651 | 0.402 | 0.003 |
| Wine | InsideForest | 0.397 | 0.245 | 0.388 | 0.009 | -0.004 | 0.001 | 0.488 | 0.029 | 5.311 |
| Wine | KMeans(k=3) | 0.966 | 0.967 | 0.966 | 0.876 | 0.875 | 0.897 | 0.937 | 0.628 | 0.003 |
| Wine | DBSCAN(eps=0.5,min=5) | 0.399 | 0.190 | 0.399 | 0.000 | 0.000 | 0.000 | 0.509 | 0.000 | 0.003 |
| SyntheticLarge | InsideForest | 0.202 | 0.071 | 0.202 | 0.000 | -0.000 | -0.000 | 0.333 | 0.002 | 340.066 |
| SyntheticLarge | KMeans(k=5) | 0.408 | 0.405 | 0.408 | 0.151 | 0.150 | 0.114 | 0.292 | 0.277 | 0.082 |
| SyntheticLarge | DBSCAN(eps=0.5,min=5) | 0.201 | 0.067 | 0.201 | 0.000 | 0.000 | 0.000 | 0.333 | 0.000 | 0.082 |

Los resultados de Titanic requieren descargar el dataset; ejecuta el
benchmark localmente con acceso a la red para reproducirlos.

### Rendimiento de la búsqueda de `eps`

La versión optimizada de `get_eps_multiple_groups_opt` precalcula la
matriz de distancias y usa una búsqueda binaria para explorar valores de
`eps` con menos iteraciones. En un conjunto sintético de 200 muestras, la
exploración previa en rejilla promedió **0.061 s** por llamada, mientras
que la versión optimizada terminó en **0.044 s**, obteniendo una mejora
aproximada de **1.4×** en tiempo.

## Flujo básico
El orden típico para aplicar InsideForest es:
1. Entrenar un modelo de bosque de decisión o `RandomForest`.
2. Usar `Trees.get_branches` para extraer las ramas de cada árbol.
3. Aplicar `Regions.prio_ranges` para priorizar áreas de interés.
4. Vincular cada observación con `Regions.labels`.
5. Opcionalmente interpretar resultados con `generate_descriptions` y `categorize_conditions`.
6. Finalmente, usar utilidades como `Models` y `Labels` para un análisis adicional.

## InsideForestClassifier e InsideForestRegressor
Para un flujo simplificado puedes utilizar las clases `InsideForestClassifier` o
`InsideForestRegressor`, que combinan el entrenamiento del bosque aleatorio y la
asignación de regiones:

Nota: InsideForest está pensado para ejecutarse sobre un subconjunto de los datos, por ejemplo usar el 35% de las observaciones y reservar el 65% restante para otros fines.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from InsideForest import InsideForestClassifier, InsideForestRegressor

iris = load_iris()
X, y = iris.data, iris.target

# Entrena con el 35% de los datos y reserva el resto para análisis posterior
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, train_size=0.35, stratify=y, random_state=15
)

in_f = InsideForestClassifier(
    rf_params={"random_state": 15},
    tree_params={"mode": "py", "n_sample_multiplier": 0.05, "ef_sample_multiplier": 10},
)

in_f.fit(X_train, y_train)
pred_labels = in_f.predict(X_rest)  # etiquetas de cluster para los datos restantes
etiquetas_entrenamiento = in_f.labels_  # etiquetas para el subconjunto de entrenamiento
```

Puedes controlar cómo se eligen las etiquetas finales mediante el parámetro
`method`. Las estrategias disponibles son:

- `"select_clusters"`: selección directa basada en reglas (por defecto)
- `"balance_lists_n_clusters"`: balancea el número de asignaciones por clúster
- `"max_prob_clusters"`: prioriza los clústers con mayor probabilidad
- `"menu"`: aplica `MenuClusterSelector` para maximizar un objetivo informativo global

Después del ajuste, puedes consultar las importancias de las variables del
bosque aleatorio y visualizarlas opcionalmente:

```python
importancias = in_f.feature_importances_
ejes = in_f.plot_importances()
```

### Guardar y cargar modelos

Las clases `InsideForestClassifier` e `InsideForestRegressor` incluyen
métodos para persistir un modelo entrenado utilizando `joblib`:

```python
in_f.save("modelo.joblib")
cargado = InsideForestClassifier.load("modelo.joblib")
```

El modelo cargado restaura el bosque aleatorio y los atributos
calculados, permitiendo continuar generando etiquetas o predicciones
sin volver a entrenar.

## Caso de uso (Iris)
Lo siguiente resume el flujo utilizado en el [notebook de ejemplo](https://colab.research.google.com/drive/11VGeB0V6PLMlQ8Uhba91fJ4UN1Bfbs90?usp=sharing).

### 1. Preparación del modelo

```python
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder.appName('Iris').getOrCreate()

# Cargar datos en Spark
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

### 2. Obtención de ramas y clusters

```python
pyspark_mod = treesSP.get_branches(df, 'species', model)
rangos_priorizados = regions.prio_ranges(pyspark_mod, df)
clusterized, descriptive = regions.labels(df, rangos_priorizados, False)
```

### 3. Visualización

```python
for rango_df in rangos_priorizados[:3]:
    if len(rango_df['linf'].columns) > 3:
        continue
    regions.plot_multi_dims(rango_df, df, 'species')
```

![Plot 1](./data/plot_1.png)

![Plot 2](./data/plot_2.png)

Las zonas azules resaltan las ramas más relevantes del bosque, revelando dónde se concentra la variable objetivo.

### `Models`

```python
from InsideForest.models import Models

m = Models()
fp_rows, rest = m.get_knn_rows(df_train, 'target', criterio_fp=True)
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5]}
cv_model = m.get_cvRF(X_train, y_train, param_grid)
```

Proporciona métodos para recuperar observaciones críticas con KNN y ajustar un bosque aleatorio con validación cruzada.

### `Labels`

```python
from InsideForest.labels import Labels

lb = Labels()
labels_out = lb.get_labels(rangos_priorizados, df, 'target', max_labels=5)
```

Genera etiquetas descriptivas para las ramas y clusters obtenidos del modelo.

### `plot_experiments`

```python
from InsideForest.regions import Regions
from sklearn.datasets import load_iris
import pandas as pd

# Fila de ejemplo de una tabla de experimentos
experimento = {
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
regions.plot_experiments(df, experimento, interactive=False)
```

Compara los clusters A y B usando las reglas de una fila de la tabla de experimentos.

## Experimentos

El módulo `experiments/benchmark.py` ejecuta comparativas de clustering
supervisado sobre un conjunto de datos de tamaño mediano (`Digits`) y
otro grande generado sintéticamente. Compara `InsideForest` con
baselines tradicionales como KMeans y DBSCAN, reportando pureza,
F1 macro y tiempo de ejecución. También incluye un análisis de
sensibilidad para los hiperparámetros clave: `K` en KMeans y
`eps`/`min_samples` en DBSCAN.

Ejecuta el script con:

```
python -m experiments.benchmark
```

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta [LICENSE](LICENSE) para más detalles.

## Uso de OpenAI para descripciones
`generate_descriptions` de `InsideForest.descrip` usa la biblioteca `openai`. Se requiere una clave API mediante el argumento `OPENAI_API_KEY` o la variable de entorno del mismo nombre.

Usando las condiciones del ejemplo **Iris** puedes generar descripciones automáticas:

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

Generaliza las condiciones de variables numéricas en categorías por niveles.

### `categorize_conditions_generalized`

Ofrece la misma generalización que `categorize_conditions` pero acepta columnas booleanas.

```python
from InsideForest.descrip import categorize_conditions_generalized
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = iris.target
df['large_petal'] = df['petal length (cm)'] > 4

bool_conds = [
    "large_petal == True and 1.0 <= petal width (cm) <= 1.8",
]
categories_bool = categorize_conditions_generalized(bool_conds, df, n_groups=2)
```

### `build_conditions_table`

Construye una tabla ordenada con condiciones categorizadas y sus métricas.

```python
from InsideForest.descrip import build_conditions_table

effectiveness = [0.75]
weights = [len(df)]

table = build_conditions_table(bool_conds, df, effectiveness, weights, n_groups=2)
```

Esto produce un `DataFrame` resumen donde cada condición se etiqueta por grupo junto con la efectividad y el peso proporcionados.

