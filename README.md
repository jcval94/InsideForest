![InsideForest](./data/inside_f1_1.jpeg)

# InsideForest

InsideForest es una técnica de **clustering supervisado** que se apoya en bosques de decisión para identificar y describir categorías dentro de un conjunto de datos. Permite descubrir regiones relevantes, asignar etiquetas y generar descripciones interpretables de forma sencilla.

El *clustering supervisado* consiste en agrupar observaciones utilizando información de la variable objetivo para guiar el proceso de segmentación. En lugar de dejar que el algoritmo encuentre los grupos por sí mismo, las etiquetas existentes orientan la búsqueda de patrones coherentes.

Sea que trabajes con datos de clientes, ventas u otra fuente de información, la biblioteca te ayudará a comprender mejor tus datos y tomar decisiones informadas.

## Ejemplos de uso

- Analizar el comportamiento de clientes para identificar segmentos rentables.
- Clasificar pacientes según su historial médico y síntomas.
- Evaluar canales de marketing a partir del tráfico de un sitio web.
- Generar sistemas de reconocimiento de imágenes más precisos.

## Beneficios

Al construir y analizar un bosque aleatorio con InsideForest puedes identificar tendencias ocultas y obtener **insights** que faciliten tus decisiones de negocio.

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

## Flujo de trabajo básico
El orden típico para aplicar InsideForest es:
1. Entrenar un modelo de bosques de decisión o `RandomForest`.
2. Usar `Trees.get_branches` para extraer las ramas de cada árbol.
3. Aplicar `Regions.prio_ranges` para priorizar las zonas de interés.
4. Asociar cada observación con `Regions.labels`.
5. Opcionalmente, interpretar con `generate_descriptions` y `categorize_conditions`.
6. Finalmente, emplear utilidades como `Models` y `Labels` si se requieren análisis adicionales.
## Caso de uso (Iris)
A continuación se muestra un resumen del flujo utilizado en el [notebook de ejemplo](https://colab.research.google.com/drive/11VGeB0V6PLMlQ8Uhba91fJ4UN1Bfbs90?usp=sharing).

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
arbolesSP = Trees('pyspark', n_sample_multiplier=0.05, ef_sample_multiplier=10)
regiones = Regions()
descript = Labels()
```

### 2. Obtención de ramas y clusters

```python
pyspark_mod = arbolesSP.get_branches(df, 'species', model)
df_reres = regiones.prio_ranges(pyspark_mod, df)
clusterizados, descriptivos = regiones.labels(df, df_reres, False)
```

### 3. Visualización

```python
for df_r in df_reres[:3]:
    if len(df_r['linf'].columns) > 3:
        continue
    regiones.plot_multi_dims(df_r, df, 'species')
```

![Plot 1](./data/plot_1.png)

![Plot 2](./data/plot_2.png)

Las zonas azules representan las ramas más relevantes del bosque y permiten interpretar dónde se concentra la variable objetivo.

### `Models`

```python
from InsideForest.models import Models

m = Models()
fp_rows, resto = m.get_knn_rows(df_train, 'target', criterio_fp=True)
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5]}
cv_model = m.get_cvRF(X_train, y_train, param_grid)
```

Proporciona métodos para obtener observaciones críticas con KNN y ajustar un bosque aleatorio con validación cruzada.

### `Labels`

```python
from InsideForest.labels import Labels

lb = Labels()
etiquetas = lb.get_labels(df_reres, df, 'target', etq_max=5)
```

Genera etiquetas descriptivas de las ramas y clusters obtenidos del modelo.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más información.

## Uso de OpenAI para descripciones
`generate_descriptions` de `InsideForest.descrip` utiliza la librería `openai`. Se requiere una API key en el argumento `OPENAI_API_KEY` o mediante la variable de entorno del mismo nombre.

Usando las condiciones del ejemplo de **Iris** se pueden generar descripciones automáticas:

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

categorias = categorize_conditions(iris_conds, df, n_groups=3)
```

Permite generalizar condiciones de variables numéricas en categorías de nivel.
