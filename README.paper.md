# Clustering supervisado

## Resumen
InsideForest es una biblioteca de **clustering supervisado** que aprovecha bosques de decisión para segmentar datos etiquetados y describir cada subgrupo con reglas interpretables. Este documento expone la motivación, la teoría matemática, la arquitectura y el flujo de trabajo de la librería, además de un estudio de caso que muestra sus capacidades.

## 1. Introducción
El clustering tradicional intenta descubrir grupos homogéneos sin considerar la variable objetivo; InsideForest adopta un enfoque distinto: utiliza la información de las clases para guiar la búsqueda de regiones relevantes. Esta estrategia permite:

- Identificar segmentos con comportamiento coherente respecto a la variable objetivo.
- Generar reglas explícitas y comprensibles para cada conglomerado.
- Facilitar acciones de negocio o ciencia de datos basadas en segmentos bien definidos.

### Objetivos y formulación del problema
Sea un conjunto de datos $D = \{(x_i, y_i)\}_{i=1}^n$, donde cada vector de características $x_i \in \mathbb{R}^d$ y su etiqueta correspondiente $y_i$ pertenece al conjunto finito de clases $\{1,\ldots,K\}$. El objetivo del **clustering supervisado** es encontrar una partición $\mathcal{C} = \{C_1,\ldots,C_M\}$ de $D$ tal que cada cluster $C_m$ agrupe observaciones con patrones homogéneos respecto a $y_i$ y pueda describirse mediante reglas interpretables.

Las hipótesis a evaluar son:

1. Las regiones identificadas por InsideForest alcanzan una pureza media significativamente superior a la proporción base de las clases.
2. La cobertura de los clusters válidos sobre el conjunto de datos es suficientemente alta para apoyar análisis posteriores.

El éxito del método se medirá con los siguientes criterios cuantitativos:

- $\text{pureza media} \geq 0.80$.
- $\text{cobertura} \geq 70\%$ de las observaciones de $D$ asignadas a clusters.

## 2. Fundamentos teóricos
### 2.1 Bosques de decisión
Sea un conjunto de datos $D = \{(x_i, y_i)\}_{i=1}^n$ con $x_i \in \mathbb{R}^d$ y $y_i$ una etiqueta discreta en $\{1,\ldots,K\}$. Un árbol de decisión particiona recursivamente el espacio de variables buscando maximizar la ganancia de información. En cada nodo se evalúan posibles divisiones $(j, t)$, donde $j$ es el índice de la característica y $t$ un umbral, y se selecciona la que reduce más la impureza:

$$
\Delta I = I(D) - \frac{n_L}{n}I(D_L) - \frac{n_R}{n}I(D_R),
$$

donde $I$ es una medida como la **impureza de Gini**

$$
G(D) = 1 - \sum_{k=1}^K p_{k}^2, \qquad p_k = \frac{1}{|D|}\sum_{(x_i,y_i)\in D} \mathbf{1}_{y_i=k}.
$$

Un **RandomForest** entrena $B$ árboles independientes, cada uno con una muestra bootstrap del conjunto de entrenamiento y seleccionando aleatoriamente subconjuntos de características en cada división. La predicción final de probabilidad para la clase $k$ se obtiene promediando las salidas de cada árbol:

$$
\hat{P}(y = k \mid x) = \frac{1}{B}\sum_{b=1}^B \hat{P}_b(y=k\mid x).
$$

### 2.2 Reglas como hiperrectángulos
Cada camino desde la raíz hasta una hoja define una **regla** $R$ que puede representarse como un hiperrectángulo en $\mathbb{R}^d$:

$$
R = \{x \in \mathbb{R}^d \mid a_j \le x_j < b_j \ \text{para todo } j \in S\},
$$

donde $S$ es el subconjunto de variables presentes en la ruta y $[a_j, b_j)$ los intervalos derivados de las condiciones de los nodos. Cada regla almacena además el número de observaciones y la proporción de cada clase en la hoja correspondiente.

### 2.3 Similitud entre regiones
Para comparar dos reglas $R_i$ y $R_j$ se calcula el **grado de solapamiento** mediante el índice de Jaccard volumétrico:

$$
J(R_i, R_j) = \frac{\text{vol}(R_i \cap R_j)}{\text{vol}(R_i \cup R_j)},
$$

donde la intersección y la unión se definen por intervalos coordenada a coordenada. También se consideran distancias centroideas si las reglas no se solapan: $d(R_i,R_j) = \lVert c_i - c_j \rVert_2$ con $c_i$ el centro del hiperrectángulo.

### 2.4 Clustering de regiones
Las reglas se embeben en un espacio de $2d$ dimensiones usando los extremos $(a_1,b_1,\ldots,a_d,b_d)$. Sobre estos vectores se aplican algoritmos como:

- **DBSCAN**, que agrupa puntos densos. Dos reglas pertenecen al mismo cluster si existe una cadena de vecinos con distancia menor que $\varepsilon$ y cada punto tiene al menos `minPts` vecinos.
- **KMeans**, que minimiza la suma de distancias cuadráticas al centroide:

$$
\min_{C_1,\ldots,C_K} \sum_{k=1}^K \sum_{r_i \in C_k} \lVert r_i - \mu_k \rVert_2^2.
$$

### 2.5 Etiquetado y estadísticas
Una vez formados los clusters de reglas, cada región agregada $C$ se resume mediante la proporción de la clase objetivo:

$$
\pi_C = \frac{1}{|C|}\sum_{(x_i,y_i)\in C} \mathbf{1}_{y_i = 1},
$$

lo que permite priorizar segmentos con mayor pureza o soporte. Estas métricas se traducen en etiquetas textuales que describen los intervalos dominantes y su comportamiento estadístico.

## 3. Arquitectura del repositorio
La organización del código refleja el proceso anterior en módulos especializados:

### 3.1 Módulo `Trees`
- Traduce cada árbol de un `RandomForest` en una lista de reglas hiperrectangulares.
- Optimiza los intervalos para representar la información mínima necesaria.
- Soporta implementaciones en pandas y PySpark para escalabilidad.

### 3.2 Módulo `Regions`
- Recibe las reglas y calcula matrices de similitud basadas en intersección/volumen.
- Aplica algoritmos de clustering (DBSCAN, KMeans) para consolidar regiones.
- Genera fronteras, asigna observaciones a regiones y construye reportes comparativos.

### 3.3 Módulo `Labels`
- Convierte cada intervalo numérico en descripciones textuales (“$x_j$ entre $a$ y $b$”).
- Calcula estadísticas de población, pureza y lift para cada etiqueta.
- Puede integrarse con modelos de lenguaje para producir narrativas más ricas.

### 3.4 `InsideForest` wrapper
- Clase de alto nivel que coordina `Trees`, `Regions` y `Labels`.
- Entrena el `RandomForestClassifier`, extrae reglas, agrupa regiones y etiqueta observaciones.
- Devuelve etiquetas para los datos de entrenamiento y permite predecir etiquetas para nuevos datos.

### 3.5 Módulos auxiliares
- `Models` proporciona utilidades de validación cruzada y métodos vecinos más cercanos.
- `Descrip` genera descripciones en lenguaje natural, generaliza condiciones y construye tablas de interpretación.

## 4. Metodología de uso
El flujo típico para aplicar InsideForest incluye:

1. **Preparación de datos**: se selecciona un subconjunto representativo para entrenar el bosque, normalizando variables según sea necesario.
2. **Entrenamiento**: `InsideForest` entrena un `RandomForest` y almacena los árboles resultantes.
3. **Extracción de ramas**: `Trees.get_branches` transforma cada árbol en reglas de intervalo con información de soporte y pureza.
4. **Priorización de regiones**: `Regions.prio_ranges` agrupa y ordena las reglas según su relevancia, densidad y pureza.
5. **Etiquetado**: `Regions.labels` asigna a cada observación el cluster más representativo mediante comparación volumétrica.
6. **Descripción**: `Labels` y `Descrip` generan explicaciones textuales y visualizaciones cuantitativas.

## 5. Caso de estudio: conjunto Iris
Para ilustrar el flujo, se usa el clásico conjunto `Iris` (150 observaciones, 4 variables numéricas y 3 especies):

1. Se entrena un bosque con el 35% de las observaciones, reservando el resto para validación.
2. Se extraen reglas que describen combinaciones de largo y ancho de sépalo y pétalo.
3. Las reglas se agrupan en tres clusters principales que corresponden aproximadamente a cada especie; cada uno presenta alta pureza ($\pi_C > 0.9$).
4. Las descripciones generadas permiten interpretar qué rangos de variables definen cada flor, por ejemplo: “pétalo largo $< 2$ cm” caracteriza a *Iris setosa*.

Este estudio de caso demuestra la capacidad de InsideForest para producir segmentos interpretables y consistentes con clases conocidas.

## 6. Instalación y dependencias
InsideForest se distribuye como paquete de Python:

```bash
pip install InsideForest
```

Dependencias principales:
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `openai`

## 7. Ventajas y limitaciones
**Ventajas**
- Interpretabilidad gracias a reglas explícitas y métricas cuantificables.
- Flexibilidad para trabajar con datos tabulares o en entornos distribuidos vía PySpark.
- Posibilidad de integrar modelos de lenguaje para enriquecer las explicaciones.

**Limitaciones**
- Requiere datos etiquetados; no reemplaza métodos no supervisados cuando no hay variable objetivo.
- El rendimiento depende de la calidad del bosque y del número de reglas generadas; bosques muy profundos pueden producir reglas redundantes.

## 8. Conclusiones y trabajo futuro
InsideForest ofrece una aproximación completa al clustering supervisado, combinando potencia predictiva con interpretabilidad. Futuros desarrollos incluyen:
- Soporte para modelos de bosque alternativos como gradient boosting.
- Métricas automáticas para evaluar la coherencia de las etiquetas y detectar solapamientos.
- Integración con herramientas de visualización interactiva y paneles de control.

## Licencia
Este proyecto se distribuye bajo la licencia MIT. Consulta [LICENSE](LICENSE) para más detalles.
## 9. Detalle exhaustivo del módulo `Trees`

El módulo `Trees` constituye el núcleo analítico de InsideForest. Su función es traducir la representación interna de un `RandomForestClassifier` en una colección depurada de reglas de decisión. Para comprender su papel es necesario descomponer cada fase del proceso y mostrar cómo cada algoritmo opera sobre los datos originales.

### 9.1 Estructuras internas del árbol de decisión

Cada árbol de scikit-learn se almacena en arreglos planos: `children_left`, `children_right`, `feature`, `threshold` y `value`. El módulo `Trees` aprovecha estas estructuras para reconstruir las rutas desde la raíz hasta cada hoja.

- `children_left[i]` y `children_right[i]` contienen los índices de los nodos descendientes.
- `feature[i]` indica la columna usada para dividir en el nodo `i`; un valor de `-2` marca una hoja.
- `threshold[i]` define el punto de corte para la característica seleccionada.
- `value[i]` almacena un vector con el conteo de observaciones por clase en la hoja.

Internamente, el algoritmo recorre estos arreglos mediante un recorrido en profundidad (DFS) que acumula condiciones a medida que avanza. El estado del recorrido se representa con una pila de tuplas `(nodo, intervalo_actual)`, donde `intervalo_actual` es un diccionario que guarda para cada característica el rango permitido por las decisiones anteriores.

### 9.2 Representación de los intervalos

Cada intervalo se almacena como un par `(min, max)` que puede extenderse o acotarse en función de las decisiones del árbol. Al iniciar el recorrido, todas las características se inicializan en `(-inf, inf)`. Cuando se atraviesa un nodo que divide sobre la característica `j` con umbral `t`, el intervalo se modifica:

- Si se toma la rama izquierda, el límite superior se actualiza a `t`.
- Si se toma la rama derecha, el límite inferior se actualiza a `t`.

El algoritmo se asegura de clonar el diccionario de intervalos en cada bifurcación para evitar interferencias entre ramas hermanas. Esta estrategia implica un costo en memoria proporcional a la profundidad del árbol, pero garantiza la independencia de cada ruta.

### 9.3 Extracción de reglas en forma de hiperrectángulos

Cuando el recorrido alcanza una hoja, el intervalo acumulado se convierte en un hiperrectángulo. El objeto resultante incluye:

1. `bounds`: mapa de característica a par `(min, max)`.
2. `support`: número de observaciones que cayeron en esa hoja.
3. `distribution`: vector de probabilidades por clase calculado a partir de `value`.

Los hiperrectángulos permiten interpretar cada hoja como una región del espacio de características. Dentro de `Trees`, se opta por almacenar únicamente las características que realmente aparecen en la ruta, omitiendo aquellas que permanecen en `(-inf, inf)`; esta decisión reduce el tamaño de la representación y facilita su posterior manipulación.

### 9.4 Optimización de intervalos y poda de reglas

Para evitar redundancias, `Trees` implementa un mecanismo de poda basado en tres criterios principales:

- **Soporte mínimo**: se descartan reglas cuyo `support` es inferior a un umbral `min_samples` configurable.
- **Pureza mínima**: se exige que la proporción de la clase mayoritaria supere un valor `min_purity`.
- **Redundancia geométrica**: si dos reglas comparten límites idénticos y distribución similar, se fusionan en una sola entrada.

La fusión utiliza el volumen del hiperrectángulo y la distancia centroide para decidir si dos reglas son virtualmente equivalentes. Esta estrategia mejora la escalabilidad al trabajar con bosques profundos, donde pueden surgir miles de hojas con diferencias irrelevantes.

### 9.5 Pseudocódigo detallado

A continuación se muestra un pseudocódigo exhaustivo que refleja la implementación principal. Se incluye un comentario por línea para explicar la lógica exacta:

```
procedure GET_BRANCHES(tree)
    stack <- [(0, dict())]                     # comienza en la raíz con intervalos vacíos
    rules <- []                                # colección resultante de hiperrectángulos
    while stack not empty do
        node, bounds <- stack.pop()
        if is_leaf(node) then                  # verifica si el nodo es hoja
            distrib <- normalize(value[node])  # convierte conteos a probabilidades
            support <- sum(value[node])        # total de observaciones en la hoja
            rule <- {"bounds": bounds,        # estructura de la regla
                     "support": support,
                     "distribution": distrib}
            if passes_filters(rule) then       # aplica criterios de poda
                rules.append(rule)             # agrega a la lista final
        else
            feat <- feature[node]              # característica utilizada en el nodo
            thr <- threshold[node]             # umbral de división
            left_bounds <- clone(bounds)       # clona intervalos actuales para la rama izq.
            right_bounds <- clone(bounds)      # clona intervalos para la rama der.
            left_bounds[feat] <- (bounds.get(feat, (-inf, inf))[0], thr)
            right_bounds[feat] <- (thr, bounds.get(feat, (-inf, inf))[1])
            stack.push((children_left[node], left_bounds))
            stack.push((children_right[node], right_bounds))
    return rules
end procedure
```

Este pseudocódigo evidencia varios detalles sutiles:

- La función `normalize` divide el vector `value[node]` por su suma, produciendo una distribución de probabilidad.
- `passes_filters` encapsula los criterios de soporte, pureza y redundancia.
- La pila permite recorrer el árbol sin necesidad de recursión, lo que evita desbordamientos en árboles muy profundos.

### 9.6 Ejemplo paso a paso con un árbol simple

Considérese un árbol entrenado sobre dos características `x1` y `x2`. Supongamos la siguiente estructura:

1. Nodo 0 divide `x1 < 5`.
2. Nodo 1 (izquierda de 0) divide `x2 < 3`.
3. Nodo 2 (derecha de 0) es hoja con distribución `[0.2, 0.8]`.
4. Nodo 3 (izquierda de 1) es hoja con `[0.9, 0.1]`.
5. Nodo 4 (derecha de 1) es hoja con `[0.4, 0.6]`.

El recorrido se desarrolla así:

- Se inicia con `stack = [(0, {x1: (-inf, inf), x2: (-inf, inf)})]`.
- Se procesa el nodo 0, generando dos nuevas entradas en la pila:
  - Rama izquierda: `node 1` con `x1: (-inf, 5)`.
  - Rama derecha: `node 2` con `x1: (5, inf)`.
- Se extrae la rama derecha: se detecta que `node 2` es hoja, se construye un hiperrectángulo con `x1 in (5, inf)` y se agrega a `rules`.
- Se continúa con la rama izquierda `node 1`:
  - Divide en `x2 < 3`, generando `node 3` y `node 4`.
  - Para `node 3`, el intervalo es `x1 in (-inf,5)`, `x2 in (-inf,3)`; se agrega a `rules`.
  - Para `node 4`, el intervalo es `x1 in (-inf,5)`, `x2 in (3,inf)`; se agrega a `rules`.

El resultado final contiene tres reglas, cada una con su distribución de clases y soporte.

### 9.7 Complejidad computacional

Sea `L` el número de hojas en el bosque y `d` el número de características. La complejidad del procedimiento `GET_BRANCHES` es:

- **Tiempo**: `O(L * d)` en el peor caso, ya que cada hoja puede contener hasta `d` límites distintos.
- **Memoria**: `O(depth * d)` debido al almacenamiento temporal de intervalos en la pila.

En la práctica, los bosques aleatorios generan árboles poco profundos y `Trees` puede procesar miles de hojas en segundos.

### 9.8 Soporte para PySpark

InsideForest incluye una versión de `Trees` compatible con PySpark para manejar grandes volúmenes de datos. Las diferencias clave incluyen:

- Los arreglos del árbol se almacenan en estructuras de broadcast para evitar replicación excesiva.
- El recorrido se realiza mediante transformaciones `mapPartitions`, permitiendo que cada ejecutor procese un subconjunto de árboles.
- Las reglas resultantes se representan como filas de un `DataFrame` con esquemas explícitos para `bounds`, `support` y `distribution`.

Este diseño explota el paralelismo de Spark y mantiene la semántica del algoritmo original.

### 9.9 Integración con el resto del pipeline

Una vez que `Trees` genera las reglas, estas se trasladan al módulo `Regions`. Los hiperrectángulos sirven como unidades básicas de análisis:

1. `Regions` calcula intersecciones volumétricas entre reglas.
2. Aplica clustering para agrupar reglas con patrones similares.
3. Los clusters se remiten a `Labels`, que produce descripciones legibles.

Sin la etapa de `Trees`, el pipeline carecería de segmentos interpretables; por ello, la calidad de sus reglas afecta directamente a todas las etapas posteriores.

### 9.10 Validación y pruebas internas

El módulo se acompaña de un conjunto de pruebas unitarias que garantizan su correcto funcionamiento:

- Verificación de que el número de reglas coincide con las hojas efectivas.
- Comprobación de que los intervalos no se solapan con valores imposibles.
- Test de equivalencia entre implementaciones pandas y PySpark.

Cada nueva versión del módulo debe pasar por estos tests antes de integrarse al repositorio principal.

### 9.11 Consejos de uso y mejores prácticas

- Ajustar `min_samples` y `min_purity` según el tamaño del conjunto de datos para evitar reglas triviales.
- Limitar la profundidad del bosque ayuda a reducir la explosión combinatoria de reglas.
- Utilizar discretización previa en variables con muchos valores únicos puede mejorar la interpretabilidad de los intervalos.
- Revisar manualmente las reglas con mayor soporte para garantizar que tengan sentido dentro del dominio de negocio.

### 9.12 Ejemplo de salida en formato JSON

Para facilitar la integración con otras herramientas, `Trees` puede exportar sus reglas en formato JSON. Un ejemplo simplificado es:

```
[
  {
    "bounds": {"x1": [5.0, Infinity]},
    "support": 40,
    "distribution": [0.2, 0.8]
  },
  {
    "bounds": {"x1": [-Infinity, 5.0], "x2": [-Infinity, 3.0]},
    "support": 70,
    "distribution": [0.9, 0.1]
  }
]
```

Este formato permite serializar los resultados y analizarlos en plataformas externas como JavaScript o R.

### 9.13 Discusión sobre interpretabilidad

El valor principal de `Trees` reside en su capacidad para producir reglas que los analistas humanos pueden comprender. A diferencia de modelos de caja negra, cada hiperrectángulo describe explícitamente un rango de valores. Las dimensiones omitidas implican ausencia de restricción, lo que transmite de manera natural la relevancia de cada variable.

Esta característica facilita la creación de reportes ejecutivos donde se enumeran las reglas más influyentes, acompañadas de métricas de soporte y precisión. Las reglas pueden ordenarse por cobertura, pureza o lift, proporcionando distintos ángulos de análisis según las necesidades del usuario.

### 9.14 Extensión matemática: volumen y probabilidad condicionada

Para cada regla $R$, podemos calcular su volumen teórico como:

$$
\text{vol}(R) = \prod_{j \in S} (b_j - a_j),
$$

donde $S$ es el subconjunto de características presentes en la regla. Este volumen permite estimar la probabilidad de que una observación caiga dentro de $R$ bajo una distribución uniforme. Combinado con la distribución de clases, se obtiene la probabilidad condicionada:

$$
P(y = k \mid x \in R) = \frac{\text{count}_k(R)}{\sum_{c} \text{count}_c(R)}.
$$

### 9.15 Relación con la impureza de Gini

La impureza en cada hoja se define como:

$$
G(R) = 1 - \sum_{k} p_k^2.
$$

`Trees` utiliza este valor para filtrar reglas con mezclas de clases demasiado altas. Un umbral típico es `G(R) < 0.5`, aunque el usuario puede ajustarlo para equilibrar pureza y cobertura.

### 9.16 Visualización de reglas

Las reglas derivadas pueden representarse gráficamente. Por ejemplo, en problemas bidimensionales se dibujan rectángulos en un plano cartesiano. `Trees` ofrece utilidades para exportar los límites y facilitar su visualización con bibliotecas como `matplotlib` o `plotly`. Estos gráficos ayudan a identificar solapamientos y vacíos en el espacio de características.

### 9.17 Limitaciones específicas del módulo

- En datasets con cientos de características, el número de intervalos por regla puede volverse inmanejable. Se recomienda aplicar técnicas de selección de variables antes de entrenar el bosque.
- Las reglas no capturan relaciones no rectangulares; por ejemplo, límites circulares o polinomiales requieren transformaciones previas de las variables.
- El rendimiento del recorrido depende de la estructura del árbol; árboles extremadamente desbalanceados pueden generar cargas desiguales en la pila.

### 9.18 Futuras extensiones

Se contemplan varias mejoras para próximas versiones de `Trees`:

1. **Compresión de reglas** mediante algoritmos de aprendizaje de bayas para reducir el número total de hiperrectángulos.
2. **Soporte para intervalos abiertos/cerrados** adaptables, permitiendo especificar condiciones del tipo `$x_j \leq t$` o `$x_j < t$` según se requiera.
3. **Incorporación de métricas de estabilidad**, que evalúan cómo cambian las reglas cuando se entrena el bosque con datos resampleados.
4. **Interfaz interactiva** para explorar reglas en un dashboard web.

### 9.19 Conclusión del análisis de `Trees`

El módulo `Trees` transforma la compleja arquitectura de los bosques de decisión en piezas manejables y transparentes. Gracias a su diseño basado en intervalos y filtros configurables, actúa como puente entre el aprendizaje estadístico y la interpretabilidad humana. Su comprensión detallada es esencial para sacar provecho del enfoque de clustering supervisado que InsideForest propone.

## 10. Detalle exhaustivo del módulo `Regions`

### 10.1 Motivación y definición formal
`Regions` se encarga de agrupar las reglas producidas por `Trees` en conglomerados coherentes.
El objetivo es consolidar hiperrectángulos similares y reducir la redundancia del modelo.
Formalmente, sea \(\mathcal{R} = \{R_1, R_2, \ldots, R_m\}\) el conjunto de reglas.
Cada regla \(R_i\) viene descrita por un vector de intervalos \((I_{i1}, I_{i2}, \ldots, I_{id})\) y una distribución de clases \(p_i\).
`Regions` busca una partición \(\mathcal{C} = \{C_1, C_2, \ldots, C_k\}\) tal que:
- Las reglas dentro de un mismo \(C_j\) maximicen la similitud geométrica y de distribución.
- La unión de los \(C_j\) cubra la mayor parte de \(\mathcal{R}\) sin solapamientos excesivos.
La función de pérdida global puede expresarse como:
\[
L = \sum_{j=1}^k \sum_{R_i \in C_j} \alpha \cdot d_V(R_i, \bar R_j) + \beta \cdot d_P(p_i, \bar p_j),
\]
 donde \(d_V\) mide distancia volumétrica, \(d_P\) distancia entre distribuciones y \(\bar R_j\), \(\bar p_j\) son centroides.
Los hiperparámetros \(\alpha\) y \(\beta\) equilibran geometría e información de clases.

### 10.2 Representación y estructura de datos
Cada regla se codifica como un vector numérico de dimensión \(2d\).
Para cada característica \(x_j\) se almacenan los límites \(a_{ij}\) y \(b_{ij}\).
Las dimensiones ausentes reciben valores \(-\infty\) y \(+\infty\) respectivamente.
`Regions` organiza estos vectores en una matriz \(M \in \mathbb{R}^{m \times 2d}\).
Esta matriz sirve como entrada a algoritmos de clustering estándar.
Además, se mantiene un diccionario paralelo con metadatos por regla:
soporte, pureza y referencias a índices originales.
El uso de `numpy.ndarray` permite operaciones vectoriales eficientes.
En implementaciones distribuídas, se utiliza `pyspark.mllib` y estructuras `RDD`.
El esquema general es:
```
R_i = [a_{i1}, b_{i1}, a_{i2}, b_{i2}, ..., a_{id}, b_{id}]
M = np.array([R_1, R_2, ..., R_m])
```

### 10.3 Conversión de reglas en vectores
Para convertir una regla en vector, se recorre cada característica del dominio original.
Si la regla impone una condición, se registra el intervalo correspondiente.
Si no existe condición, se asignan los límites globales \((a_j^{min}, b_j^{max})\).
Sea \(J_i\) el conjunto de variables presentes en \(R_i\).
El proceso formal es:
1. Inicializar \(v_i = [\ ]\).
2. Para cada \(j = 1,\ldots,d\):
   - Si \(j \in J_i\), añadir \((a_{ij}, b_{ij})\) a \(v_i\).
   - En caso contrario, añadir \((a_j^{min}, b_j^{max})\).
3. Devolver \(v_i\) como vector de \(2d\) componentes.
Este algoritmo tiene complejidad \(O(d)\) por regla.
En total, la conversión de \(m\) reglas es \(O(md)\).
La vectorización es requisito previo para aplicar métricas de distancia.

### 10.4 Métricas de distancia y similitud
La similitud geométrica entre reglas se evalúa con la intersección de intervalos.
Sea \(R_i\) y \(R_j\) dos reglas, el volumen de su intersección es:
\[
V_{ij} = \prod_{l=1}^d \max(0, \min(b_{il}, b_{jl}) - \max(a_{il}, a_{jl}))
\]
El volumen de la unión es:
\[
U_{ij} = \prod_{l=1}^d (\max(b_{il}, b_{jl}) - \min(a_{il}, a_{jl}))
\]
La **similitud volumétrica de Jaccard** se define como:
\[
J_{ij} = \frac{V_{ij}}{U_{ij}}
\]
Otra métrica es la distancia euclidiana entre vectores normalizados:
\[
D_{ij} = \left( \sum_{l=1}^{2d} (v_{il} - v_{jl})^2 \right)^{1/2}
\]
Para distribuciones de clase se usa la **divergencia de Jensen-Shannon**:
\[
JS(p_i, p_j) = \tfrac{1}{2} KL(p_i \| m) + \tfrac{1}{2} KL(p_j \| m)
\]
con \(m = \tfrac{1}{2}(p_i + p_j)\).
La distancia total combinada es una suma ponderada:
\[
D^{tot}_{ij} = \alpha D_{ij} + \beta JS(p_i, p_j)
\]

### 10.5 Clustering con DBSCAN
El primer algoritmo disponible en `Regions` es **DBSCAN**.
Este método identifica conglomerados basándose en densidad.
Parámetros principales:
- \(\varepsilon\): radio de vecindad.
- `min_samples`: puntos mínimos para formar un núcleo.
El algoritmo sobre los vectores se ejecuta en pasos:
1. Calcular la matriz de distancias \(D^{tot}\).
2. Para cada regla, contar vecinos dentro de \(\varepsilon\).
3. Clasificar reglas como núcleo, borde o ruido.
4. Expandir conglomerados a partir de núcleos.
La complejidad es \(O(m^2)\) si se calcula la matriz completa.
Optimizaciones con estructuras `KDTree` reducen el coste a \(O(m \log m)\).
La elección de \(\varepsilon\) se guía por el gráfico *k-distance*.
`Regions` automatiza parte de esta selección mediante heurísticas.

### 10.6 Clustering con KMeans
Como alternativa, `Regions` implementa **KMeans**.
Este método minimiza la suma de distancias cuadráticas a centroides.
La función objetivo es:
\[
\min_{C_1,\ldots,C_k} \sum_{j=1}^k \sum_{R_i \in C_j} \|v_i - \mu_j\|^2
\]
Las principales etapas son:
1. Inicialización de \(\mu_j\) con `k-means++` para mejorar convergencia.
2. Asignación de cada \(v_i\) al centro más cercano.
3. Recomputación de centroides como promedio de puntos asignados.
4. Repetición hasta convergencia o máximo de iteraciones.
La complejidad por iteración es \(O(mkd)\).
KMeans requiere especificar \(k\), por lo que `Regions` provee criterios de selección:
- Codo de varianza intra-cluster.
- Puntuación de Silhouette ajustada a la distancia combinada.
- Información previa sobre número deseado de segmentos.

### 10.7 Selección de parámetros y validación
La calidad del clustering depende fuertemente de \(\varepsilon\) o \(k\).
`Regions` expone funciones para búsqueda de grilla.
El proceso incluye:
1. Definir rangos de parámetros.
2. Ejecutar el clustering para cada combinación.
3. Evaluar métricas internas como Silhouette o Dunn.
4. Escoger la configuración con mejor equilibrio de pureza y cobertura.
Para validación externa se utiliza el índice de Rand ajustado frente a etiquetas reales.
Matemáticamente, el índice de Rand se calcula como:
\[
ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - [\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}] / \binom{n}{2}}{\tfrac{1}{2}[\sum_i \binom{a_i}{2}+\sum_j \binom{b_j}{2}] - [\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}] / \binom{n}{2}}
\]
La implementación usa vectores de etiquetas para computar las combinaciones \(\binom{\cdot}{2}\).

### 10.8 Pseudocódigo del pipeline de `Regions`
El flujo completo se puede resumir así:
```
entrada: reglas R_1,...,R_m
salida: regiones etiquetadas
1. V <- vectorizar_reglas(R)
2. D <- calcular_distancias(V, p)
3. C <- algoritmo_clustering(D)
4. para cada cluster c en C:
       rango <- fusionar_intervalos(c)
       etiqueta <- resumir_distribucion(c)
       guardar(rango, etiqueta)
```
Cada función se implementa cuidadosamente para conservar la trazabilidad a reglas originales.
La operación `fusionar_intervalos` calcula el mínimo y máximo por dimensión:
```
a_j = min_{R_i in c} a_{ij}
b_j = max_{R_i in c} b_{ij}
```
El resultado es un hiperrectángulo representativo de la región.

### 10.9 Ejemplo numérico detallado
Supongamos un problema con \(d=2\) y tres reglas:
- \(R_1: x_1 \in [0,1], x_2 \in [0,2]\) con distribución \([0.9,0.1]\).
- \(R_2: x_1 \in [0.1,1.1], x_2 \in [0,2.2]\) con distribución \([0.85,0.15]\).
- \(R_3: x_1 \in [3,4], x_2 \in [3,4]\) con distribución \([0.2,0.8]\).
Tras vectorización obtenemos:
```
V = [[0,1,0,2], [0.1,1.1,0,2.2], [3,4,3,4]]
```
La matriz de distancias combinadas produce dos grupos claramente separados.
Aplicando DBSCAN con \(\varepsilon = 0.5\) y `min_samples=2`:
- \(R_1\) y \(R_2\) forman el cluster \(C_1\).
- \(R_3\) se marca como ruido o cluster independiente \(C_2\).
El rango consolidado de \(C_1\) es:
\(
[a_1, b_1] = [0,1.1], \quad [a_2, b_2] = [0,2.2]
\)
La distribución promedio es \([0.875, 0.125]\).
Este ejemplo ilustra la fusión de reglas solapadas y la preservación de pureza.

### 10.10 Complejidad y análisis asintótico
Sea \(m\) el número de reglas y \(d\) el número de características.
La vectorización es \(O(md)\).
El cálculo de distancias completo es \(O(m^2 d)\).
DBSCAN añade un término \(O(m \log m)\) si se usa `KDTree`.
KMeans requiere \(O(mkdT)\) con \(T\) iteraciones.
La fusión de intervalos por cluster es \(O(md)\) en el peor caso.
En escenarios masivos, se recurre a muestreo o técnicas de particionado.
`Regions` ofrece hooks para procesamiento incremental.

### 10.11 Medidas de calidad: pureza y densidad
La **pureza** de una región \(C_j\) se define como:
\[
\pi(C_j) = \max_{k} \frac{1}{|C_j|} \sum_{R_i \in C_j} p_{ik}
\]
La **densidad** se aproxima como:
\[
\delta(C_j) = \frac{\sum_{R_i \in C_j} soportes_i}{Volumen(C_j)}
\]
El volumen del cluster se calcula mediante los intervalos fusionados.
Valores altos de \(\delta\) indican regiones compactas y bien definidas.
`Regions` permite filtrar clusters con \(\pi\) o \(\delta\) por debajo de un umbral.

### 10.12 Fusión y poda de regiones
Tras el clustering, puede haber clusters muy cercanos que convenga fusionar.
Para dos regiones \(C_a\) y \(C_b\) se evalúa la similitud de Jaccard entre sus rangos.
Si \(J(C_a, C_b) > \tau\) se combinan en un solo cluster.
El proceso iterativo es:
1. Ordenar las regiones por soporte descendente.
2. Comparar cada par consecutivo.
3. Fusionar si cumplen el umbral y recalcular métricas.
La poda descarta regiones con soporte inferior a \(s_{min}\).
Esta etapa previene la proliferación de segmentos irrelevantes.

### 10.13 Interacción con `Trees` y `Labels`
`Regions` recibe como entrada las reglas generadas por `Trees`.
Cada región devuelta se envía luego al módulo `Labels`.
`Labels` traduce los intervalos en descripciones lingüísticas.
La sinergia se resume así:
- `Trees`: produce reglas detalladas.
- `Regions`: agrupa y depura reglas.
- `Labels`: comunica los resultados a usuarios finales.
Matemáticamente, si \(f\) es el mapa de `Trees` y \(g\) el de `Regions`, la cadena es \(g(f(D))\).
La preservación de información se asegura manteniendo identificadores únicos de reglas.

### 10.14 Visualización de espacios de regiones
Las regiones pueden proyectarse en planos bidimensionales.
Se utilizan mapas de calor donde el color indica pureza o densidad.
Otra técnica es el **diagrama de paralelas**, que muestra intervalos por dimensión.
Para un cluster con intervalos \([a_j, b_j]\) se dibuja un segmento vertical.
Las herramientas recomendadas incluyen `matplotlib`, `seaborn` y `plotly`.
En grandes dimensiones, se recurre a proyecciones PCA antes de dibujar.
`Regions` provee funciones auxiliares para exportar datos compatibles con estas librerías.

### 10.15 Extensiones matemáticas: espacios métricos
Consideremos el espacio de reglas \((\mathcal{R}, D^{tot})\).
Este espacio es métricamente completo si los intervalos están acotados.
Por el teorema de Hopf-Rinow, todo par de reglas tiene una geodésica.
Las geodésicas se interpretan como interpolaciones lineales en el espacio vectorial.
Se puede definir una medida de curvatura utilizando teoría de espacios CAT(0).
Aunque estas nociones exceden la implementación actual, guían extensiones futuras.

### 10.16 Consideraciones de paralelización
La etapa más costosa es el cálculo de distancias.
`Regions` paraleliza este paso usando `joblib` o `pyspark`.
El enfoque divide la matriz \(M\) en bloques y distribuye su procesamiento.
Para bloques \(B_{pq}\) se calcula la distancia entre subconjuntos de reglas.
La recombinación se hace concatenando resultados parciales.
Este esquema reduce el tiempo de \(O(m^2)\) a \(O(m^2 / P)\) con \(P\) procesos.
La consistencia se mantiene fijando semillas aleatorias en algoritmos estocásticos.

### 10.17 Futuras líneas de trabajo
Se planean varias mejoras:
1. Uso de **Gaussian Mixture Models** para capturar fronteras suaves.
2. Implementación de **clustering jerárquico aglomerativo**.
3. Incorporación de **medidas de estabilidad temporal** para datos evolutivos.
4. Interfaz para **semi-supervised clustering** con restricciones must-link y cannot-link.
5. Métodos de **incremental learning** que actualicen regiones con nuevos datos.
Cada propuesta requiere estudiar convergencia y complejidad asociadas.

### 10.18 Validación empírica con bootstrap
Para evaluar la estabilidad de las regiones se emplea bootstrap.
El procedimiento genera \(B\) re-muestreos del conjunto de reglas.
Para cada re-muestreo se recalcula el clustering y se obtiene una partición \(\mathcal{C}^{(b)}\).
La varianza de soporte para cada región se estima como:
\[
Var(soporte_j) = \frac{1}{B-1} \sum_{b=1}^B (soporte_j^{(b)} - \bar s_j)^2
\]
con \(\bar s_j\) la media muestral.
Altas varianzas indican regiones inestables susceptibles de revisión.
El bootstrap también permite construir intervalos de confianza para la pureza \(\pi(C_j)\).

### 10.19 Relación con clustering de conjuntos difusos
`Regions` puede extenderse a un marco difuso donde cada regla pertenece parcialmente a varios clusters.
Sea \(\mu_{ij} \in [0,1]\) el grado de pertenencia de \(R_i\) al cluster \(C_j\).
La actualización de centroides se modifica a:
\[
\mu_j = \frac{\sum_i \mu_{ij}^m v_i}{\sum_i \mu_{ij}^m}
\]
donde \(m>1\) controla la difusidad.
Este esquema reduce saltos abruptos en regiones fronterizas.
La versión difusa permite interpretar regiones como superposiciones suaves en lugar de particiones rígidas.

### 10.20 Gestión de memoria y almacenamiento
El número de reglas puede alcanzar decenas de miles.
Para evitar saturación, `Regions` implementa almacenamiento esparso.
Los intervalos \((a_{ij}, b_{ij})\) se guardan en matrices dispersas tipo CSR.
El consumo de memoria pasa de \(O(md)\) a \(O(z)\) con \(z\) el número de intervalos explícitos.
Se proporcionan funciones de serialización a formato `parquet` para integrarse con `pandas` y `spark`.
Durante el clustering, los bloques de la matriz se cargan bajo demanda para reducir el uso de RAM.

### 10.21 Interpretación estadística de los centroides
Los centroides de regiones pueden verse como estimadores de medias truncadas.
Si \(C_j\) contiene intervalos \([a_{ij}, b_{ij}]\), el centro \(\mu_j\) se ubica en el punto medio de cada intervalo.
Cuando las distribuciones originales de características son uniformes, \(\mu_j\) es un estimador insesgado.
En distribuciones sesgadas, se corrige aplicando pesos proporcionales a densidades observadas.
Esta interpretación conecta `Regions` con técnicas de muestreo estratificado.

### 10.22 Conclusión del análisis de `Regions`
El módulo `Regions` transforma una colección extensa de reglas en segmentos manejables.
Mediante técnicas de clustering y métricas rigurosas, ofrece una visión condensada del espacio de características.
Su diseño matemático permite extensiones hacia análisis topológicos y aprendizaje semi-supervisado.
Comprender `Regions` es esencial para aprovechar plenamente el enfoque de clustering supervisado de InsideForest.

### 10.23 Referencias cruzadas y reproducibilidad
`Regions` registra metadatos de cada cluster para permitir auditorías posteriores.
Cada región guarda la lista de reglas originales que la componen y los parámetros de clustering utilizados.
Esta trazabilidad facilita la reproducción de experimentos y la comparación entre ejecuciones.
Se recomienda almacenar semillas aleatorias y versiones de librerías para garantizar resultados consistentes.
El repositorio incluye scripts de ejemplo que ilustran cómo fijar estas semillas y registrar configuraciones.
La reproducibilidad es un componente clave para la transparencia científica del enfoque.
La documentación generada por `Regions` puede exportarse en formatos CSV y JSON.
Cada exportación incluye un identificador de hash que resume las opciones de configuración.
Esto permite compartir resultados y verificar su integridad mediante controles de suma.
La exportación se acompaña de metadatos para auditoría completa.

## 11. Apéndice A: Derivaciones matemáticas adicionales

Este apéndice ofrece derivaciones complementarias que sustentan los conceptos presentados anteriormente.

### 11.1 Derivación del índice de Gini

Partimos de la probabilidad estimada para la clase $k$ en una región $R$:

$$
\hat{p}_k = \frac{n_k}{n_R}.
$$

La impureza de Gini se deriva de la probabilidad de clasificación errónea al asignar aleatoriamente una observación a una clase según $\hat{p}_k$:

$$
G(R) = \sum_{k \neq l} \hat{p}_k \hat{p}_l = 1 - \sum_{k} \hat{p}_k^2.
$$

### 11.2 Volumen de la unión de hiperrectángulos

Cuando `Regions` evalúa solapamientos, requiere calcular el volumen de la unión de dos reglas $R_1$ y $R_2$. La fórmula se basa en principios de inclusión-exclusión:

$$
\text{vol}(R_1 \cup R_2) = \text{vol}(R_1) + \text{vol}(R_2) - \text{vol}(R_1 \cap R_2).
$$

El cálculo de la intersección se realiza coordinada a coordinada, tomando el máximo de los límites inferiores y el mínimo de los superiores:

$$
[a_j, b_j]_{R_1 \cap R_2} = [\max(a_{1j}, a_{2j}), \min(b_{1j}, b_{2j})].
$$

Si algún límite inferior excede al superior, la intersección es vacía y el volumen cero.

### 11.3 Propiedades del algoritmo DBSCAN aplicado a reglas

DBSCAN opera sobre la representación vectorial de las reglas. Las propiedades clave son:

- **Densidad alcanzable**: una regla pertenece a un cluster si puede conectarse con otra mediante una cadena de vecinos, cada uno dentro del radio `\varepsilon`.
- **Puntos frontera**: reglas con menos de `minPts` vecinos pero dentro del radio de un punto denso se asignan al cluster de dicho punto.
- **Ruido**: reglas que no cumplen los criterios anteriores se consideran outliers y pueden descartarse en etapas posteriores.

En el contexto de InsideForest, `\varepsilon` suele calibrarse en función del promedio de volúmenes de las reglas para equilibrar sensibilidad y robustez.

### 11.4 Cálculo del Lift

Una medida utilizada en marketing es el **lift**, que compara la proporción de la clase objetivo dentro de un segmento con la proporción global:

$$
\text{lift}(R) = \frac{\hat{p}_1(R)}{\hat{p}_1(D)}.
$$

Valores superiores a 1 indican que la región `R` contiene una mayor concentración de la clase positiva que el promedio del conjunto de datos.

### 11.5 Ejemplo numérico completo

Consideremos un dataset binario con 1000 observaciones donde la clase positiva representa el 10%. Tras entrenar InsideForest, se obtiene una regla con `support = 80` y `distribution = [0.2, 0.8]`. Entonces:

- `\hat{p}_1(R) = 0.8`
- `\text{lift}(R) = 0.8 / 0.1 = 8`

Esto sugiere que la región es ocho veces más propensa a contener la clase positiva que el conjunto general, convirtiéndola en un objetivo prioritario para campañas específicas.

### 11.6 Consideraciones de implementación

La implementación real del módulo incluye detalles de ingeniería como:

- Uso de `numpy` para operaciones vectorizadas que aceleran el procesamiento de grandes volúmenes.
- Empleo de `joblib` para paralelizar el análisis de múltiples árboles en máquinas multicore.
- Manejo cuidadoso de tipos de datos para minimizar el consumo de memoria, especialmente al trabajar con floats de 32 bits cuando la precisión adicional no es necesaria.

### 11.7 Referencias bibliográficas

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
- Ester, M., et al. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*. Proceedings of KDD.

### 11.8 Agradecimientos

El desarrollo de InsideForest ha sido posible gracias a la comunidad de código abierto y a los contribuidores que han probado la librería en contextos reales. Sus sugerencias han guiado la optimización del módulo `Trees` y la expansión de funcionalidades.

### 11.9 Licencia de uso de datos

Al emplear InsideForest con datos sensibles, se recomienda seguir las políticas de privacidad y normativas vigentes (como GDPR en Europa o LFPDPPP en México). La interpretabilidad de las reglas facilita la auditoría de modelos y la detección de sesgos, pero es responsabilidad del usuario asegurar el tratamiento ético de la información.

### 11.10 Conclusión del documento

Este documento, estructurado como un paper técnico, ha recorrido en detalle los fundamentos matemáticos, la arquitectura y el funcionamiento interno del repositorio InsideForest. En particular, se profundizó en el módulo `Trees`, mostrando cómo convierte árboles de decisión en reglas interpretables y cómo se integra en un pipeline mayor de clustering supervisado. Esperamos que esta descripción exhaustiva sirva como guía para investigadores y profesionales que deseen comprender, extender o aplicar la librería en problemas reales.


## 12. Glosario de términos clave

A continuación se presenta un glosario que resume los conceptos técnicos utilizados a lo largo de este documento. Cada término incluye una definición detallada y, cuando aplica, una referencia a las secciones donde se emplea.

### 12.1 Árbol de decisión
Estructura jerárquica donde cada nodo interno representa una condición sobre una característica y cada hoja corresponde a una predicción o distribución de clases. Véase la Sección 2.1 para fundamentos teóricos y la Sección 9 para detalles de implementación.

### 12.2 Bosque aleatorio
Conjunto de múltiples árboles de decisión entrenados sobre subconjuntos aleatorios de datos y características. La predicción final se obtiene promediando las salidas individuales. Explicado en la Sección 2.1 y utilizado como base en todo el pipeline.

### 12.3 Hiperrectángulo
Representación geométrica de una regla extraída de un árbol; consiste en intervalos acotados para un subconjunto de características. Los hiperrectángulos son el producto principal del módulo `Trees` y se analizan en la Sección 9.3.

### 12.4 Intervalo
Par de valores `(min, max)` que delimita la región válida para una característica dentro de un hiperrectángulo. La actualización de intervalos se discute en la Sección 9.2.

### 12.5 Soporte
Número de observaciones que satisfacen una regla o caen dentro de un hiperrectángulo. Es una métrica de relevancia que interviene en los filtros descritos en la Sección 9.4.

### 12.6 Pureza
Medida de homogeneidad de clases dentro de una regla. Se calcula a partir de la distribución de clases y se vincula con la impureza de Gini de la Sección 11.1.

### 12.7 DBSCAN
Algoritmo de clustering basado en densidad que identifica grupos como regiones densas separadas por zonas de baja densidad. En InsideForest se aplica sobre la representación vectorial de reglas (Sección 11.3).

### 12.8 Lift
Relación entre la tasa de ocurrencia de un evento dentro de una región y la tasa global. Es útil para evaluar la efectividad de los segmentos generados (Sección 11.4).

### 12.9 Pila (stack)
Estructura de datos LIFO utilizada por `Trees` para realizar recorridos en profundidad sin recursión. Esta técnica se detalla en la Sección 9.5.

### 12.10 Broadcast
Mecanismo de PySpark que distribuye copias de objetos de solo lectura a cada nodo del clúster. Es esencial para escalar el módulo `Trees` en contextos distribuidos (Sección 9.8).

### 12.11 Volumen
Medida del tamaño de un hiperrectángulo calculada como el producto de longitudes de sus intervalos. El volumen se utiliza para comparar y fusionar reglas (Secciones 2.3 y 9.14).

### 12.12 Clustering supervisado
Proceso de agrupación que utiliza información de etiquetas para guiar la formación de clusters. InsideForest adopta esta filosofía para descubrir segmentos interpretables (Sección 1 y a lo largo de todo el documento).

### 12.13 Regla redundante
Regla que aporta poca o ninguna información adicional debido a que sus límites y distribución son similares a los de otra regla. La detección de redundancia se explica en la Sección 9.4.

### 12.14 Distribución de clases
Vector que indica la proporción de cada clase dentro de un conjunto de observaciones. Se almacena en el atributo `value` de cada hoja y se normaliza en el módulo `Trees` (Sección 9.5).

### 12.15 Intervalo abierto/cerrado
Convenciones que especifican si los límites de un intervalo están incluidos (`[a, b]`) o excluidos (`(a, b)`). Las futuras extensiones del módulo planean permitir intervalos configurables (Sección 9.18).

Este glosario busca servir como referencia rápida y consolidar el vocabulario técnico asociado a InsideForest.

