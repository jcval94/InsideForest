from openai import OpenAI
import pandas as pd
import copy
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import re
import logging

logger = logging.getLogger(__name__)

def primer_punto_inflexion_decreciente(data, bins=10, window_length=5, polyorder=2):
    """
    Encuentra el primer punto de inflexión decreciente en un histograma.

    Parámetros:
    - data: array-like, los datos para construir el histograma.
    - bins: int o sequence, número de bins o los bordes de los bins.
    - window_length: int, longitud de la ventana para el filtro Savitzky-Golay.
    - polyorder: int, orden del polinomio para el filtro Savitzky-Golay.

    Retorna:
    - punto_inflexion: valor del bin donde ocurre el primer punto de inflexión decreciente.
    """

    if len(data) == 0:
        logger.error("La lista de datos está vacía")
        return None

    try:
        # Calcular el histograma
        counts, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    except Exception as exc:
        logger.exception("Error al calcular el histograma: %s", exc)
        return None

    # Suavizar el histograma para reducir ruido
    # Asegurarse de que window_length es impar y menor que el tamaño de counts
    if window_length >= len(counts):
        window_length = len(counts) - 1 if len(counts) % 2 == 0 else len(counts)
    if window_length % 2 == 0:
        window_length += 1
    if window_length < polyorder + 2:
        window_length = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3

    try:
        counts_smooth = savgol_filter(counts, window_length=window_length, polyorder=polyorder)
    except Exception as exc:
        logger.exception("Error aplicando savgol_filter: %s", exc)
        return None

    # Calcular la segunda derivada
    second_derivative = np.gradient(np.gradient(counts_smooth))

    # Encontrar los puntos de inflexión donde la segunda derivada cambia de signo
    # De positivo a negativo indica un cambio de concavidad hacia abajo (punto de inflexión decreciente)
    sign_changes = np.diff(np.sign(second_derivative))
    # Un cambio de +1 a -1 en la segunda derivada
    inflection_indices = np.where(sign_changes < 0)[0] + 1  # +1 para corregir el desplazamiento de diff

    if len(inflection_indices) == 0:
        return None  # No se encontró un punto de inflexión decreciente

    # Seleccionar el primer punto de inflexión decreciente
    primer_inflexion = bin_centers[inflection_indices[0]]

    return primer_inflexion

def replace_with_dict(df, columns, var_rename):
    """
    Reemplaza valores en columnas especificadas de un DataFrame usando un diccionario.
    Reemplaza coincidencias exactas y subcadenas que contienen las claves del diccionario.

    Parámetros
    ----------
    df : pd.DataFrame
        El DataFrame original.
    columns : list of str
        Lista de nombres de columnas donde se aplicarán los reemplazos.
    var_rename : dict
        Diccionario donde las claves son los valores a reemplazar y los valores son los nuevos valores.

    Retorna
    -------
    df_replaced : pd.DataFrame
        DataFrame con los reemplazos realizados en las columnas especificadas.
    replace_info : dict
        Información necesaria para revertir los reemplazos.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un DataFrame de pandas")

    df_replaced = df.copy()
    replace_info = {}
    
    # Ordenar las claves por longitud descendente para evitar conflictos en subcadenas
    sorted_keys = sorted(var_rename.keys(), key=len, reverse=True)
    escaped_keys = [re.escape(k) for k in sorted_keys]
    pattern = re.compile('|'.join(escaped_keys))
    
    for col in columns:
        if col not in df_replaced.columns:
            logger.warning(
                f"Advertencia: La columna '{col}' no se encontró en el DataFrame."
            )
            continue
        
        # Almacenar información de reemplazo por columna
        replace_info[col] = {
            'var_rename': var_rename.copy()
        }
        
        # Definir la función de reemplazo
        def repl(match):
            return var_rename[match.group(0)]
        
        try:
            df_replaced[col] = df_replaced[col].astype(str).str.replace(pattern, repl, regex=True)
        except Exception as exc:
            logger.exception("Error al reemplazar valores en la columna %s: %s", col, exc)
            continue
    
    return df_replaced, replace_info




def get_descripciones_valiosas(
    df_datos_descript,
    df_datos_clusterizados,
    TARGETS,
    var_rename,
    inflex_pond_sup=0.4,
    inflex_pond_inf=0.5
):
    """
    Versión modificada donde NO se filtra el resultado final, 
    sino que se agrega una columna 'buenos' con valor 1 u 0 
    según los mismos criterios de inflexión usados en la versión anterior.
    """

    # --- 1) Ordenamos df_datos_descript ---
    df_datos_descript = df_datos_descript.sort_values('cluster_ponderador', ascending=False)

    # --- 2) Merge para tener descripciones en df clusterizados ---
    df_datos_clusterizados_desc = df_datos_clusterizados.merge(
        df_datos_descript, on='cluster', how='left'
    )
    
    # --- 3) Generamos la matriz (unstack) para conteo de TARGETS[0] vs cluster ---
    stacked_data = df_datos_clusterizados_desc.groupby(
        [TARGETS[0], 'cluster']
    ).size().unstack(fill_value=0)
    
    # Proporción real de la clase 1 en todo el dataset
    proporcion_real = df_datos_clusterizados_desc[TARGETS[0]].value_counts(normalize=True).loc[1]
    
    # Conteo total por cada cluster
    stacked_data_total = stacked_data.sum(axis=0)
    
    # Proporción de la clase 1 en cada cluster respecto a su total
    # (es decir, #1 en cluster / total cluster)
    proprcin_ = (stacked_data / stacked_data.sum(axis=0)).loc[1]
    
    # --- 4) Creamos un dataframe con la razón y el soporte ---
    #     Los índices son los clusters. En la col[0] = (proporcion cluster / proporcion_real)
    #     En la col[1] = total de ese cluster
    los_custers = pd.concat([proprcin_ / proporcion_real, stacked_data_total], axis=1)
    # Ordenamos por la primera columna (índice 0), de mayor a menor
    los_custers = los_custers.sort_values(by=0, ascending=False)
    
    # --- 5) Hacemos una copia de los_custers para usarla completa,
    #     y luego generamos la versión "valiosos" (con [1] > 1).
    los_custers_original = los_custers.copy()
    los_custers_valiosos = los_custers_original[los_custers_original[1] > 1].copy()

    # --- 6) Escalamos las columnas numéricas en "los_custers_valiosos" ---
    numeric_cols = los_custers_valiosos.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    los_custers_valiosos[numeric_cols] = scaler.fit_transform(los_custers_valiosos[numeric_cols])

    # --- 7) Hacemos una copia del DF ya escalado para calcular 'importancia' y puntos de inflexión ---
    los_custers_valiosos_original = copy.deepcopy(los_custers_valiosos)

    # Sumamos todas las columnas como "importancia" (en este caso col 0 y col 1, ya escaladas)
    los_custers_valiosos_original['importancia'] = los_custers_valiosos.sum(axis=1)
    # (No se usa el sort_values("importancia") para filtrar nada, pero lo dejamos si deseas inspeccionarlo)
    # los_custers_valiosos_original = los_custers_valiosos_original.sort_values('importancia', ascending=False)

    # --- 8) Calculamos los puntos de inflexión sobre las columnas 0 y 1 (ya escaladas) ---
    #     Aquí se asume que la función "primer_punto_inflexion_decreciente" está definida fuera.
    punto = primer_punto_inflexion_decreciente(
        los_custers_valiosos_original[0], bins=20, window_length=5, polyorder=2
    )
    punto_1 = primer_punto_inflexion_decreciente(
        los_custers_valiosos_original[1], bins=20, window_length=5, polyorder=2
    )

    cond_buenos = (
        (los_custers_original[1] > 1) & (  # (1) total > 1
            (los_custers_original[0] > (punto * inflex_pond_sup)) |  # (2a)
            (los_custers_original[1] > punto_1)                    |  # (2b)
            (los_custers_original[0] < inflex_pond_inf)               # (2c)
        )
    )

    # Creamos la columna "buenos" en los_custers_original
    los_custers_original['buenos'] = np.where(cond_buenos, 1, 0)

    # --- 10) Armamos ahora el dataframe final con TODAS las filas y la nueva columna ---
    # 1) Hacemos copia de df_datos_descript para no tocar el original
    df_datos_descript_valiosas = df_datos_descript.copy()

    # 2) Reemplazamos (si aplica) los textos de "cluster_descripcion" según var_rename
    df_datos_descript_valiosas, _ = replace_with_dict(
        df_datos_descript_valiosas, ['cluster_descripcion'], var_rename
    )

    # 3) Mergeamos la probabilidad (proprcin_) => col 0 = (prop. cluster / proporción_global)
    #    NOTA: en la salida final lo renombraremos a "Soporte" o el nombre que gustes
    df_datos_descript_valiosas = df_datos_descript_valiosas.merge(
        proprcin_.reset_index(), on='cluster', how='left'
    )

    # 4) Mergeamos los_custers_original para obtener col[0], col[1] y la nueva col 'buenos'
    df_datos_descript_valiosas = df_datos_descript_valiosas.merge(
        los_custers_original.reset_index(), on='cluster', how='left'
    )

    # 5) Renombramos las columnas que vienen duplicadas en el merge
    df_datos_descript_valiosas = df_datos_descript_valiosas.rename(
        columns={
            '1_x': 'Probabilidad',      # Proporción de la clase 1 en ese cluster
            '1_y': 'N_probabilidad',    # Conteo total del cluster
            0:     'Soporte',           # Ratio (prop.cluster / prop.global)
        }
    )

    # 6) Retornamos el DF final (ya con 'buenos') y la tabla stacked_data
    return df_datos_descript_valiosas.drop(columns=['cluster_ponderador']), stacked_data


def generate_descriptions(condition_list, language='en', OPENAI_API_KEY=None, default_params=None):

    client = OpenAI(api_key=OPENAI_API_KEY)

    if default_params is None:
        def get_default_params():
            return {
                'model': 'gpt-4-turbo',
                'temperature': 0.5,
                'max_tokens': 1500,
                'n': 1,
                'stop': None,
            }
        default_params = get_default_params()

    # Crear un único mensaje con todas las condiciones
    conditions_text = "\n".join([f"{i+1}. {condition}" for i, condition in enumerate(condition_list)])

    # Prompt mejorado para descripciones simples y comprensibles
    system_prompt = "You are an assistant that helps to describe dataset groups in very simple terms."
    user_prompt = (
        f"Generate a very simple description for each of the following conditions. "
        f"Use everyday language. Avoid specific numbers and ranges; instead, "
        f"use general groups like 'elderly people', 'classic cars', etc."
        f"Make each description visually friendly highlight what makes that condition unique and using emojis. Structure: 'EMOJI': 'RESPONSE'"
        f"Only respond with the descriptions in {language}. Conditions:\n\n{conditions_text}"
    )

    mensajes = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        respuesta = client.chat.completions.create(
            messages=mensajes,
            **default_params
        )
    except Exception as exc:
        logger.exception("Error al llamar a la API de OpenAI: %s", exc)
        return {'respuestas': []}

    # Dividir la respuesta en una lista de descripciones por línea
    descriptions = respuesta.choices[0].message.content.strip().split("\n")
    descriptions = [desc.strip() for desc in descriptions if desc.strip()]

    # Return a dictionary with the responses
    result = {'respuestas': descriptions}
    return result

def _categorize_conditions(condition_list, df, n_groups=2, handle_bools=False):
    """Función base para categorizar condiciones.

    Si ``handle_bools`` es ``True`` también se admiten comparaciones
    explícitas con columnas booleanas. En tal caso, los valores ``True`` se
    mapean a ``ALTO`` y ``False`` a ``BAJO``.

    Parameters
    ----------
    condition_list : list[str]
        Lista de strings con las condiciones a procesar.
    df : pd.DataFrame
        DataFrame de referencia para el cálculo de cuantiles.
    n_groups : int
        Número de grupos para la categorización de variables numéricas.
    handle_bools : bool, default False
        Si es ``True`` procesa comparaciones de igualdad contra ``True`` o
        ``False`` para columnas booleanas.

    Returns
    -------
    dict
        Diccionario ``{"respuestas": [str, ...]}`` con las descripciones
        generadas o ``{"error": <mensaje>}`` si los parámetros son inválidos.
    """
    # --- Validación de Entradas ---
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {'error': 'Se requiere un DataFrame de pandas válido y no vacío.'}
    if not isinstance(n_groups, int) or n_groups < 2:
        return {'error': 'n_groups debe ser un entero igual o mayor a 2.'}

    # --- Definición de Etiquetas de Categoría ---
    if n_groups <= 5:
        label_map = {
            2: ['BAJO', 'ALTO'],
            3: ['BAJO', 'MEDIO', 'ALTO'],
            4: ['MUY BAJO', 'BAJO', 'ALTO', 'MUY ALTO'],
            5: ['MUY BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY ALTO']
        }
        labels = label_map.get(n_groups)
    else:
        labels = [f'NIVEL_{i+1}' for i in range(n_groups)]

    # --- Cálculo de Umbrales (Quantiles) ---
    bool_cols = []
    if handle_bools:
        bool_cols = df.select_dtypes(include="bool").columns.tolist()

    thresholds = {}
    quantile_points = [i / n_groups for i in range(1, n_groups)]

    for column in df.columns:
        if handle_bools and column in bool_cols:
            continue
        thresholds[column] = df[column].quantile(quantile_points).tolist()

    # --- Procesamiento de Condiciones ---
    descriptions = []
    pattern_num = r'(\d+\.?\d*)\s*<=\s*(\w+)\s*<=\s*(\d+\.?\d*)'
    pattern_bool = r'(\w+)\s*==\s*(True|False)'

    for condition in condition_list:
        tokens = [t.strip() for t in re.split(r'\band\b', condition)]
        parts = []
        for token in tokens:
            m_num = re.match(pattern_num, token)
            if m_num:
                min_v, feat, max_v = m_num.groups()
                avg_value = (float(min_v) + float(max_v)) / 2
                if feat in thresholds:
                    cuts = thresholds[feat]
                    category = labels[np.searchsorted(cuts, avg_value)]
                elif handle_bools and feat in bool_cols:
                    category = 'ALTO' if avg_value >= 0.5 else 'BAJO'
                else:
                    category = 'N/A'
                parts.append(f"{feat} es {category}")
                continue

            if handle_bools:
                m_bool = re.match(pattern_bool, token)
                if m_bool:
                    feat, val = m_bool.groups()
                    if feat in bool_cols:
                        category = 'ALTO' if val == 'True' else 'BAJO'
                    else:
                        category = 'N/A'
                    parts.append(f"{feat} es {category}")

        descriptions.append(', '.join(parts) + '.')

    # --- Retorno de Resultados ---
    return {'respuestas': descriptions}


def categorize_conditions(condition_list, df, n_groups=2):
    """Generaliza condiciones numéricas en descripciones de texto."""
    return _categorize_conditions(condition_list, df, n_groups=n_groups, handle_bools=False)


def categorize_conditions_generalized(condition_list, df, n_groups=2):
    """Generaliza condiciones con soporte para columnas booleanas."""

    return _categorize_conditions(
        condition_list,
        df,
        n_groups=n_groups,
        handle_bools=True,
    )


def build_conditions_table(
    condition_list,
    df,
    efectividades,
    ponderadores=None,
    n_groups=3,
    fill_value="N/A",
):
    """Construye una tabla con descripciones categóricas y métricas.

    Parameters
    ----------
    condition_list : list[str]
        Lista de condiciones en formato ``"min <= VAR <= max …"``.
    df : pd.DataFrame
        DataFrame de referencia usado para calcular cuantiles.
    efectividades : list[float] | pd.Series
        Métrica de efectividad por cada condición.
    ponderadores : list[float] | pd.Series | None, default None
        Métrica opcional (soporte, frecuencia, etc.) por condición.
    n_groups : int, default 3
        Número de grupos para ``categorize_conditions``.
    fill_value : str, default "N/A"
        Valor para variables ausentes en la descripción de una condición.

    Returns
    -------
    pd.DataFrame
        Tabla resultante con columnas ``Grupo``, ``Efectividad``, ``Ponderador``
        y las variables extraídas en orden alfabético.
    """

    cat_results = categorize_conditions(condition_list, df, n_groups=n_groups)
    if "error" in cat_results:
        raise ValueError(cat_results["error"])
    descriptions = cat_results["respuestas"]

    var_pattern = r"(\w+)\s+es\s+([^,\.]+)"
    all_vars = set()
    for desc in descriptions:
        all_vars.update(re.findall(var_pattern, desc))
    variables = sorted({var for var, _ in all_vars})

    rows = []
    n = len(descriptions)
    if len(efectividades) != n:
        raise ValueError("`efectividades` debe tener la misma longitud que `condition_list`")
    if ponderadores is None:
        ponderadores = [np.nan] * n
    elif len(ponderadores) != n:
        raise ValueError("`ponderadores` debe tener la misma longitud que `condition_list`")

    for idx, desc in enumerate(descriptions):
        row = {
            "Grupo": idx + 1,
            "Efectividad": efectividades[idx],
            "Ponderador": ponderadores[idx],
        }
        row.update({var: fill_value for var in variables})

        for var, level in re.findall(var_pattern, desc):
            row[var] = level.strip()

        rows.append(row)

    final_cols = ["Grupo", "Efectividad", "Ponderador"] + variables
    return pd.DataFrame(rows, columns=final_cols)
