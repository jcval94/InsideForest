from openai import OpenAI
import pandas as pd
import copy
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import re

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

    # Calcular el histograma
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Suavizar el histograma para reducir ruido
    # Asegurarse de que window_length es impar y menor que el tamaño de counts
    if window_length >= len(counts):
        window_length = len(counts) - 1 if len(counts) % 2 == 0 else len(counts)
    if window_length % 2 == 0:
        window_length += 1
    if window_length < polyorder + 2:
        window_length = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3

    counts_smooth = savgol_filter(counts, window_length=window_length, polyorder=polyorder)

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
    df_replaced = df.copy()
    replace_info = {}
    
    # Ordenar las claves por longitud descendente para evitar conflictos en subcadenas
    sorted_keys = sorted(var_rename.keys(), key=len, reverse=True)
    escaped_keys = [re.escape(k) for k in sorted_keys]
    pattern = re.compile('|'.join(escaped_keys))
    
    for col in columns:
        if col not in df_replaced.columns:
            print(f"Advertencia: La columna '{col}' no se encontró en el DataFrame.")
            continue
        
        # Almacenar información de reemplazo por columna
        replace_info[col] = {
            'var_rename': var_rename.copy()
        }
        
        # Definir la función de reemplazo
        def repl(match):
            return var_rename[match.group(0)]
        
        # Aplicar el reemplazo usando expresiones regulares
        df_replaced[col] = df_replaced[col].astype(str).str.replace(pattern, repl, regex=True)
    
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

    # Crear una solicitud de finalización de chat con todos los mensajes
    respuesta = client.chat.completions.create(
        messages=mensajes,
        **default_params
    )

    # Dividir la respuesta en una lista de descripciones por línea
    descriptions = respuesta.choices[0].message.content.strip().split("\n")
    descriptions = [desc.strip() for desc in descriptions if desc.strip()]

    # Return a dictionary with the responses
    result = {'respuestas': descriptions}
    return result

def categorize_conditions(condition_list, df, n_groups=2):
    """
    Generaliza una lista de condiciones en descripciones de texto, categorizando
    los valores de las características en 'n_groups' niveles.

    Args:
        condition_list (list): Una lista de strings, donde cada string representa
                               una condición con rangos de variables.
                               Ej: ['0.0 <= Var1 <= 10.0 and 5.0 <= Var2 <= 15.0']
        df (pd.DataFrame): El DataFrame que contiene los datos de referencia para
                           calcular los umbrales de los cuantiles.
        n_groups (int): El número de grupos en los que se dividirán los datos.
                        Debe ser 2 o mayor.

    Returns:
        dict: Un diccionario que contiene una clave 'respuestas' con una lista de
              las descripciones generadas. O un diccionario con un error si los
              parámetros son inválidos.
    """
    # --- Validación de Entradas ---
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {'error': 'Se requiere un DataFrame de pandas válido y no vacío.'}
    if not isinstance(n_groups, int) or n_groups < 2:
        return {'error': 'n_groups debe ser un entero igual o mayor a 2.'}

    # --- Definición de Etiquetas de Categoría ---
    labels = []
    if n_groups <= 5:
        # Mapeo predefinido para 2 a 5 grupos
        label_map = {
            2: ['BAJO', 'ALTO'],
            3: ['BAJO', 'MEDIO', 'ALTO'],
            4: ['MUY BAJO', 'BAJO', 'ALTO', 'MUY ALTO'],
            5: ['MUY BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY ALTO']
        }
        labels = label_map.get(n_groups)
    else:
        # Etiquetas genéricas para más de 5 grupos
        labels = [f'NIVEL_{i+1}' for i in range(n_groups)]

    # --- Cálculo de Umbrales (Quantiles) ---
    thresholds = {}
    # Puntos de corte para los cuantiles. Ej: para n_groups=4, queremos [0.25, 0.5, 0.75]
    quantile_points = [i / n_groups for i in range(1, n_groups)]

    for column in df.columns:
        # Se calcula un umbral por cada punto de corte
        thresholds[column] = df[column].quantile(quantile_points).tolist()

    # --- Procesamiento de Condiciones ---
    descriptions = []
    for condition in condition_list:
        features = {}
        # Patrón Regex para extraer nombre de variable y sus rangos
        pattern = r'(\d+\.?\d*)\s*<=\s*(\w+)\s*<=\s*(\d+\.?\d*)'
        matches = re.findall(pattern, condition)

        for match in matches:
            min_value_str, feature_name, max_value_str = match
            min_value = float(min_value_str)
            max_value = float(max_value_str)
            
            # Se usa el valor promedio del rango para la categorización
            avg_value = (min_value + max_value) / 2

            if feature_name in thresholds:
                feature_thresholds = thresholds[feature_name]
                
                # Se busca en qué intervalo cae el valor promedio
                # np.searchsorted encuentra el índice donde el elemento debería insertarse
                # para mantener el orden. Esto nos da directamente el índice de la categoría.
                category_index = np.searchsorted(feature_thresholds, avg_value)
                
                # Asignamos la etiqueta correspondiente
                category = labels[category_index]
                features[feature_name] = category
            else:
                # Si la variable de la condición no está en el DataFrame
                features[feature_name] = 'N/A'

        # Se construye la descripción final para la condición
        description_parts = [f"{feature} es {category}" for feature, category in features.items()]
        description = ', '.join(description_parts) + '.'
        descriptions.append(description)

    # --- Retorno de Resultados ---
    result = {'respuestas': descriptions}
    return result
