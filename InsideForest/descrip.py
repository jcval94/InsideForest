import re
from openai import OpenAI
import pandas as pd
import copy
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import re
from .regions import expandir_clusters_binario

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



def get_descripciones_valiosas(df_datos_descript,df_datos_clusterizados, TARGETS, var_rename, 
                               inflex_pond_sup = .4, inflex_pond_inf=.5):

    df_datos_descript = df_datos_descript.sort_values('cluster_ponderador', ascending=False)
    # descrip_generales = [x for x in df_datos_descript['cluster_descripcion'].unique().tolist() if type('')==type(x)]
    df_datos_clusterizados_desc = df_datos_clusterizados.merge(df_datos_descript, on='cluster', how='left')
    stacked_data = df_datos_clusterizados_desc.groupby([TARGETS[0], 'cluster']).size().unstack(fill_value=0)
    # best_clusters = df_datos_descript['cluster'].head(10).values.tolist()

    proporcion_real = df_datos_clusterizados_desc[TARGETS[0]].value_counts(normalize=True).loc[1]
    stacked_data_total = stacked_data.sum(axis=0)
    proprcin_ = (stacked_data/stacked_data.sum(axis=0)).loc[1]
    los_custers = pd.concat([proprcin_/proporcion_real, stacked_data_total], axis=1).sort_values(0, ascending=False)
    los_custers_valiosos = los_custers[los_custers[1]>1].copy()

    los_custers_valiosos_original = copy.deepcopy(los_custers_valiosos)
    # Selecciona las columnas numéricas para la estandarización
    numeric_cols = los_custers_valiosos.select_dtypes(include=np.number).columns
    # Crea un StandardScaler
    scaler = StandardScaler()
    # Ajusta y transforma las columnas numéricas
    los_custers_valiosos[numeric_cols] = scaler.fit_transform(los_custers_valiosos[numeric_cols])

    los_custers_valiosos_original['importancia'] =los_custers_valiosos.sum(axis=1)
    los_custers_valiosos_original.sort_values('importancia', ascending=False)

    punto = primer_punto_inflexion_decreciente(los_custers_valiosos_original[0], bins=20, window_length=5, polyorder=2)
    punto_1 = primer_punto_inflexion_decreciente(los_custers_valiosos_original[1], bins=20, window_length=5, polyorder=2)

    los_custers_valiosos_original_cond = los_custers_valiosos_original[0]>punto*inflex_pond_sup

    los_custers_valiosos_original_cond_1 = los_custers_valiosos_original[1]>punto_1

    los_custers_valiosos_original_cond_2 = los_custers_valiosos_original[0]<inflex_pond_inf
    

    los_custers_valiosos_original = los_custers_valiosos_original[los_custers_valiosos_original_cond\
                                                                  |los_custers_valiosos_original_cond_1\
                                                                    |los_custers_valiosos_original_cond_2]

    df_datos_descript_valiosas = df_datos_descript[df_datos_descript['cluster'].isin(los_custers_valiosos_original.index.tolist())]

    df_datos_descript_valiosas,_ = replace_with_dict(df_datos_descript_valiosas, ['cluster_descripcion'], var_rename)
    df_datos_descript_valiosas = df_datos_descript_valiosas.merge(proprcin_.reset_index(), on='cluster', how='left')
    df_datos_descript_valiosas = df_datos_descript_valiosas.merge(los_custers.reset_index(), on='cluster', how='left')
    df_datos_descript_valiosas = df_datos_descript_valiosas.rename(columns={'1_x':'Probabilidad','1_y':'N_probabilidad',0:'Soporte'})

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


def categorize_conditions(condition_list, df=None):
    descriptions = []

    # If df is provided, calculate thresholds using quantiles
    if df is not None:
        thresholds = {}
        for column in df.columns:
            # Calculate quantiles for low, medium, high categories
            low = df[column].quantile(0.33)
            high = df[column].quantile(0.66)
            thresholds[column] = {'low': low, 'high': high}

    for condition in condition_list:
        features = {}
        # Regex pattern to extract variable ranges
        pattern = r'(\d+\.?\d*) <= (\w+) <= (\d+\.?\d*)'
        matches = re.findall(pattern, condition)

        for match in matches:
            min_value, feature_name, max_value = match
            min_value = float(min_value)
            max_value = float(max_value)
            # Calculate average value
            avg_value = (min_value + max_value) / 2
            # Categorize based on thresholds
            if feature_name in thresholds:
                low = thresholds[feature_name]['low']
                high = thresholds[feature_name]['high']
                # Determine category based on where the average value falls within the thresholds
                if avg_value <= low:
                    category = 'BAJO'
                elif avg_value <= high:
                    category = 'MEDIO'
                else:
                    category = 'ALTO'
                features[feature_name] = category
            else:
                features[feature_name] = 'N/A'

        # Create description using the categories
        description_parts = []
        for feature, category in features.items():
            description_parts.append(f"{feature} es {category}")
        description = ', '.join(description_parts) + '.'
        descriptions.append(description)

    # Return a dictionary with the responses
    result = {'respuestas': descriptions}
    return result


def get_corr_clust(df_datos_clusterizados):
    df_clusterizado_diff = df_datos_clusterizados[['clusters_list']].drop_duplicates()
    df_clusterizado_diff['n_ls'] = df_clusterizado_diff.apply(lambda x: len(x['clusters_list']), axis=1)
    df_expanded = expandir_clusters_binario(df_clusterizado_diff,'clusters_list','cluster_')
    cluster_cols = [x for x in df_expanded.columns if 'cluster_' in x]

    df_corr=df_expanded[cluster_cols].corr()

    return df_corr

def obtener_clusters(df_clust, cluster_objetivo, n=5, direccion='ambos'):
    """
    Retorna los clusters más cercanos o menos correlacionados con el cluster objetivo según la dirección especificada.

    Parámetros:
    - corr_matrix (pd.DataFrame): Matriz de correlación de los clusters.
    - cluster_objetivo (str): Nombre del cluster para el cual se buscan correlaciones.
    - n (int): Número de clusters a retornar.
    - direccion (str): Dirección de interés para la correlación. Puede ser:
        - 'arriba': Para las correlaciones más positivas.
        - 'abajo': Para las correlaciones más negativas.
        - 'ambos': Para considerar ambas direcciones (positiva y negativa).
        - 'bottom': Para las correlaciones más cercanas a cero (menos correlacionadas).

    Retorna:
    - pd.Series: Series con los nombres de los clusters y sus valores de correlación.
    """
    corr_matrix = get_corr_clust(df_clust)

    if cluster_objetivo not in corr_matrix.columns:
        raise ValueError(f"El cluster '{cluster_objetivo}' no se encuentra en la matriz de correlación.")
    
    # Obtener la serie de correlaciones para el cluster objetivo y eliminar la autocorrelación
    correlaciones = corr_matrix[cluster_objetivo].drop(labels=[cluster_objetivo])
    
    if direccion == 'arriba':
        # Ordenar de mayor a menor (correlaciones positivas más fuertes)
        correlaciones_ordenadas = correlaciones.sort_values(ascending=False)
        top_n = correlaciones_ordenadas.head(n)
    elif direccion == 'abajo':
        # Ordenar de menor a mayor (correlaciones negativas más fuertes)
        correlaciones_ordenadas = correlaciones.sort_values(ascending=True)
        top_n = correlaciones_ordenadas.head(n)
    elif direccion == 'ambos':
        # Obtener las top n positivas y las top n negativas
        top_n_arriba = correlaciones.sort_values(ascending=False).head(n)
        top_n_abajo = correlaciones.sort_values(ascending=True).head(n)
        top_n = pd.concat([top_n_arriba, top_n_abajo])
    elif direccion == 'bottom':
        # Obtener las n correlaciones más cercanas a cero
        correlaciones_ordenadas = correlaciones.reindex(correlaciones.abs().sort_values(ascending=True).index)
        top_n = correlaciones_ordenadas.head(n)
    else:
        raise ValueError("El parámetro 'direccion' debe ser 'arriba', 'abajo', 'ambos' o 'bottom'.")
    
    return top_n.sort_values(ascending=False)

