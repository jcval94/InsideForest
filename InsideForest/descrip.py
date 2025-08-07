from __future__ import annotations

import ast
import copy
import json
import logging
import os
import re
from functools import lru_cache
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

try:  # OpenAI SDK is optional
    from openai import OpenAI  # type: ignore
except Exception as exc:  # pragma: no cover - import failure path
    OpenAI = None  # type: ignore[assignment]
    _client = None
    logging.warning("OpenAI package not available (%s)", exc)
else:  # Only executed if import succeeded
    try:
        from google.colab import userdata  # type: ignore
        OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    except Exception:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    if not OPENAI_API_KEY:
        _client = None
        logging.warning("OPENAI_API_KEY not set; GPT features disabled")
    else:
        try:
            _client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as exc:
            _client = None
            logging.warning("OpenAI deshabilitado (%s)", exc)

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
        logger.error("Data list is empty")
        return None

    try:
        # Calcular el histograma
        counts, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    except Exception as exc:
        logger.exception("Error computing histogram: %s", exc)
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
        logger.exception("Error applying savgol_filter: %s", exc)
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
        raise TypeError("df must be a pandas DataFrame")

    df_replaced = df.copy()
    replace_info = {}
    
    # Ordenar las claves por longitud descendente para evitar conflictos en subcadenas
    sorted_keys = sorted(var_rename.keys(), key=len, reverse=True)
    escaped_keys = [re.escape(k) for k in sorted_keys]
    pattern = re.compile('|'.join(escaped_keys))
    
    for col in columns:
        if col not in df_replaced.columns:
            logger.warning(
                f"Warning: column '{col}' not found in the DataFrame."
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
            logger.exception("Error replacing values in column %s: %s", col, exc)
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
        logger.exception("Error calling the OpenAI API: %s", exc)
        return {'responses': []}

    # Dividir la respuesta en una lista de descripciones por línea
    descriptions = respuesta.choices[0].message.content.strip().split("\n")
    descriptions = [desc.strip() for desc in descriptions if desc.strip()]

    # Return a dictionary with the responses
    result = {'responses': descriptions}
    return result

import re
import numpy as np   # Asegúrate de tenerlo importado
import pandas as pd  # solo para el ejemplo, tu código ya lo usa

def _categorize_conditions(condition_list, df, n_groups=2, handle_bools=False):

    # ── Validaciones idénticas ──
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {'error': 'A valid, non-empty pandas DataFrame is required.'}
    if not isinstance(n_groups, int) or n_groups < 2:
        return {'error': 'n_groups must be an integer greater than or equal to 2.'}
    
    df.columns = df.columns.str.replace(' ', '_')
    # ── Etiquetas de percentil ──
    def format_percentile(p):
        return str(int(p)) if float(p).is_integer() else f"{p:.2f}".rstrip('0').rstrip('.')

    labels = [
        f"Percentile {format_percentile((i + 1) * 100 / n_groups)}"
        for i in range(n_groups)
    ]

    # ── Umbrales idénticos ──
    bool_cols = df.select_dtypes(include="bool").columns.tolist() if handle_bools else []
    quantile_points = [i / n_groups for i in range(1, n_groups)]
    thresholds = {c: df[c].quantile(quantile_points).tolist()
                  for c in df.columns if c not in bool_cols}

    # ── Patrones mejorados ──────────────────────────────────────────────
    # 1) Nombre de columna: letras, números, _, -, ., (), [] …
    var_name = r'[A-Za-z0-9_\-\.\(\)\[\]]+'

    # 2) Número:
    #    - Parte entera opcional
    #    - Punto decimal opcional
    #    - Notación científica opcional (e o E, con signo opcional)
    #    - Signo ± al inicio opcional
    num = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

    pattern_num  = rf'({num})\s*<=\s*({var_name})\s*<=\s*({num})'
    pattern_bool = rf'({var_name})\s*==\s*(True|False)'

    # ── Procesamiento idéntico ──
    descriptions = []
    for condition in condition_list:
        tokens = [t.strip() for t in re.split(r'(?i)\band\b', condition)]

        parts = []
        for token in tokens:
            m_num = re.match(pattern_num, token)
            if m_num:
                min_v, feat, max_v = m_num.groups()
                avg_value = (float(min_v) + float(max_v)) / 2
                if feat in thresholds:
                    cuts = thresholds[feat]
                    # In case multiple percentiles produce the same cut value,
                    # ``side='right'`` ensures we keep the higher percentile
                    # label for that threshold
                    category = labels[np.searchsorted(cuts, avg_value, side="right")]
                elif handle_bools and feat in bool_cols:
                    category = 'TRUE' if avg_value >= 0.5 else 'FALSE'
                else:
                    category = 'N/A'
                parts.append(f"{feat} = {category}")
                continue

            if handle_bools:
                m_bool = re.match(pattern_bool, token)
                if m_bool:
                    feat, val = m_bool.groups()
                    if feat in bool_cols:
                        category = 'TRUE' if val == 'True' else 'FALSE'
                    else:
                        category = 'N/A'
                    parts.append(f"{feat} = {category}")

        descriptions.append(', '.join(parts) + '.')

    return {'responses': descriptions}


def categorize_conditions(condition_list, df, n_groups=2):
    return _categorize_conditions(condition_list, df, n_groups=n_groups, handle_bools=True)


def categorize_conditions_generalized(condition_list, df, n_groups=2):
    """Wrapper para retrocompatibilidad.

    Permite procesar condiciones con soporte de booleanos.
    """
    return _categorize_conditions(condition_list, df, n_groups=n_groups, handle_bools=True)



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
    descriptions = cat_results["responses"]

    var_pattern = r"(\w+)\s*=\s*([^,.]+(?:\.[^,.]+)*)"
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


import pandas as pd
from typing import Union, Dict, List, Any

def _parse_relative_description(desc: Union[str, float, None]) -> Dict[str, str]:
    """
    Convierte una cadena del tipo
        'petal_width_(cm) = Percentile 80, sepal_length_(cm) = Percentile 40.'
    en un diccionario
        {'petal_width_(cm)': 'PERCENTILE 80',
         'sepal_length_(cm)': 'PERCENTILE 40'}
    
    Si la entrada no es texto válido ‒NaN, None, ''‒ devuelve {}.
    """
    if not isinstance(desc, str) or not desc.strip():
        return {}
    
    # Eliminamos el punto final (si existe) y separamos por comas.
    tokens: List[str] = [t.strip().rstrip('.')            # “petal = Percentile 25”
                         for t in desc.split(',')
                         if t.strip()]
    
    pares: Dict[str, str] = {}
    for token in tokens:
        if ' = ' in token:                                # Seguro que existe por formato
            var, cat = token.split(' = ', 1)
            pares[var.strip()] = cat.strip().upper()      # Normalizamos a MAYÚSCULAS
    return pares


def expandir_categorias(
        df: pd.DataFrame,
        desc_col: str = 'cluster_desc_relative',
        inplace: bool = False
    ) -> pd.DataFrame:
    """
    A partir de *df[desc_col]* crea una columna nueva por cada variable
    mencionada en **cluster_desc_relative** y coloca la categoría (Percentile 10,
    Percentile 25…);
    donde la variable no se mencione, deja `NaN`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.
    desc_col : str, default 'cluster_desc_relative'
        Nombre de la columna que contiene las descripciones.
    inplace : bool, default False
        - True : añade las columnas directamente en `df` y lo devuelve.
        - False: devuelve **una copia** con las columnas nuevas, dejando
          `df` intacto.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas adicionales de categorías.
    """
    if desc_col not in df.columns:
        raise KeyError(f'Column "{desc_col}" does not exist in the DataFrame.')
    
    # 1) Transformamos cada descripción en un dict variable → categoría.
    mapeos: pd.Series = df[desc_col].apply(_parse_relative_description)
    
    # 2) Convertimos la serie de dicts en DataFrame columnar (wide).
    cat_df: pd.DataFrame = pd.json_normalize(mapeos)
    # Las columnas ausentes en una fila quedan automáticamente como NaN.
    
    # 3) Unimos al original.
    if inplace:
        for col in cat_df.columns:
            df[col] = cat_df[col]
        return df
    else:
        return pd.concat([df.copy(), cat_df], axis=1)



# ───────────────────────────── utilidades ─────────────────────────────

def feature_cols(df: pd.DataFrame) -> List[str]:
    """Devuelve las columnas de características (todo lo que
    esté a la derecha de 'cluster_desc_relative')."""
    idx = df.columns.get_loc('cluster_desc_relative')
    return list(df.columns[idx + 1:])


def encode_features(df: pd.DataFrame,
                    ord_map: Dict[str, int],
                    *,
                    scale: bool = True) -> pd.DataFrame:
    """
    Convierte variables ordinales según `ord_map` y, opcionalmente,
    escala numéricamente cada columna al rango [0, 1].

    Parameters
    ----------
    df : DataFrame fuente
    ord_map : dict  mapa de texto→valor ordinal
    scale : bool    si True, normaliza columnas no constantes

    Returns
    -------
    DataFrame con las columnas transformadas
    """
    feats = feature_cols(df)
    enc = df[feats].copy()

    for c in feats:
        # mapea solo objetos (categóricas/ordinales)
        if enc[c].dtype == object:
            enc[c] = enc[c].map(ord_map)

        # re-escala si tiene más de un valor distinto
        if scale and enc[c].dropna().nunique() > 1:
            enc[c] = (enc[c] - enc[c].min()) / (enc[c].max() - enc[c].min())

    return enc


def similarity(a_idx: int,
               b_idx: int,
               df_enc: pd.DataFrame,
               *,
               cov_weight: bool = True) -> float:
    """Similitud (1-distancia media absoluta) entre dos filas ya codificadas."""
    v_a, v_b = df_enc.iloc[a_idx], df_enc.iloc[b_idx]
    mask = ~(v_a.isna() | v_b.isna())
    if mask.sum() == 0:
        return 0.0
    d = np.abs(v_a[mask] - v_b[mask]).mean()
    sim = 1 - d
    return sim * (mask.sum() / df_enc.shape[1]) if cov_weight else sim


# ───────────────────────────── API de alto nivel ─────────────────────────────

def similarity_matrix(df: pd.DataFrame,
                      ord_map: Dict[str, int],
                      *,
                      cov_weight: bool = True) -> pd.DataFrame:
    """
    Matriz de similitud S (diagonal = 1).

    Parameters
    ----------
    df         : DataFrame con los clusters y sus features
    ord_map    : dict mapa ordinal que se usará en `encode_features`
    cov_weight : bool pondera la similitud por cobertura de datos no-nulos
    """
    df_enc = encode_features(df.reset_index(drop=True), ord_map)
    n = len(df_enc)
    clusters = df['cluster'].tolist()
    S = pd.DataFrame(np.eye(n), index=clusters, columns=clusters)

    for i, j in combinations(range(n), 2):
        s = similarity(i, j, df_enc, cov_weight=cov_weight)
        S.iat[i, j] = S.iat[j, i] = round(s, 3)
    return S


def cluster_pairs_sim(df: pd.DataFrame,
                      ord_map: Dict[str, int],
                      *,
                      metric: str = 'cluster_ef_sample',
                      cov_weight: bool = True) -> pd.DataFrame:
    """
    Devuelve un DataFrame con:
      cluster_1 | cluster_2 | similitud | delta_<metric> | score

    El score prioriza pares con alta similitud y mejora positiva
    en la métrica indicada.
    """
    df_r = df.reset_index(drop=True)
    df_enc = encode_features(df_r, ord_map)
    n = len(df_r)

    # — recolectamos pares —
    pairs: List[Tuple[str, str, float, float]] = []
    for i, j in combinations(range(n), 2):
        sim = similarity(i, j, df_enc, cov_weight=cov_weight)
        delta = abs(df_r.at[j, metric] - df_r.at[i, metric])
        pairs.append((df_r.at[i, 'cluster'], df_r.at[j, 'cluster'], sim, delta))

    # — escala robusta para deltas positivos —
    pos_deltas = np.array([d for *_, d in pairs if d > 0])
    mad = np.median(np.abs(pos_deltas - np.median(pos_deltas))) + 1e-9
    sigma_rob = 1.4826 * mad if pos_deltas.size else 1.0

    # — calculamos score —
    rows = []
    for c1, c2, sim, delta in pairs:
        grow = 1 - np.exp(-max(0, delta) / sigma_rob)
        score = sim * grow
        rows.append([c1, c2, round(sim, 3), round(delta, 3), round(score, 3)])

    return pd.DataFrame(rows,
                        columns=['cluster_1', 'cluster_2',
                                 'similitud', f'delta_{metric}', 'score'])


def get_frontiers(df_datos_descript: pd.DataFrame,
                  df: pd.DataFrame,
                  divide: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Genera fronteras entre clusters a partir de descripciones.

    Parameters
    ----------
    df_datos_descript : pd.DataFrame
        DataFrame con descripciones de clusters.
    df : pd.DataFrame
        DataFrame de referencia para calcular percentiles.
    divide : int, default 5
        Número de divisiones para ``categorize_conditions``.

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame)
        ``df_datos_explain`` con columnas expandidas y ``frontiers`` con
        pares de clusters ordenados por score.
    """
    descrip_generales = [
        x for x in df_datos_descript['cluster_descripcion'].unique().tolist()
        if isinstance(x, str)
    ]

    descrip = categorize_conditions(descrip_generales, df, divide)
    df_datos_descript['cluster_desc_relative'] = (
        df_datos_descript['cluster_descripcion'].replace(
            {k: v for k, v in zip(descrip_generales, descrip['responses'])}
        )
    )

    df_datos_explain = expandir_categorias(df_datos_descript)
    df_datos_explain = df_datos_explain[df_datos_explain['cluster_n_sample'] > 0]

    ORD_MAP_LOCAL = {f"PERCENTILE {i}": i for i in range(1, 101)}
    df_datos_explain_gen = (
        df_datos_explain.sort_values('cluster_n_sample', ascending=False).head(40)
    )
    similarity_matrix(df_datos_explain_gen, ORD_MAP_LOCAL)
    frontiers = cluster_pairs_sim(
        df_datos_explain_gen, ORD_MAP_LOCAL, metric='cluster_ef_sample'
    )
    frontiers.sort_values('score', ascending=False, inplace=True)
    return df_datos_explain, frontiers


# ╭──────────────────╮
# │  INIT OpenAI SDK │
# ╰──────────────────╯
# Las utilidades siguientes permiten generar hipótesis y traducir
# reglas en texto comprensible.


def get_range_re() -> re.Pattern:
    pat = r"""
        (?P<low>-?\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][-+]?\d+)?)\s*
        (?:<=|<)\s*
        (?P<tok>[A-Za-z_][A-Za-z0-9_]*)\s*
        (?:<=|<)\s*
        (?P<high>-?\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][-+]?\d+)?)\s*
    """
    return re.compile(pat, re.VERBOSE)


@lru_cache
def _tpl_rules(lang: str) -> dict[str, str]:
    base = {
        "between": {
            "es": "{label} entre {low:,.2f} y {high:,.2f}",
            "en": "{label} between {low:,.2f} and {high:,.2f}",
        },
        "is":      {"es": "Es {label}",    "en": "Is {label}"},
        "not":     {"es": "No es {label}", "en": "Is not {label}"},
        "generic": {"es": "Condición sobre {label}", "en": "Condition on {label}"},
    }
    return {k: v[lang] for k, v in base.items()}


def get_FALLBACK_LABELS(df_meta_sub):

  idiomas_d = [y.split('.')[-1] for y in [x for x in df_meta_sub.columns if 'description' in x]]
  df_lang = {}
  for y in idiomas_d:
    df_by_lang = []
    for x in df_meta_sub.columns:
      if x.endswith('.'+y):
        df_by_lang.append(x)

    df_lang[y] = (df_meta_sub[['rule_token']+df_by_lang].drop_duplicates())


  dicts_rts = [{(x, ess_):z for x, y, z in df_lang[ess_].values} for ess_ in df_lang.keys()]

  def concatenate_dictionaries(list_of_dicts):
    result_dict = {}
    for d in list_of_dicts:
      result_dict.update(d)
    return result_dict

  return concatenate_dictionaries(dicts_rts)


def _extract_tokens(series: pd.Series) -> Set[str]:
    token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    return {m.group(0)
            for txt in series.astype(str)
            for m in token_re.finditer(txt)}


def _meta_lookup(token: str, meta: pd.DataFrame, *, lang: str
                 ) -> Tuple[str, str, str, str | None]:
    col_lbl = f"identity.label_i18n.{lang}"
    col_des = f"identity.description_i18n.{lang}"
    row     = meta.loc[meta["rule_token"] == token]
    _FALLBACK_LABELS = get_FALLBACK_LABELS(meta)

    if not row.empty:
        r = row.iloc[0]
        return (
            str(r.get(col_lbl) or _FALLBACK_LABELS.get((token, lang), token)).capitalize(),
            str(r.get(col_des) or ""),
            str(r.get("domain.categorical.codes") or ""),
            str(r.get("actionability.side_effects") or "") or None
        )
    return (_FALLBACK_LABELS.get((token, lang), token).capitalize(), "", "", None)


def _rule_to_text(rule: str, meta_df: pd.DataFrame, *, lang: str) -> str:
    toks = _extract_tokens(pd.Series([rule]))
    tok  = next(iter(toks)) if toks else None

    m = get_range_re().search(rule)
    low = high = None
    if m:
        low, high, tok = float(m["low"].replace("_", "")), float(m["high"].replace("_", "")), m["tok"]

    if tok is None:
        return rule

    label, *_ = _meta_lookup(tok, meta_df, lang=lang)
    tpl = _tpl_rules(lang)
    if tok.startswith("cat__"):
        if low is not None and high is not None:
            if high <= 0.5: return tpl["not"].format(label=label)
            if low  >= 0.5: return tpl["is"].format(label=label)
        return tpl["generic"].format(label=label)

    if low is not None and high is not None:
        return tpl["between"].format(label=label, low=low, high=high)
    return tpl["generic"].format(label=label)


def _list_rules_to_text(col, meta_df, *, lang: str, placeholder: str = "—") -> str:
    """
    Convierte lista o str-lista a texto legible.
    Devuelve 'placeholder' si la lista está vacía.
    """
    # 1) Normaliza a lista
    if col is None or (isinstance(col, float) and pd.isna(col)):
        col = []
    elif isinstance(col, str):
        try:
            col = ast.literal_eval(col)
        except Exception:
            col = [col]
    if not isinstance(col, (list, tuple)):
        col = [col]

    # 2) Sin reglas → placeholder
    if len(col) == 0 or all(str(r).strip() == "" for r in col):
        return placeholder

    # 3) Traducción normal
    return ", ".join(_rule_to_text(r, meta_df, lang=lang) for r in col)


def _local_significance(delta: float, *, lang: str) -> str:
    if lang == "es":
        return "significativo" if abs(delta) > 0.1 else "exploratorio"
    return "significant" if abs(delta) > 0.1 else "exploratory"


def _local_hypothesis_text(inter_txt: str, a_txt: str, b_txt: str,
                           p_a: float, p_b: float, delta: float,
                           target_lbl: str, side: list[str], *,
                           lang: str) -> str:
    """Genera texto local para la hipótesis (A vs B con intersección común)."""
    sig_word = _local_significance(delta, lang=lang)

    side_txt = ""
    if side:
        se = "\n- " + "\n- ".join(sorted(set(side)))
        side_txt = ("\n\n**Posibles efectos secundarios**" if lang == "es"
                    else "\n\n**Possible side-effects**") + se

    # Encabezados
    header_ctx = ("**Reglas compartidas (intersección)**"
                  if lang == "es" else "**Shared rules (intersection)**")
    header_a   = "**Subgrupo A**" if lang == "es" else "**Subgroup A**"
    header_b   = "**Subgrupo B**" if lang == "es" else "**Subgroup B**"
    header_an  = ("**Análisis y recomendaciones**"
                  if lang == "es" else "**Analysis & Recommendations**")

    # Cuerpo de análisis
    analysis = (
        f"Resultado {sig_word}. Ajustar las variables que diferencian A y B "
        f"podría aumentar la probabilidad de {target_lbl.lower()}."
        if lang == "es" else
        f"{sig_word.capitalize()} result. Adjusting features that differentiate "
        f"A and B could raise the probability of {target_lbl.lower()}."
    )

    return (
        f"{header_ctx}\n- Reglas: **{inter_txt}**\n\n"
        f"{header_a}\n"
        f"- Condiciones adicionales: **{a_txt}**\n"
        f"- Probabilidad de {target_lbl}: **{p_a:.2%}**\n\n"
        f"{header_b}\n"
        f"- Condiciones adicionales: **{b_txt}**\n"
        f"- Probabilidad de {target_lbl}: **{p_b:.2%}**\n"
        f"- Diferencia B – A: **{delta:+.2%}**\n\n"
        f"{header_an}\n{analysis}"
        f"{side_txt}"
    )


def _gpt_hypothesis(payload: dict[str, Any], *,
                    model: str,
                    temperature: float) -> Optional[str]:
    """Wrapper: envía a GPT el payload y devuelve el reporte estructurado."""
    if _client is None:
        return None
    try:
        rsp = _client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data-science expert assistant.\n"
                        "Return **one** Markdown report (≤ 150 words) using H2 headings (##).\n\n"
                        "Sections in order:\n"
                        "1) Shared features (intersection)\n"
                        "2) Subgroup A (extra conditions)\n"
                        "3) Subgroup B (extra conditions)\n"
                        "4) Comparison groups (intersection + A  vs. intersection + B)\n"
                        "5) Hypothesis & key metrics (pA, pB)\n"
                        "6) Possible side-effects (if any)\n"
                        "7) A/B Testing Actions"
                        "Choose Language with the 'lang' field "
                        "and honour the variable labels supplied."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
        )
        if not rsp.choices:
            logging.error("GPT call returned no choices")
            return None
        msg = rsp.choices[0].message
        content = getattr(msg, "content", None)
        if not content:
            logging.error("GPT call returned empty content")
            return None
        return content.strip()
    except Exception as err:
        logging.error("GPT call failed → %s", err)
        return None


def generar_hypotesis(meta_df: pd.DataFrame,
                      exp_df: pd.DataFrame,
                      *,
                      target: str,
                      lang: str = "es",
                      use_gpt: bool = False,
                      gpt_model: str = "gpt-4o-mini",
                      temperature: float = 0.2) -> str:
    """
    Crea un reporte de hipótesis comparando los subgrupos
    (intersección ∧ A) vs (intersección ∧ B).
    """
    row = exp_df.iloc[0]

    # Posibles efectos secundarios
    side = []
    for col in ("only_cluster_A", "only_cluster_B"):
        for tok in _extract_tokens(pd.Series([row[col]])):
            se = _meta_lookup(tok, meta_df, lang=lang)[3]
            if se:
                side.append(se)

    # ===== GPT PATH =====
    if use_gpt and _client is not None:
        payload = {
            "lang": lang,
            "target": _meta_lookup(target, meta_df, lang=lang)[0],
            "target_description": _meta_lookup(target, meta_df, lang=lang)[1],
            "shared_rules": row["intersection"],
            "subgroup_a":   row["only_cluster_A"],
            "subgroup_b":   row["only_cluster_B"],
            "metrics": {
                "p_a":   row["cluster_ef_A"],
                "p_b":   row["cluster_ef_B"],
                "delta": row["delta_ef"],
            },
            "tokens_info": {
                t: {
                    "label":        _meta_lookup(t, meta_df, lang=lang)[0],
                    "description":  _meta_lookup(t, meta_df, lang=lang)[1],
                    "domain":       _meta_lookup(t, meta_df, lang=lang)[2],
                    "side_effect":  _meta_lookup(t, meta_df, lang=lang)[3],
                }
                for t in (
                    _extract_tokens(pd.Series([row["intersection"]]))
                    | _extract_tokens(pd.Series([row["only_cluster_A"]]))
                    | _extract_tokens(pd.Series([row["only_cluster_B"]]))
                )
            }
        }

        txt = _gpt_hypothesis(payload, model=gpt_model, temperature=temperature)
        if txt:
            return txt

    # ===== LOCAL PATH =====
    inter_txt = _list_rules_to_text(row["intersection"],      meta_df, lang=lang)
    a_txt     = _list_rules_to_text(row["only_cluster_A"],    meta_df, lang=lang)
    b_txt     = _list_rules_to_text(row["only_cluster_B"],    meta_df, lang=lang)
    target_lbl = _meta_lookup(target, meta_df, lang=lang)[0]

    return _local_hypothesis_text(
        inter_txt, a_txt, b_txt,
        row["cluster_ef_A"], row["cluster_ef_B"], row["delta_ef"],
        target_lbl, side, lang=lang
    )


