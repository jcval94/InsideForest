from __future__ import annotations

import ast
import json
import logging
import os
import re
from functools import lru_cache
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
_client = None
_client_key: Optional[str] = None


def get_openai_client(api_key: Optional[str] = None):
    """Return a cached OpenAI client if credentials are available.

    Parameters
    ----------
    api_key : str or None, optional
        Explicit API key to use. When ``None`` the helper tries Colab userdata
        first and then the ``OPENAI_API_KEY`` environment variable.
    """

    global _client, _client_key

    resolved_key: Optional[str] = api_key

    if not resolved_key:
        try:  # Optional dependency – only available in Colab.
            from google.colab import userdata  # type: ignore

            resolved_key = userdata.get("OPENAI_API_KEY")
        except Exception:
            resolved_key = os.getenv("OPENAI_API_KEY")

    if not resolved_key:
        return None

    if _client is not None and resolved_key == _client_key:
        return _client

    try:  # OpenAI SDK is optional
        from openai import OpenAI  # type: ignore
    except Exception:
        return None
    try:
        _client = OpenAI(api_key=resolved_key)
        _client_key = resolved_key
    except Exception:
        _client = None
        _client_key = None
    return _client


def first_decreasing_inflection_point(data, bins=10, window_length=5, polyorder=2):
    """
    Find the first decreasing inflection point in a histogram.

    Parameters:
    - data: array-like, data used to build the histogram.
    - bins: int or sequence, number of bins or their edges.
    - window_length: window length for the Savitzky-Golay filter.
    - polyorder: polynomial order for the Savitzky-Golay filter.

    Returns:
    - inflection_point: bin value where the first decreasing inflection occurs.
    """

    if len(data) == 0:
        logger.error("Data list is empty")
        return None

    try:
        # Compute the histogram
        counts, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    except Exception as exc:
        logger.exception("Error computing histogram: %s", exc)
        return None

    # Smooth the histogram to reduce noise
    # Ensure window_length is odd and smaller than counts length
    if window_length >= len(counts):
        window_length = len(counts) - 1 if len(counts) % 2 == 0 else len(counts)
    if window_length % 2 == 0:
        window_length += 1
    if window_length < polyorder + 2:
        window_length = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3

    try:
        counts_smooth = savgol_filter(
            counts, window_length=window_length, polyorder=polyorder
        )
    except Exception as exc:
        logger.exception("Error applying savgol_filter: %s", exc)
        return None

    # Compute the second derivative
    second_derivative = np.gradient(np.gradient(counts_smooth))

    # Find inflection points where second derivative changes sign
    # Positive to negative indicates concavity change downward
    sign_changes = np.diff(np.sign(second_derivative))
    # A change from +1 to -1 in the second derivative
    inflection_indices = (
        np.where(sign_changes < 0)[0] + 1
    )  # +1 to correct the shift from diff

    if len(inflection_indices) == 0:
        return None  # No decreasing inflection point found

    # Select the first decreasing inflection point
    first_inflection = bin_centers[inflection_indices[0]]

    return first_inflection


def replace_with_dict(df, columns, var_rename):
    """
    Replace values in specified DataFrame columns using a dictionary.
    Matches are exact or substrings containing dictionary keys.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    columns : list of str
        Columns where replacements will be applied.
    var_rename : dict
        Mapping of values to replace and their new values.

    Returns
    -------
    df_replaced : pd.DataFrame
        DataFrame with replacements in the specified columns.
    replace_info : dict
        Information needed to revert the replacements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df_replaced = df.copy()
    replace_info = {}

    # Sort keys by descending length to avoid substring conflicts
    sorted_keys = sorted(var_rename.keys(), key=len, reverse=True)
    escaped_keys = [re.escape(k) for k in sorted_keys]
    pattern = re.compile("|".join(escaped_keys))

    for col in columns:
        if col not in df_replaced.columns:
            logger.warning(f"Warning: column '{col}' not found in the DataFrame.")
            continue

        # Store replacement info per column
        replace_info[col] = {"var_rename": var_rename.copy()}

        # Replacement function
        def repl(match):
            return var_rename[match.group(0)]

        try:
            df_replaced[col] = (
                df_replaced[col].astype(str).str.replace(pattern, repl, regex=True)
            )
        except Exception as exc:
            logger.exception("Error replacing values in column %s: %s", col, exc)
            continue

    return df_replaced, replace_info


def _prepare_cluster_data(
    df_descriptive: pd.DataFrame,
    df_clustered: pd.DataFrame,
    targets: List[str],
):
    """Compute cluster statistics and auxiliary tables."""

    sorted_descriptions = df_descriptive.sort_values(
        "cluster_weight", ascending=False
    )
    clusters_with_desc = df_clustered.merge(
        sorted_descriptions, on="cluster", how="left"
    )
    class_cluster_counts = (
        clusters_with_desc.groupby([targets[0], "cluster"]).size().unstack(fill_value=0)
    )
    global_positive_rate = (
        clusters_with_desc[targets[0]].value_counts(normalize=True).loc[1]
    )
    cluster_totals = class_cluster_counts.sum(axis=0)
    class1_rate_by_cluster = (
        class_cluster_counts / class_cluster_counts.sum(axis=0)
    ).loc[1]
    class1_ratio = class1_rate_by_cluster / global_positive_rate
    class1_ratio.name = 0
    cluster_totals.name = 1
    cluster_stats = pd.concat([class1_ratio, cluster_totals], axis=1).sort_values(
        by=0, ascending=False
    )
    valuable_clusters = cluster_stats[cluster_stats[1] > 1].copy()
    return (
        sorted_descriptions,
        class_cluster_counts,
        class1_rate_by_cluster,
        cluster_stats,
        valuable_clusters,
    )


def _scale_clusters(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric columns and add an importance metric."""

    scaled_clusters = cluster_df.copy()
    numeric_cols = scaled_clusters.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    scaled_clusters[numeric_cols] = scaler.fit_transform(scaled_clusters[numeric_cols])
    scaled_clusters["importance"] = scaled_clusters.sum(axis=1)
    return scaled_clusters


def _compute_inflection_points(
    cluster_stats: pd.DataFrame,
    scaled_clusters: pd.DataFrame,
    inflex_pond_sup: float,
    inflex_pond_inf: float,
):
    """Mark clusters considered good based on inflection points."""

    first_inflection = first_decreasing_inflection_point(
        scaled_clusters[0], bins=20, window_length=5, polyorder=2
    )
    second_inflection = first_decreasing_inflection_point(
        scaled_clusters[1], bins=20, window_length=5, polyorder=2
    )
    good_mask = (cluster_stats[1] > 1) & (
        (cluster_stats[0] > (first_inflection * inflex_pond_sup))
        | (cluster_stats[1] > second_inflection)
        | (cluster_stats[0] < inflex_pond_inf)
    )
    cluster_stats = cluster_stats.copy()
    cluster_stats["good"] = np.where(good_mask, 1, 0)
    return cluster_stats, first_inflection, second_inflection


def _merge_outputs(
    df_descriptive: pd.DataFrame,
    class1_rate_by_cluster: pd.Series,
    cluster_stats: pd.DataFrame,
    var_rename: Dict[str, str],
):
    """Merge descriptive data with computed cluster statistics."""

    final_descriptions = df_descriptive.copy()
    final_descriptions, _ = replace_with_dict(
        final_descriptions, ["cluster_description"], var_rename
    )
    final_descriptions = final_descriptions.merge(
        class1_rate_by_cluster.reset_index(), on="cluster", how="left"
    )
    final_descriptions = final_descriptions.merge(
        cluster_stats.reset_index(), on="cluster", how="left"
    )
    final_descriptions = final_descriptions.rename(
        columns={"1_x": "Probability", "1_y": "N_probability", 0: "Support"}
    )
    return final_descriptions.drop(columns=["cluster_weight"])


def get_valuable_descriptions(
    df_descriptive,
    df_clustered,
    targets,
    var_rename,
    inflex_pond_sup=0.4,
    inflex_pond_inf=0.5,
):
    """Return cluster descriptions with a flag for relevant clusters."""

    (
        sorted_descriptions,
        stacked_data,
        class1_rate_by_cluster,
        cluster_stats,
        valuable_clusters,
    ) = _prepare_cluster_data(df_descriptive, df_clustered, targets)
    scaled_clusters = _scale_clusters(valuable_clusters)
    cluster_stats, _, _ = _compute_inflection_points(
        cluster_stats, scaled_clusters, inflex_pond_sup, inflex_pond_inf
    )
    final_df = _merge_outputs(
        sorted_descriptions, class1_rate_by_cluster, cluster_stats, var_rename
    )
    return final_df, stacked_data


def generate_descriptions(
    condition_list, language="en", OPENAI_API_KEY=None, default_params=None
):
    """Generate plain language descriptions for rule conditions.

    Parameters
    ----------
    condition_list : list[str]
        List of textual rule conditions to describe.
    language : str, default "en"
        Language in which the descriptions should be returned.
    OPENAI_API_KEY : str or None, optional
        API key used to initialize the OpenAI client. If ``None`` the
        environment variable ``OPENAI_API_KEY`` must be set.
    default_params : dict or None, optional
        Parameters forwarded to ``client.chat.completions.create`` when
        generating the text. If ``None`` sensible defaults are used.

    Returns
    -------
    dict
        Dictionary with a single key ``responses`` containing a list of the
        generated descriptions, one per condition in ``condition_list``.
    """

    client = get_openai_client(OPENAI_API_KEY)

    if client is None:
        logger.warning("OpenAI client unavailable; returning empty descriptions")
        return {"responses": []}

    if default_params is None:

        def get_default_params():
            return {
                "model": "gpt-4-turbo",
                "temperature": 0.5,
                "max_tokens": 1500,
                "n": 1,
                "stop": None,
            }

        default_params = get_default_params()

    # Create a single message with all conditions
    conditions_text = "\n".join(
        [f"{i+1}. {condition}" for i, condition in enumerate(condition_list)]
    )

    # Improved prompt for simple and clear descriptions
    system_prompt = "You are an assistant that helps to describe dataset groups in very simple terms."
    user_prompt = (
        f"Generate a very simple description for each of the following conditions. "
        f"Use everyday language. Avoid specific numbers and ranges; instead, "
        f"use general groups like 'elderly people', 'classic cars', etc."
        f"Make each description visually friendly highlight what makes that condition unique and using emojis. Structure: 'EMOJI': 'RESPONSE'"
        f"Only respond with the descriptions in {language}. Conditions:\n\n{conditions_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(messages=messages, **default_params)
    except Exception as exc:
        logger.exception("Error calling the OpenAI API: %s", exc)
        return {"responses": []}

    # Split the response into a list of descriptions per line
    descriptions = response.choices[0].message.content.strip().split("\n")
    descriptions = [desc.strip() for desc in descriptions if desc.strip()]

    # Return a dictionary with the responses
    result = {"responses": descriptions}
    return result


def _categorize_conditions(condition_list, df, n_groups=2, handle_bools=False):
    """Categorize numeric or boolean conditions into percentile labels.

    Parameters
    ----------
    condition_list : list[str]
        List of condition strings.
    df : pd.DataFrame
        DataFrame used to compute percentiles.
    n_groups : int, default 2
        Number of percentile groups.
    handle_bools : bool, default False
        Whether to handle boolean conditions.

    Returns
    -------
    dict
        Dictionary with a ``responses`` key mapping to a list of descriptions.
    """

    # -- Common validations --
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "A valid, non-empty pandas DataFrame is required."}
    if not isinstance(n_groups, int) or n_groups < 2:
        return {"error": "n_groups must be an integer greater than or equal to 2."}

    df.columns = df.columns.str.replace(" ", "_")

    # -- Percentile labels --
    def format_percentile(p):
        return (
            str(int(p)) if float(p).is_integer() else f"{p:.2f}".rstrip("0").rstrip(".")
        )

    labels = [
        f"Percentile {format_percentile((i + 1) * 100 / n_groups)}"
        for i in range(n_groups)
    ]

    # -- Common thresholds --
    bool_cols = (
        df.select_dtypes(include="bool").columns.tolist() if handle_bools else []
    )
    quantile_points = [i / n_groups for i in range(1, n_groups)]
    thresholds = {
        c: df[c].quantile(quantile_points).tolist()
        for c in df.columns
        if c not in bool_cols
    }

    # -- Improved patterns --------------------------------------------------
    # 1) Column name: letters, numbers, _, -, ., (), [] …
    var_name = r"[A-Za-z0-9_\-\.\(\)\[\]]+"

    # 2) Number:
    #    - Optional integer part
    #    - Optional decimal point
    #    - Optional scientific notation (e or E, with optional sign)
    #    - Optional leading ± sign
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    pattern_num = rf"({num})\s*<=\s*({var_name})\s*<=\s*({num})"
    pattern_bool = rf"({var_name})\s*==\s*(True|False)"

    # -- Common processing --
    descriptions = []
    for condition in condition_list:
        tokens = [t.strip() for t in re.split(r"(?i)\band\b", condition)]

        parts = []
        for token in tokens:
            m_num = re.match(pattern_num, token)
            if m_num:
                min_v, feat, max_v = m_num.groups()
                avg_value = (float(min_v) + float(max_v)) / 2
                if feat in thresholds:
                    cuts = thresholds[feat]
                    # In case multiple percentiles produce the same cut value,
                    # ``side='right'`` ensures we keep the higher percentile label
                    category = labels[np.searchsorted(cuts, avg_value, side="right")]
                elif handle_bools and feat in bool_cols:
                    category = "TRUE" if avg_value >= 0.5 else "FALSE"
                else:
                    category = "N/A"
                parts.append(f"{feat} = {category}")
                continue

            if handle_bools:
                m_bool = re.match(pattern_bool, token)
                if m_bool:
                    feat, val = m_bool.groups()
                    if feat in bool_cols:
                        category = "TRUE" if val == "True" else "FALSE"
                    else:
                        category = "N/A"
                    parts.append(f"{feat} = {category}")

        descriptions.append(", ".join(parts) + ".")

    return {"responses": descriptions}


def categorize_conditions(condition_list, df, n_groups=2):
    """Categorize conditions, including booleans, into percentile labels.

    This is a thin wrapper around :func:`_categorize_conditions` with
    ``handle_bools`` enabled for backwards compatibility.

    Parameters
    ----------
    condition_list : list[str]
        Conditions expressed as strings (e.g. ``"0 <= age <= 10"``).
    df : pd.DataFrame
        DataFrame used to compute percentile thresholds.
    n_groups : int, default 2
        Number of percentile groups to split numeric variables into.

    Returns
    -------
    dict
        Mapping with a ``responses`` key containing the categorized
        descriptions.
    """
    return _categorize_conditions(
        condition_list, df, n_groups=n_groups, handle_bools=True
    )


def categorize_conditions_generalized(condition_list, df, n_groups=2):
    """Categorize conditions handling numeric and boolean features.

    Parameters
    ----------
    condition_list : list[str]
        Conditions expressed as strings.
    df : pd.DataFrame
        DataFrame used to compute percentile thresholds.
    n_groups : int, default 2
        Number of percentile groups to split numeric variables into.

    Returns
    -------
    dict
        Mapping with a ``responses`` key containing the categorized
        descriptions.
    """
    return _categorize_conditions(
        condition_list, df, n_groups=n_groups, handle_bools=True
    )


def build_conditions_table(
    condition_list,
    df,
    effectiveness,
    weights=None,
    n_groups=3,
    fill_value="N/A",
):
    """Build a table with categorical descriptions and metrics.

    Parameters
    ----------
    condition_list : list[str]
        List of conditions in the form ``"min <= VAR <= max …"``.
    df : pd.DataFrame
        Reference DataFrame used to compute quantiles.
    effectiveness : list[float] | pd.Series
        Effectiveness metric for each condition.
    weights : list[float] | pd.Series | None, default None
        Optional metric (support, frequency, etc.) for each condition.
    n_groups : int, default 3
        Number of groups for ``categorize_conditions``.
    fill_value : str, default "N/A"
        Value for variables missing in a condition description.

    Returns
    -------
    pd.DataFrame
        Resulting table with columns ``Group``, ``Effectiveness``, ``Weight``
        and the extracted variables in alphabetical order.
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
    if len(effectiveness) != n:
        raise ValueError(
            "`effectiveness` must have the same length as `condition_list`"
        )
    if weights is None:
        weights = [np.nan] * n
    elif len(weights) != n:
        raise ValueError("`weights` must have the same length as `condition_list`")

    for idx, desc in enumerate(descriptions):
        row = {
            "Group": idx + 1,
            "Effectiveness": effectiveness[idx],
            "Weight": weights[idx],
        }
        row.update({var: fill_value for var in variables})

        for var, level in re.findall(var_pattern, desc):
            row[var] = level.strip()

        rows.append(row)

    final_cols = ["Group", "Effectiveness", "Weight"] + variables
    return pd.DataFrame(rows, columns=final_cols)


def _parse_relative_description(desc: Union[str, float, None]) -> Dict[str, str]:
    """
    Convert a string like
        'petal_width_(cm) = Percentile 80, sepal_length_(cm) = Percentile 40.'
    into a dictionary
        {'petal_width_(cm)': 'PERCENTILE 80',
         'sepal_length_(cm)': 'PERCENTILE 40'}

    If the input is not valid text (NaN, None, ''), return {}.
    """
    if not isinstance(desc, str) or not desc.strip():
        return {}

    # Remove final period (if present) and split by commas.
    tokens: List[str] = [
        t.strip().rstrip(".")  # “petal = Percentile 25”
        for t in desc.split(",")
        if t.strip()
    ]

    pares: Dict[str, str] = {}
    for token in tokens:
        if " = " in token:  # Guaranteed by format
            var, cat = token.split(" = ", 1)
            pares[var.strip()] = cat.strip().upper()  # Normalize to UPPERCASE
    return pares


def expand_categories(
    df: pd.DataFrame, desc_col: str = "cluster_desc_relative", inplace: bool = False
) -> pd.DataFrame:
    """
    From *df[desc_col]* create a new column for each variable mentioned in
    **cluster_desc_relative** and place its category (Percentile 10, Percentile 25…).
    Where the variable is not mentioned, leave `NaN`.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    desc_col : str, default 'cluster_desc_relative'
        Name of the column containing the descriptions.
    inplace : bool, default False
        - True: add columns directly to `df` and return it.
        - False: return **a copy** with the new columns, leaving `df` untouched.

    Returns
    -------
    pd.DataFrame
        DataFrame with the additional category columns.
    """
    if desc_col not in df.columns:
        raise KeyError(f'Column "{desc_col}" does not exist in the DataFrame.')

    # 1) Transform each description into a variable → category dict.
    mapeos: pd.Series = df[desc_col].apply(_parse_relative_description)

    # 2) Convert the series of dicts into a columnar DataFrame (wide).
    cat_df: pd.DataFrame = pd.json_normalize(mapeos)
    # Missing columns in a row automatically become NaN.

    # 3) Join back to the original.
    if inplace:
        for col in cat_df.columns:
            df[col] = cat_df[col]
        return df
    else:
        return pd.concat([df.copy(), cat_df], axis=1)


# ----------------------------- utilities -----------------------------


def feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature columns (everything to the right of 'cluster_desc_relative')."""
    idx = df.columns.get_loc("cluster_desc_relative")
    return list(df.columns[idx + 1 :])


def encode_features(
    df: pd.DataFrame, ord_map: Dict[str, int], *, scale: bool = True
) -> pd.DataFrame:
    """
    Convert ordinal variables using ``ord_map`` and optionally scale each column to [0, 1].

    Parameters
    ----------
    df : Source DataFrame
    ord_map : dict mapping text to ordinal value
    scale : bool, if True, normalize non-constant columns

    Returns
    -------
    DataFrame with transformed columns
    """
    feats = feature_cols(df)
    enc = df[feats].copy()

    for c in feats:
        # pandas 3 uses StringDtype by default, while older releases used
        # object. Both represent the ordinal labels consumed by ord_map.
        if pd.api.types.is_object_dtype(enc[c].dtype) or pd.api.types.is_string_dtype(
            enc[c].dtype
        ):
            enc[c] = enc[c].map(ord_map)

        # rescale if it has more than one distinct value
        if scale and enc[c].dropna().nunique() > 1:
            enc[c] = (enc[c] - enc[c].min()) / (enc[c].max() - enc[c].min())

    return enc


def similarity(
    a_idx: int, b_idx: int, df_enc: pd.DataFrame, *, cov_weight: bool = True
) -> float:
    """Similarity (1 - mean absolute distance) between two encoded rows."""
    v_a, v_b = df_enc.iloc[a_idx], df_enc.iloc[b_idx]
    mask = ~(v_a.isna() | v_b.isna())
    if mask.sum() == 0:
        return 0.0
    d = np.abs(v_a[mask] - v_b[mask]).mean()
    sim = 1 - d
    return sim * (mask.sum() / df_enc.shape[1]) if cov_weight else sim


# ----------------------------- high-level API -----------------------------


def similarity_matrix(
    df: pd.DataFrame, ord_map: Dict[str, int], *, cov_weight: bool = True
) -> pd.DataFrame:
    """
    Similarity matrix ``S`` (diagonal = 1).

    Parameters
    ----------
    df         : DataFrame with clusters and their features
    ord_map    : dict mapping used in ``encode_features``
    cov_weight : bool weights similarity by coverage of non-null data
    """
    df_enc = encode_features(df.reset_index(drop=True), ord_map)
    n = len(df_enc)
    clusters = df["cluster"].tolist()
    S = pd.DataFrame(np.eye(n), index=clusters, columns=clusters)

    for i, j in combinations(range(n), 2):
        s = similarity(i, j, df_enc, cov_weight=cov_weight)
        S.iat[i, j] = S.iat[j, i] = round(s, 3)
    return S


def cluster_pairs_sim(
    df: pd.DataFrame,
    ord_map: Dict[str, int],
    *,
    metric: str = "cluster_ef_sample",
    cov_weight: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      cluster_1 | cluster_2 | similarity | delta_<metric> | score

    The score prioritizes pairs with high similarity and positive improvement
    in the specified metric.
    """
    df_r = df.reset_index(drop=True)
    df_enc = encode_features(df_r, ord_map)
    n = len(df_r)

    # -- collect pairs --
    pairs: List[Tuple[str, str, float, float]] = []
    for i, j in combinations(range(n), 2):
        sim = similarity(i, j, df_enc, cov_weight=cov_weight)
        delta = abs(df_r.at[j, metric] - df_r.at[i, metric])
        pairs.append((df_r.at[i, "cluster"], df_r.at[j, "cluster"], sim, delta))

    # -- robust scale for positive deltas --
    pos_deltas = np.array([d for *_, d in pairs if d > 0])
    mad = np.median(np.abs(pos_deltas - np.median(pos_deltas))) + 1e-9
    sigma_rob = 1.4826 * mad if pos_deltas.size else 1.0

    # -- compute score --
    rows = []
    for c1, c2, sim, delta in pairs:
        grow = 1 - np.exp(-max(0, delta) / sigma_rob)
        score = sim * grow
        rows.append([c1, c2, round(sim, 3), round(delta, 3), round(score, 3)])

    return pd.DataFrame(
        rows,
        columns=["cluster_1", "cluster_2", "similarity", f"delta_{metric}", "score"],
    )


def get_frontiers(
    df_descriptive: pd.DataFrame, df: pd.DataFrame, divide: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate cluster frontiers from descriptions.

    Parameters
    ----------
    df_descriptive : pd.DataFrame
        DataFrame with cluster descriptions.
    df : pd.DataFrame
        Reference DataFrame to compute percentiles.
    divide : int, default 5
        Number of divisions for ``categorize_conditions``.

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame)
        ``df_explain`` with expanded columns and ``frontiers`` with
        cluster pairs ordered by score.
    """
    general_descriptions = [
        x
        for x in df_descriptive["cluster_description"].unique().tolist()
        if isinstance(x, str)
    ]

    desc = categorize_conditions(general_descriptions, df, divide)
    df_descriptive["cluster_desc_relative"] = df_descriptive[
        "cluster_description"
    ].replace({k: v for k, v in zip(general_descriptions, desc["responses"])})

    df_explain = expand_categories(df_descriptive)
    df_explain = df_explain[df_explain["cluster_n_sample"] > 0]

    ORD_MAP_LOCAL = {f"PERCENTILE {i}": i for i in range(1, 101)}
    df_explain_gen = df_explain.sort_values(
        "cluster_n_sample", ascending=False
    ).head(40)
    similarity_matrix(df_explain_gen, ORD_MAP_LOCAL)
    frontiers = cluster_pairs_sim(
        df_explain_gen, ORD_MAP_LOCAL, metric="cluster_ef_sample"
    )
    frontiers.sort_values("score", ascending=False, inplace=True)
    return df_explain, frontiers


# ╭──────────────────╮
# │  INIT OpenAI SDK │
# ╰──────────────────╯
# The following utilities generate hypotheses and translate
# rules into readable text.


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
        "is": {"es": "Es {label}", "en": "Is {label}"},
        "not": {"es": "No es {label}", "en": "Is not {label}"},
        "generic": {"es": "Condición sobre {label}", "en": "Condition on {label}"},
    }
    return {k: v[lang] for k, v in base.items()}


def get_FALLBACK_LABELS(df_meta_sub):
    """Build a mapping of ``(rule_token, lang)`` to label strings.

    The function scans all ``description`` columns in ``df_meta_sub`` to create
    a dictionary usable as a fallback when a specific translation is missing.

    Parameters
    ----------
    df_meta_sub : pd.DataFrame
        Metadata slice containing ``rule_token`` and language-specific
        description columns, e.g. ``identity.description_i18n.es``.

    Returns
    -------
    dict
        Dictionary keyed by ``(rule_token, language)`` returning the label
        string for that variable and language.
    """

    idiomas_d = [
        y.split(".")[-1] for y in [x for x in df_meta_sub.columns if "description" in x]
    ]
    df_lang = {}
    for y in idiomas_d:
        df_by_lang = []
        for x in df_meta_sub.columns:
            if x.endswith("." + y):
                df_by_lang.append(x)

        df_lang[y] = df_meta_sub[["rule_token"] + df_by_lang].drop_duplicates()

    dicts_rts = [
        {(x, ess_): z for x, y, z in df_lang[ess_].values} for ess_ in df_lang.keys()
    ]

    def concatenate_dictionaries(list_of_dicts):
        result_dict = {}
        for d in list_of_dicts:
            result_dict.update(d)
        return result_dict

    return concatenate_dictionaries(dicts_rts)


def _extract_tokens(series: pd.Series) -> Set[str]:
    token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    return {m.group(0) for txt in series.astype(str) for m in token_re.finditer(txt)}


def _meta_lookup(
    token: str, meta: pd.DataFrame, *, lang: str
) -> Tuple[str, str, str, str | None]:
    col_lbl = f"identity.label_i18n.{lang}"
    col_des = f"identity.description_i18n.{lang}"
    row = meta.loc[meta["rule_token"] == token]
    _FALLBACK_LABELS = get_FALLBACK_LABELS(meta)

    if not row.empty:
        r = row.iloc[0]
        return (
            str(
                r.get(col_lbl) or _FALLBACK_LABELS.get((token, lang), token)
            ).capitalize(),
            str(r.get(col_des) or ""),
            str(r.get("domain.categorical.codes") or ""),
            str(r.get("actionability.side_effects") or "") or None,
        )
    return (_FALLBACK_LABELS.get((token, lang), token).capitalize(), "", "", None)


def _rule_to_text(rule: str, meta_df: pd.DataFrame, *, lang: str) -> str:
    toks = _extract_tokens(pd.Series([rule]))
    tok = next(iter(toks)) if toks else None

    m = get_range_re().search(rule)
    low = high = None
    if m:
        low, high, tok = (
            float(m["low"].replace("_", "")),
            float(m["high"].replace("_", "")),
            m["tok"],
        )

    if tok is None:
        return rule

    label, *_ = _meta_lookup(tok, meta_df, lang=lang)
    tpl = _tpl_rules(lang)
    if tok.startswith("cat__"):
        if low is not None and high is not None:
            if high <= 0.5:
                return tpl["not"].format(label=label)
            if low >= 0.5:
                return tpl["is"].format(label=label)
        return tpl["generic"].format(label=label)

    if low is not None and high is not None:
        return tpl["between"].format(label=label, low=low, high=high)
    return tpl["generic"].format(label=label)


def _list_rules_to_text(col, meta_df, *, lang: str, placeholder: str = "—") -> str:
    """
    Convert a list or string-list into readable text.
    Return 'placeholder' if the list is empty.
    """
    # 1) Normalize to list
    if col is None or (isinstance(col, float) and pd.isna(col)):
        col = []
    elif isinstance(col, str):
        try:
            col = ast.literal_eval(col)
        except Exception:
            col = [col]
    if not isinstance(col, (list, tuple)):
        col = [col]

    # 2) No rules → placeholder
    if len(col) == 0 or all(str(r).strip() == "" for r in col):
        return placeholder

    # 3) Standard translation
    return ", ".join(_rule_to_text(r, meta_df, lang=lang) for r in col)


def _local_significance(delta: float, *, lang: str) -> str:
    if lang == "es":
        return "significativo" if abs(delta) > 0.1 else "exploratorio"
    return "significant" if abs(delta) > 0.1 else "exploratory"


def _local_hypothesis_text(
    inter_txt: str,
    a_txt: str,
    b_txt: str,
    p_a: float,
    p_b: float,
    delta: float,
    target_lbl: str,
    side: list[str],
    *,
    lang: str,
) -> str:
    """Generate localized hypothesis text (A vs B with a common intersection)."""
    sig_word = _local_significance(delta, lang=lang)

    side_txt = ""
    if side:
        se = "\n- " + "\n- ".join(sorted(set(side)))
        side_txt = (
            "\n\n**Posibles efectos secundarios**"
            if lang == "es"
            else "\n\n**Possible side-effects**"
        ) + se

    # Headers
    header_ctx = (
        "**Reglas compartidas (intersección)**"
        if lang == "es"
        else "**Shared rules (intersection)**"
    )
    header_a = "**Subgrupo A**" if lang == "es" else "**Subgroup A**"
    header_b = "**Subgrupo B**" if lang == "es" else "**Subgroup B**"
    header_an = (
        "**Análisis y recomendaciones**"
        if lang == "es"
        else "**Analysis & Recommendations**"
    )

    # Analysis body
    analysis = (
        f"Resultado {sig_word}. Ajustar las variables que diferencian A y B "
        f"podría aumentar la probabilidad de {target_lbl.lower()}."
        if lang == "es"
        else f"{sig_word.capitalize()} result. Adjusting features that differentiate "
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


def _call_openai_text(
    payload: dict[str, Any],
    *,
    system_prompt: str,
    model: str,
    temperature: float,
    api_key: Optional[str] = None,
    client: Any | None = None,
) -> Optional[str]:
    """Return one OpenAI response, or ``None`` so callers can fall back."""
    if client is None:
        client = get_openai_client(api_key)
    if client is None:
        logger.warning("OpenAI client unavailable; using the local report")
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        choices = getattr(response, "choices", None)
        content = (
            getattr(getattr(choices[0], "message", None), "content", None)
            if choices
            else None
        )
        if not content:
            logger.warning("OpenAI returned no text; using the local report")
            return None
        return str(content).strip()
    except Exception as exc:
        logger.warning("OpenAI request failed; using the local report: %s", exc)
        return None


def _model_hypothesis_task(regions: pd.DataFrame) -> str:
    """Infer an InsideForest task from its public region schema."""
    columns = set(regions.columns)
    if "target_mean" in columns and (
        "mean_shift" in columns or "dispersion_reduction" in columns
    ):
        return "regression"
    if "class_distribution" in columns and (
        "region_target_class" in columns
        or "target_class" in columns
        or "dominant_class" in columns
    ):
        return "multiclass"
    if "target_distribution" in columns and "dominant_target" in columns:
        return "classification"
    raise ValueError(
        "Could not infer the task from explain_regions(); expected an "
        "InsideForest traditional, multiclass, or regression schema"
    )


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _class_distribution(value: Any, classes: Any = None) -> dict[str, float]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)
        labels = list(classes) if classes is not None else list(range(len(values)))
        items = zip(labels, values)
    else:
        return {}
    result = {}
    for label, raw_value in items:
        number = _finite_float(raw_value)
        if number is not None:
            result[str(label)] = number
    return result


def _region_id_column(regions: pd.DataFrame) -> str:
    for column in ("region_id", "cluster_id"):
        if column in regions.columns:
            return column
    raise ValueError("explain_regions() must expose region_id or cluster_id")


def _dominant_class(row: pd.Series, task: str) -> Any:
    columns = (
        ("dominant_target", "region_target_class")
        if task == "classification"
        else ("dominant_class", "target_class", "region_target_class")
    )
    for column in columns:
        value = _json_value(row.get(column))
        if value is not None:
            return value
    return None


def _row_score(row: pd.Series) -> float:
    for column in ("region_score", "score", "weight", "support"):
        value = _finite_float(row.get(column))
        if value is not None:
            return value
    return 0.0


def _select_model_regions(
    regions: pd.DataFrame, task: str, region_ids: Optional[Any]
) -> tuple[pd.Series, pd.Series, str]:
    id_column = _region_id_column(regions)
    if region_ids is not None:
        requested = list(region_ids)
        if len(requested) != 2 or requested[0] == requested[1]:
            raise ValueError("region_ids must contain exactly two distinct IDs")
        selected = []
        for region_id in requested:
            match = regions[regions[id_column] == region_id]
            if match.empty:
                raise ValueError(f"Unknown region ID: {region_id!r}")
            selected.append(match.iloc[0])
        return selected[0], selected[1], id_column

    if task == "regression":
        means = pd.to_numeric(regions["target_mean"], errors="coerce")
        valid = regions.loc[means.notna()].copy()
        valid["__mean"] = means[means.notna()]
        if len(valid) < 2:
            raise ValueError("At least two regions with numeric target_mean are required")
        ordered = valid.sort_values("__mean", kind="stable")
        return ordered.iloc[0], ordered.iloc[-1], id_column

    order = sorted(
        range(len(regions)),
        key=lambda index: _row_score(regions.iloc[index]),
        reverse=True,
    )
    first = regions.iloc[order[0]]
    first_class = _dominant_class(first, task)
    for index in order[1:]:
        candidate = regions.iloc[index]
        if _dominant_class(candidate, task) != first_class:
            return first, candidate, id_column
    return first, regions.iloc[order[1]], id_column


def _normalized_model_region(
    row: pd.Series, *, task: str, id_column: str, classes: Any = None
) -> dict[str, Any]:
    region = {
        "region_id": _json_value(row.get(id_column)),
        "description": str(row.get("description", "—")),
        "support": _json_value(row.get("support")),
        "coverage": _json_value(row.get("coverage")),
        "region_score": _json_value(
            row.get("region_score", row.get("score", row.get("weight")))
        ),
    }
    if task == "classification":
        region.update(
            dominant_class=_dominant_class(row, task),
            dominant_probability=_json_value(row.get("dominant_probability")),
            class_distribution=_class_distribution(
                row.get("target_distribution"), classes
            ),
            lift=_json_value(row.get("lift")),
        )
    elif task == "multiclass":
        region.update(
            dominant_class=_dominant_class(row, task),
            target_probability=_json_value(
                row.get("target_probability", row.get("dominant_probability"))
            ),
            class_distribution=_class_distribution(
                row.get("class_distribution"), classes
            ),
            class_margin=_json_value(row.get("class_margin")),
            lift=_json_value(row.get("lift")),
            entropy=_json_value(row.get("entropy")),
        )
    else:
        for column in (
            "target_mean",
            "target_median",
            "target_std",
            "target_iqr",
            "mean_shift",
            "standardized_mean_shift",
            "dispersion_reduction",
        ):
            region[column] = _json_value(row.get(column))
    return region


def _number_text(value: Any) -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{number:,.3f}"


def _probability_text(value: Any) -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{number:.2%}"


def _distribution_text(distribution: dict[str, float]) -> str:
    if not distribution:
        return "—"
    return ", ".join(
        f"{label}: {_probability_text(value)}"
        for label, value in distribution.items()
    )


def _local_model_hypothesis(payload: dict[str, Any]) -> str:
    task, lang = payload["task"], payload["lang"]
    target = payload["target"]
    a, b = payload["regions"]
    if lang == "es" and task == "regression":
        difference = (_finite_float(b["target_mean"]) or 0.0) - (
            _finite_float(a["target_mean"]) or 0.0
        )
        return (
            f"# Comparación de regiones — regresión ({target})\n\n"
            f"## Región {a['region_id']}\n- Reglas: **{a['description']}**\n"
            f"- Media del objetivo: **{_number_text(a['target_mean'])}**\n"
            f"- Mediana: **{_number_text(a['target_median'])}**\n"
            f"- Desviación estándar: **{_number_text(a['target_std'])}**\n"
            f"- Desplazamiento de la media: **{_number_text(a['mean_shift'])}**\n\n"
            f"## Región {b['region_id']}\n- Reglas: **{b['description']}**\n"
            f"- Media del objetivo: **{_number_text(b['target_mean'])}**\n"
            f"- Mediana: **{_number_text(b['target_median'])}**\n"
            f"- Desviación estándar: **{_number_text(b['target_std'])}**\n"
            f"- Reducción de dispersión: **{_number_text(b['dispersion_reduction'])}**\n\n"
            f"## Hipótesis\nLa región {b['region_id']} presenta una media "
            f"**{difference:+,.3f}** mayor que la región {a['region_id']}. "
            "Conviene validar las reglas diferenciales fuera de muestra."
        )
    if lang == "es" and task == "multiclass":
        return (
            f"# Comparación de regiones — multiclase ({target})\n\n"
            f"## Región {a['region_id']}\n- Reglas: **{a['description']}**\n"
            f"- Clase dominante: **{a['dominant_class']}**\n"
            f"- Distribución: **{_distribution_text(a['class_distribution'])}**\n"
            f"- Margen / lift / entropía: **{_number_text(a['class_margin'])} / "
            f"{_number_text(a['lift'])} / {_number_text(a['entropy'])}**\n\n"
            f"## Región {b['region_id']}\n- Reglas: **{b['description']}**\n"
            f"- Clase dominante: **{b['dominant_class']}**\n"
            f"- Distribución: **{_distribution_text(b['class_distribution'])}**\n"
            f"- Margen / lift / entropía: **{_number_text(b['class_margin'])} / "
            f"{_number_text(b['lift'])} / {_number_text(b['entropy'])}**\n\n"
            "## Hipótesis\nLas reglas separan clases dominantes distintas. Un margen "
            "bajo o una entropía alta señalan mayor ambigüedad."
        )
    if lang == "es":
        return (
            f"# Comparación de regiones — clasificación ({target})\n\n"
            f"## Región {a['region_id']}\n- Reglas: **{a['description']}**\n"
            f"- Clase dominante: **{a['dominant_class']}**\n"
            f"- Probabilidad dominante: **{_probability_text(a['dominant_probability'])}**\n"
            f"- Distribución: **{_distribution_text(a['class_distribution'])}**\n\n"
            f"## Región {b['region_id']}\n- Reglas: **{b['description']}**\n"
            f"- Clase dominante: **{b['dominant_class']}**\n"
            f"- Probabilidad dominante: **{_probability_text(b['dominant_probability'])}**\n"
            f"- Distribución: **{_distribution_text(b['class_distribution'])}**\n\n"
            "## Hipótesis\nLas condiciones diferenciales concentran clases distintas; "
            "la probabilidad dominante estima la pureza de cada región."
        )
    if task == "regression":
        difference = (_finite_float(b["target_mean"]) or 0.0) - (
            _finite_float(a["target_mean"]) or 0.0
        )
        return (
            f"# Region comparison — regression ({target})\n\n"
            f"## Region {a['region_id']}\n- Rules: **{a['description']}**\n"
            f"- Target mean: **{_number_text(a['target_mean'])}**\n"
            f"- Median: **{_number_text(a['target_median'])}**\n"
            f"- Standard deviation: **{_number_text(a['target_std'])}**\n\n"
            f"## Region {b['region_id']}\n- Rules: **{b['description']}**\n"
            f"- Target mean: **{_number_text(b['target_mean'])}**\n"
            f"- Median: **{_number_text(b['target_median'])}**\n"
            f"- Standard deviation: **{_number_text(b['target_std'])}**\n\n"
            f"## Hypothesis\nRegion {b['region_id']} has a **{difference:+,.3f}** "
            f"higher mean than region {a['region_id']}; validate it out of sample."
        )
    noun = "multiclass" if task == "multiclass" else "classification"
    probability_a = (
        _distribution_text(a["class_distribution"])
        if task == "multiclass"
        else _probability_text(a["dominant_probability"])
    )
    probability_b = (
        _distribution_text(b["class_distribution"])
        if task == "multiclass"
        else _probability_text(b["dominant_probability"])
    )
    return (
        f"# Region comparison — {noun} ({target})\n\n"
        f"## Region {a['region_id']}\n- Rules: **{a['description']}**\n"
        f"- Dominant class: **{a['dominant_class']}**\n- Profile: **{probability_a}**\n\n"
        f"## Region {b['region_id']}\n- Rules: **{b['description']}**\n"
        f"- Dominant class: **{b['dominant_class']}**\n- Profile: **{probability_b}**\n\n"
        "## Hypothesis\nThe differing rules concentrate different target profiles."
    )


def generate_model_hypothesis(
    estimator: Any,
    *,
    target: Optional[str] = None,
    meta_df: Optional[pd.DataFrame] = None,
    lang: str = "es",
    use_gpt: bool = False,
    openai_model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    region_ids: Optional[Any] = None,
    top_n: int = 50,
    api_key: Optional[str] = None,
    client: Any | None = None,
) -> str:
    """Compare regions from any fitted canonical InsideForest estimator."""
    if not hasattr(estimator, "explain_regions"):
        raise TypeError("estimator must expose an explain_regions() method")
    if lang not in {"es", "en"}:
        raise ValueError("lang must be 'es' or 'en'")
    try:
        regions = estimator.explain_regions(
            top_n=None if region_ids is not None else top_n
        )
    except Exception as exc:
        raise ValueError(
            "The estimator must be fitted before generating a hypothesis"
        ) from exc
    if not isinstance(regions, pd.DataFrame):
        regions = pd.DataFrame(regions)
    if len(regions) < 2:
        raise ValueError("At least two explained regions are required")

    task = _model_hypothesis_task(regions)
    row_a, row_b, id_column = _select_model_regions(regions, task, region_ids)
    target_label = target or ("objetivo" if lang == "es" else "target")
    if target is not None and meta_df is not None:
        try:
            target_label = _meta_lookup(target, meta_df, lang=lang)[0]
        except Exception:
            target_label = target
    payload = {
        "lang": lang,
        "task": task,
        "target": target_label,
        "regions": [
            _normalized_model_region(
                row_a,
                task=task,
                id_column=id_column,
                classes=getattr(estimator, "classes_", None),
            ),
            _normalized_model_region(
                row_b,
                task=task,
                id_column=id_column,
                classes=getattr(estimator, "classes_", None),
            ),
        ],
    }
    local_report = _local_model_hypothesis(payload)
    if not use_gpt:
        return local_report
    generated = _call_openai_text(
        payload,
        system_prompt=(
            "You are a data-science expert. Return one concise Markdown report "
            "in the requested language. Compare exactly the supplied regions "
            "without inventing metrics. Adapt terminology to the task: class "
            "probabilities for classification; distributions, margin, lift and "
            "entropy for multiclass; numeric means, medians and dispersion for "
            "regression. Never format continuous target values as percentages. "
            "End with a testable, non-causal hypothesis."
        ),
        model=openai_model,
        temperature=temperature,
        api_key=api_key,
        client=client,
    )
    return generated or local_report


def _gpt_hypothesis(
    payload: dict[str, Any], *, model: str, temperature: float, client: Any | None = None
) -> Optional[str]:
    """Wrapper: send payload to GPT and return the structured report."""
    if client is None:
        client = get_openai_client()
    if client is None:
        return None
    try:
        rsp = client.chat.completions.create(
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
                    ),
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


def generate_hypothesis(
    meta_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    *,
    target: str,
    lang: str = "es",
    use_gpt: bool = False,
    gpt_model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    client: Any | None = None,
) -> str:
    """Create a hypothesis report comparing two subgroups."""
    row = exp_df.iloc[0]
    row_ci = {k.lower(): v for k, v in row.items()}

    def _get(key: str, default: Any | None = None):
        return row_ci.get(key.lower(), default)

    # Possible side effects
    side = []
    for col in ("only_cluster_a", "only_cluster_b"):
        for tok in _extract_tokens(pd.Series([_get(col, "")])):
            se = _meta_lookup(tok, meta_df, lang=lang)[3]
            if se:
                side.append(se)

    # ===== GPT PATH =====
    if use_gpt:
        resolved_client = client or get_openai_client(api_key)
        if resolved_client is not None:
            payload = {
                "lang": lang,
                "target": _meta_lookup(target, meta_df, lang=lang)[0],
                "target_description": _meta_lookup(target, meta_df, lang=lang)[1],
                "shared_rules": _get("intersection", ""),
                "subgroup_a": _get("only_cluster_a", ""),
                "subgroup_b": _get("only_cluster_b", ""),
                "metrics": {
                    "p_a": _get("cluster_ef_a"),
                    "p_b": _get("cluster_ef_b"),
                    "delta": _get("delta_ef"),
                },
                "tokens_info": {
                    t: {
                        "label": _meta_lookup(t, meta_df, lang=lang)[0],
                        "description": _meta_lookup(t, meta_df, lang=lang)[1],
                        "domain": _meta_lookup(t, meta_df, lang=lang)[2],
                        "side_effect": _meta_lookup(t, meta_df, lang=lang)[3],
                    }
                    for t in (
                        _extract_tokens(pd.Series([_get("intersection", "")]))
                        | _extract_tokens(pd.Series([_get("only_cluster_a", "")]))
                        | _extract_tokens(pd.Series([_get("only_cluster_b", "")]))
                    )
                },
            }

            txt = _call_openai_text(
                payload,
                system_prompt=(
                    "You are a data-science expert assistant. Return one concise "
                    "Markdown report in the language specified by 'lang'. Compare "
                    "the two subgroups using only the supplied rules and metrics, "
                    "then propose a testable A/B hypothesis."
                ),
                model=gpt_model,
                temperature=temperature,
                client=resolved_client,
            )
            if txt:
                return txt
        else:
            logger.warning("OpenAI client unavailable; using the local report")

    # ===== LOCAL PATH =====
    inter_txt = _list_rules_to_text(_get("intersection", ""), meta_df, lang=lang)
    a_txt = _list_rules_to_text(_get("only_cluster_a", ""), meta_df, lang=lang)
    b_txt = _list_rules_to_text(_get("only_cluster_b", ""), meta_df, lang=lang)
    target_lbl = _meta_lookup(target, meta_df, lang=lang)[0]

    return _local_hypothesis_text(
        inter_txt,
        a_txt,
        b_txt,
        _get("cluster_ef_a"),
        _get("cluster_ef_b"),
        _get("delta_ef"),
        target_lbl,
        side,
        lang=lang,
    )
