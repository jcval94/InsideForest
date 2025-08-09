from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances

import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import ast

from matplotlib.patches import Rectangle
from collections import Counter, defaultdict
from typing import Sequence, Mapping, Any, List
from itertools import combinations
import random, math

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import logging

logger = logging.getLogger(__name__)


def _clean_bounds(lows, highs, nan_policy: str = "raise"):
    """Ensure ``lows <= highs`` and handle NaNs according to ``nan_policy``.

    Parameters
    ----------
    lows, highs : np.ndarray
        Arrays with lower and upper bounds.
    nan_policy : {"raise", "zero"}, default "raise"
        How to handle NaN values in bounds.
    """

    if nan_policy == "raise" and (np.isnan(lows).any() or np.isnan(highs).any()):
        raise ValueError("NaNs in bounds")
    if nan_policy == "zero":
        lows, highs = np.nan_to_num(lows), np.nan_to_num(highs)
    return np.minimum(lows, highs), np.maximum(lows, highs)


def choose_block(n: int, d: int, target_mb: int = 512) -> int:
    """Heuristic to pick a block size that keeps memory under ``target_mb``."""

    bytes_per_pair = 8 * (3 * d + 4)
    pairs = int((target_mb * 1024 * 1024) / bytes_per_pair)
    return max(64, min(n, int(np.sqrt(max(1, pairs)))))


def pairwise_iou_blocked(lows: np.ndarray, highs: np.ndarray, block: int = 1024) -> np.ndarray:
    """Compute pairwise IoU distances between hypercubes in blocks."""

    n, d = lows.shape
    vol = np.prod(highs - lows, axis=1)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(0, n, block):
        i2 = min(i + block, n)
        Li, Hi, Vi = lows[i:i2], highs[i:i2], vol[i:i2]
        for j in range(i, n, block):
            j2 = min(j + block, n)
            Lj, Hj, Vj = lows[j:j2], highs[j:j2], vol[j:j2]
            inter_low = np.maximum(Li[:, None, :], Lj[None, :, :])
            inter_high = np.minimum(Hi[:, None, :], Hj[None, :, :])
            inter_dims = np.clip(inter_high - inter_low, 0, None)
            inter_vol = inter_dims.prod(axis=2)
            union = Vi[:, None] + Vj[None, :] - inter_vol
            deg = union == 0
            same = deg & (
                np.all(Li[:, None, :] == Lj[None, :, :], axis=2)
                & np.all(Hi[:, None, :] == Hj[None, :, :], axis=2)
            )
            iou = np.where(
                same,
                1.0,
                np.divide(inter_vol, union, out=np.zeros_like(inter_vol), where=union > 0),
            )
            block_dist = 1.0 - iou
            dist[i:i2, j:j2] = block_dist
            if i != j:
                dist[j:j2, i:i2] = block_dist.T
    np.fill_diagonal(dist, 0.0)
    return dist


class Regions:
    
  def search_original_tree(self, df_clusterizado, separacion_dim):
    """Return the original separation matching the clustered DataFrame.

    Parameters
    ----------
    df_clusterizado : pd.DataFrame
        Resulting clustered DataFrame with ``linf`` and ``lsup`` columns.
    separacion_dim : Sequence[pd.DataFrame]
        List of DataFrames with the separations prior to clustering.

    Returns
    -------
    pd.DataFrame
        The separation from ``separacion_dim`` whose dimensions match
        those of ``df_clusterizado``.
    """

    dims_after_clus = list(df_clusterizado['linf'].columns)

    for i in range(len(separacion_dim)):
      dims = list(set(separacion_dim[i].dimension.values))
      if len(dims) == len(dims_after_clus):
        if all([a == b for a, b in zip(sorted(dims),
                                       sorted(dims_after_clus))]):
          break

    return separacion_dim[i]
    
  def mean_distance_ndim(self, df_sep_dm_agg):
    """Compute the average distance of each dimension.

    Parameters
    ----------
    df_sep_dm_agg : pd.DataFrame
        DataFrame with multi-index columns including ``linf`` and ``lsup``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the average of ``linf`` and ``lsup`` for each row.
    """

    df_p1 = df_sep_dm_agg.xs('linf', axis=1, level=0)
    df_p2 = df_sep_dm_agg.xs('lsup', axis=1, level=0)
    m_medios = [(df_p1.iloc[i] + df_p2.iloc[i]) / 2 for i in range(len(df_p1))]
    return pd.DataFrame(m_medios)
  
  def mean_distance_ndim_fast(self, df_sep_dm_agg, verbose):
      """
      Optimized version of mean_distance_ndim computing (linf + lsup)/2
      vectorized with NumPy.

      Parameters:
      - df_sep_dm_agg: DataFrame with multi-level indices to extract
        'linf' and 'lsup' using xs.

      Returns:
      - DataFrame with the average of linf and lsup per row and dimension.
      """

      # Extraemos linf y lsup
      df_p1 = df_sep_dm_agg.xs('linf', axis=1, level=0)
      df_p2 = df_sep_dm_agg.xs('lsup', axis=1, level=0)

      # Vectorized operation with NumPy:
      # (df_p1 + df_p2) / 2
      m_medios_values = (df_p1.values + df_p2.values) / 2.0

      # Rebuild DataFrame with the same index and columns as df_p1
      df_result = pd.DataFrame(
          m_medios_values,
          index=df_p1.index,
          columns=df_p1.columns
      )

      return df_result

  def posiciones_valores_frecuentes(self, lista):
    """Return the positions of the most frequent values in a list."""

    frecuentes = Counter(lista).most_common()
    if len(set(lista)) == len(lista):
      resultado = list(range(len(lista)))
    else:
      frecuencia_maxima = frecuentes[0][1]
      resultado = [
        i
        for i, v in enumerate(lista)
        if v in dict(frecuentes).keys() and dict(frecuentes)[v] == frecuencia_maxima
      ]
    return resultado
  
  def get_eps_multiple_groups_opt(
      self,
      data,
      eps_min: float = 1e-5,
      eps_max: float | None = None,
      *,
      strategy: str = "binary",
      grid_points: int = 75,
      min_samples: int = 2,
      max_precompute: int = 4000,
      tol: float = 1e-6,
      max_iter: int = 32,
      target_clusters: int | None = None,
  ):
    """Compute an ``eps`` value for DBSCAN that yields multiple groups.

    Parameters
    ----------
    data : array-like
        Dataset to cluster.
    eps_min, eps_max : float, optional
        Search interval bounds for ``eps``. ``eps_max`` defaults to the maximum
        pairwise distance (or an upper bound when distances aren't precomputed).
    strategy : {"binary", "grid"}
        Strategy to search for ``eps``. ``binary`` performs a binary search,
        ``grid`` sweeps ``grid_points`` evenly spaced values between ``eps_min``
        and ``eps_max``.
    grid_points : int, default 75
        Number of grid points when ``strategy='grid'``.
    min_samples : int, default 2
        ``min_samples`` parameter for DBSCAN.
    max_precompute : int, default 4000
        Maximum number of samples for which the pairwise distance matrix is
        precomputed. Above this limit the Euclidean metric is used directly.
    tol : float, default 1e-6
        Relative tolerance to stop the binary search.
    max_iter : int, default 32
        Maximum iterations for the binary search.
    target_clusters : int, optional
        Desired number of clusters. Defaults to 2 (i.e. obtain at least two
        clusters).
    """

    X = np.asarray(data)
    n = len(X)
    if n == 0:
      return eps_min
    if n == 1:
      return max(eps_min, 0.0)
    if n == 2:
      d12 = float(np.linalg.norm(X[0] - X[1]))
      return max(eps_min, d12)

    use_precomputed = n <= max_precompute
    if use_precomputed:
      D = pairwise_distances(X)
      if eps_max is None:
        eps_max = float(np.nanmax(D))
    else:
      D = None
      if eps_max is None:
        c = X.mean(axis=0)
        eps_max = float(2.0 * np.max(np.linalg.norm(X - c, axis=1)))

    eps_min = max(np.finfo(float).eps, float(eps_min))
    if not np.isfinite(eps_max) or eps_max <= eps_min:
      eps_max = eps_min * 10.0

    def fit_dbscan(eps: float) -> int:
      if use_precomputed:
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(D)
      else:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
      lbl = np.asarray(labels)
      mask = lbl != -1
      return 0 if not np.any(mask) else int(np.unique(lbl[mask]).size)

    target = 2 if target_clusters is None else int(target_clusters)

    if strategy == "grid":
      eps_values = np.linspace(eps_min, eps_max, num=max(2, grid_points))
      k_values = [fit_dbscan(eps) for eps in eps_values]
      ok = [i for i, k in enumerate(k_values) if k >= target]
      if ok:
        return float(eps_values[min(ok)])

      counts = Counter(k_values)
      mode = max(counts, key=counts.get)
      idxs = [i for i, k in enumerate(k_values) if k == mode]
      medias = np.mean([k_values[i] for i in idxs])
      mejor = min(idxs, key=lambda i: abs(k_values[i] - medias))
      return float(eps_values[mejor])

    low, high = float(eps_min), float(eps_max)
    best = None
    for _ in range(max_iter):
      mid = 0.5 * (low + high)
      k = fit_dbscan(mid)
      if k >= target:
        best = mid
        high = mid
      else:
        low = mid
      if abs(high - low) <= tol * max(1.0, high):
        break
    return float(best if best is not None else high)
  
    
  def fill_na_pond(self, df_sep_dm, df, features_val):
    """Replace ``±inf`` values in region limits with boundary values.

    Parameters
    ----------
    df_sep_dm : pd.DataFrame
        DataFrame with multi-index columns ``('linf', dim)`` and ``('lsup', dim)``
        describing lower and upper bounds for each dimension.
    df : pd.DataFrame
        Original data used to infer realistic replacement limits.
    features_val : list[str]
        Names of the dimensions present in ``df``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df_sep_dm`` where infinite values are replaced by
        ``min(df[features_val]) - 1`` for ``linf`` and ``max(df[features_val]) + 1``
        for ``lsup``. The original ``ponderador`` column is preserved.
    """

    df_ppfd = df_sep_dm.copy()
    lsup_limit = list(df[features_val].max()+1)
    linf_limit = list(df[features_val].min()-1)

    df_lilu = df_ppfd[['linf','lsup']].copy()
    replacement_dict = dict(zip(df_lilu.columns, linf_limit+lsup_limit))

    for index, value in replacement_dict.items():
      col_name = index[1]
      df_lilu.loc[(df_lilu[(index[0], col_name)] == -np.inf) | (df_lilu[(index[0], col_name)] == np.inf),
                    (index[0], col_name)] = value
    return pd.concat([df_lilu, df_sep_dm[['ponderador']]], axis=1)



  def fill_na_pond_fastest(self, df_sep_dm, df, features_val, verbose):
      """Vectorized replacement of ``±inf`` values in region limits.

      Parameters
      ----------
      df_sep_dm : pd.DataFrame
          DataFrame with multi-level columns (``linf``, ``lsup``, ``ponderador``…).
      df : pd.DataFrame
          Original dataset used to compute replacement limits for each
          dimension.
      features_val : list[str]
          Names of the dimensions present in ``df``.
      verbose : Any
          Unused; kept for backward compatibility.

      Returns
      -------
      pd.DataFrame
          DataFrame with infinite values replaced by the corresponding bounds
          and including the ``ponderador`` column.
      """
      # Extraer las columnas 'linf' y 'lsup'
      df_lilu = df_sep_dm[['linf', 'lsup']].copy()
      
      # Calculate replacement limits for each dimension
      lsup_limit = df[features_val].max() + 1  # Upper limit
      linf_limit = df[features_val].min() - 1  # Lower limit
      
      # Ensure features_val order matches column order
      # Get dimension names from the MultiIndex columns
      linf_features = df_lilu['linf'].columns.tolist()
      lsup_features = df_lilu['lsup'].columns.tolist()
      
      # For 'linf' columns
      linf_repl_df = pd.DataFrame(
          np.tile(linf_limit.values, (df_lilu['linf'].shape[0], 1)),
          columns=df_lilu['linf'].columns,
          index=df_lilu.index
      )
      
      # For 'lsup' columns
      lsup_repl_df = pd.DataFrame(
          np.tile(lsup_limit.values, (df_lilu['lsup'].shape[0], 1)),
          columns=df_lilu['lsup'].columns,
          index=df_lilu.index
      )
      
      # Create masks to identify where -inf and inf occur
      mask_linf = np.isinf(df_lilu['linf'].values)
      mask_lsup = np.isinf(df_lilu['lsup'].values)
      
      # Apply masks and replace values
      # Use where to assign replacement values where mask is True
      df_lilu['linf'] = np.where(mask_linf, linf_repl_df.values, df_lilu['linf'].values)
      df_lilu['lsup'] = np.where(mask_lsup, lsup_repl_df.values, df_lilu['lsup'].values)
      
      # Concatenar la columna 'ponderador' de vuelta al DataFrame
      df_replaced = pd.concat([df_lilu, df_sep_dm[['ponderador','ef_sample','n_sample']]], axis=1)
      
      return df_replaced

  def group_by_cluster(self, df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
      """Group by cluster keeping the first value of other columns.

      Parameters
      ----------
      df : pd.DataFrame
          Input DataFrame containing a cluster column and additional metrics.
      cluster_col : str, default "cluster"
          Name of the column used for grouping.

      Returns
      -------
      pd.DataFrame
          DataFrame with one row per cluster including the first occurrence of
          every other column. If multiple ``ef_sample``/``n_sample`` columns are
          present only the first is preserved. ``count`` indicates how many
          original rows contributed to each cluster.
      """
      
      # Verificar si la columna 'cluster' contiene valores no escalares (listas, sets, dicts, etc.)
      if df[cluster_col].apply(lambda x: isinstance(x, (list, tuple, set, dict))).any():
          df = df.copy()  # Para evitar SettingWithCopyWarning
          df[cluster_col] = df[cluster_col].apply(
              lambda x: tuple(x) if isinstance(x, (list, set)) else x
          )
      
      # Identify columns containing 'ef_sample' and 'n_sample'
      ef_sample_cols = [col for col in df.columns if "ef_sample" in col]
      n_sample_cols = [col for col in df.columns if "n_sample" in col]
      ponderador_cols = [col for col in df.columns if "ponderador" in col]

      # Select the first column of each type if present
      first_ef_sample = ef_sample_cols[0] if ef_sample_cols else None
      first_n_sample = n_sample_cols[0] if n_sample_cols else None
      ponderador_sample = ponderador_cols[0] if ponderador_cols else None

      # Build list of columns to keep:
      # - Include the cluster column explicitly.
      # - Add all columns not part of ef_sample or n_sample lists nor the cluster column itself (to avoid duplicates).
      cols_to_keep = [cluster_col] + [
          col for col in df.columns if col not in (ef_sample_cols + n_sample_cols + ponderador_cols + [cluster_col])
      ]
      
      # Add the first 'ef_sample' and 'n_sample' columns if they exist
      if first_ef_sample:
          cols_to_keep.append(first_ef_sample)
      if first_n_sample:
          cols_to_keep.append(first_n_sample)
      if ponderador_sample:
          cols_to_keep.append(ponderador_sample)
      
      # Remove duplicates in 'cols_to_keep' preserving order
      cols_to_keep = list(dict.fromkeys(cols_to_keep))
      
      # Group by 'cluster' and keep the first value of each selected column
      df_grouped = df[cols_to_keep].groupby(cluster_col, as_index=False).first()
      df_grouped_c = df[cols_to_keep[:2]].groupby(cluster_col, as_index=False).count()
      df_grouped_c = df_grouped_c.rename(columns={cols_to_keep[1]:'count'})
      
      return df_grouped.merge(df_grouped_c, how='left', on='cluster')


  def get_agg_regions_j(self, df_eval, df, *, nan_policy: str = "raise"):
    """Aggregate similar rectangles using IoU and clustering.

    Parameters
    ----------
    df_eval : pd.DataFrame
        DataFrame with columns ``['rectangulo', 'dimension', 'linf', 'lsup',
        'n_sample', 'ef_sample']`` describing candidate regions.
    df : pd.DataFrame
        Original dataset used to compute replacement limits when filling
        ``±inf`` values.
    nan_policy : {"raise", "zero"}, default "raise"
        How to handle NaNs in bounds when computing IoU.

    Returns
    -------
    pd.DataFrame
        Aggregated regions with multi-index columns and cluster assignments.
    """

    features_val = sorted(df_eval['dimension'].unique())
    df_sep_dm = pd.pivot_table(df_eval, index='rectangulo', columns='dimension')
    df_sep_dm = self.fill_na_pond_fastest(df_sep_dm, df, features_val,None)

    # Flatten MultiIndex if already present
    df_sep_dm.columns = [f"{col[1].strip()}&&{col[0]}" for col in df_sep_dm.columns]
    df_raw = df_sep_dm.copy()

    if len(df_raw) == 1:
        df_raw["cluster"] = 1
        return self.group_by_cluster(df_raw)

    # List of dimensions extracted from column names
    dims = sorted(set(col.split('&&')[0] for col in df_raw.columns))

    # Vectorized computation of IoU for all pairs of hypercubes
    low_cols = [f"{dim}&&linf" for dim in dims]
    high_cols = [f"{dim}&&lsup" for dim in dims]

    lows = df_raw[low_cols].to_numpy(dtype=np.float64, copy=False)
    highs = df_raw[high_cols].to_numpy(dtype=np.float64, copy=False)
    lows, highs = _clean_bounds(lows, highs, nan_policy=nan_policy)

    n = len(df_raw)
    block = choose_block(n, len(dims))
    distance_matrix = pairwise_iou_blocked(lows, highs, block=block)

    # Convert matrix to condensed distance vector for clustering
    dist_vector = squareform(distance_matrix)

    # Apply hierarchical clustering
    Z = linkage(dist_vector, method='average')  # Average linkage method

    # Determine number of clusters (adjust threshold as needed)

    for tr in [tr/100 for tr in range(65,5,-1)]:
      clusters = fcluster(Z, tr, criterion='distance')
      if len(set(clusters))*2>len(distance_matrix):
        break

    # Add cluster assignments to DataFrame
    df_raw["cluster"] = clusters

    n_sample_col = [col for col in df_raw.columns if 'n_sample' in col][0]
    eff_sample_col = [col for col in df_raw.columns if 'ef_sample' in col][0]

    df_raw.sort_values(by=['cluster', n_sample_col, eff_sample_col], ascending=False, inplace=True)

    df_raw_agg = self.group_by_cluster(df_raw)

    return df_raw_agg


  def set_multiindex(self, df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
      """Set ``cluster`` as index and convert columns to a MultiIndex.

      The column names are expected to use ``"&&"`` as a separator between a
      dimension name and a statistic (e.g. ``"age&&linf"``). Special statistic
      names (``ef_sample``, ``n_sample``, ``ponderador``, ``count``) are grouped
      under the ``"metrics"`` level.

      Parameters
      ----------
      df : pd.DataFrame
          Input DataFrame with a cluster column and region metrics.
      cluster_col : str, default "cluster"
          Name of the column to use as index.

      Returns
      -------
      pd.DataFrame
          DataFrame indexed by ``cluster`` with a two-level column MultiIndex
          ``(level_0, dimension)`` where ``level_0`` is ``metrics`` or the
          original identifier.
      """
      # Define special words
      special_words = {"ef_sample", "n_sample", "ponderador", "count"}
      new_cols = []
      
      # Process each column (except the cluster column)
      for col in df.columns:
          if col == cluster_col:
              continue  # Will be used as index
          if "&&" in col:
              # Split into left and right
              left, right = col.split("&&", 1)
              left = left.strip()
              right = right.strip()
              if right in special_words:
                  # For special columns: ("metrics", <special name>) -> right
                  new_label = ("metrics", right)
              else:
                  # For non-special columns: (identifier, measure name)
                  new_label = (right, left)
          else:
              # Columns without "&&" separator
              if any(sw in col for sw in special_words):
                  new_label = ("metrics", col)
              else:
                  new_label = (None, col)
          new_cols.append(new_label)
      
      # Create MultiIndex with level names [None, 'dimension']
      multi_cols = pd.MultiIndex.from_tuples(new_cols, names=[None, "dimension"])
      
      # Set cluster column as index and assign new MultiIndex to columns
      df_new = df.set_index(cluster_col).copy()
      df_new.columns = multi_cols
      return df_new

  def prio_ranges(self, separacion_dim, df):
    """Prioritize regions ordering by sample size and effectiveness.

    Parameters
    ----------
    separacion_dim : Sequence[pd.DataFrame]
        List of DataFrames describing candidate regions for each tree.
    df : pd.DataFrame
        Original dataset used to compute metrics.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames with MultiIndex columns sorted by ``n_sample``,
        ``count`` and ``ef_sample``.
    """

    df_res = [self.set_multiindex(self.get_agg_regions_j(df_, df))
              for df_ in separacion_dim]

    df_res = [df.sort_values(by=[('metrics', 'n_sample'), 
                                ('metrics', 'count'), 
                                ('metrics', 'ef_sample')], ascending=False) for df in df_res]

    return df_res


  def plot_bidim(self, df, df_sep_dm_agg, x_axis, y_axis, var_obj,
                 interactive=False, ax=None):
    """Plot two-dimensional regions and associated data points.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset used for plotting.
    df_sep_dm_agg : pd.DataFrame
        DataFrame containing region limits with ``linf`` and ``lsup``.
    x_axis : str
        Name of the column to use on the x-axis.
    y_axis : str
        Name of the column to use on the y-axis.
    var_obj : str
        Target variable used for coloring the scatter plot.
    interactive : bool, optional (default ``False``)
        If ``True``, enable Matplotlib's interactive mode by calling
        ``plt.ion()`` and displaying the figure with
        ``plt.show(block=False)``. Interactivity depends on the
        Matplotlib backend and may not be available in all
        environments.
    ax : matplotlib.axes.Axes, optional
        Axes object on which to draw. If ``None``, a new figure and
        axes are created implicitly by Seaborn.
    """

    df_sep_dm_agg['width'] = df_sep_dm_agg[('lsup', x_axis)] - \
                            df_sep_dm_agg[('linf', x_axis)]
    df_sep_dm_agg['height'] = df_sep_dm_agg[('lsup', y_axis)] - \
                             df_sep_dm_agg[('linf', y_axis)]

    lower_bounds = df_sep_dm_agg['linf'].values
    width = df_sep_dm_agg['width'].values
    height = df_sep_dm_agg['height'].values

    ax = sns.scatterplot(x=x_axis, y=y_axis, hue=var_obj,
                         palette='RdBu', data=df, ax=ax)
    norm = plt.Normalize(df[var_obj].min(), df[var_obj].max())
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])
    for i in range(len(lower_bounds)):
      if i > 25:
        break
      ax.add_patch(Rectangle(lower_bounds[i], width[i], height[i],
                             alpha=0.15, color='#0099FF'))
    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm, ax=ax)

    if interactive:
      plt.ion()
      plt.show(block=False)

    
  def plot_scatter3d(self, df_r, df, ax, var_obj):
    """Draw a 3D scatter plot of the data associated with a region."""

    dimension_columns = df_r['linf'].columns
    df_scatter = df[list(dimension_columns) + [var_obj]].copy()
    df_scatter.replace(dict(zip([True, False], [1, 0])), inplace=True)

    target_values = list(df_scatter[var_obj].unique())
    discrete_colors = ['red', 'blue', 'green', 'yellow', 'orange']
    replace_var = dict(zip(target_values, discrete_colors[:len(target_values)]))

    # Generate scatter plot points
    xs = df_scatter[dimension_columns[0]].values
    ys = df_scatter[dimension_columns[1]].values
    zs = df_scatter[dimension_columns[2]].values
    colors = df_scatter[var_obj].replace(replace_var).values
    hex_colors = [mcolors.to_hex(c) for c in colors]
    # Draw the points
    return ax.scatter(xs, ys, zs, c=hex_colors, s=1)
    

  def plot_rect3d(self, df_r, i, ax):
    """Draw a 3D rectangle corresponding to region ``i``."""

    # Obtain rectangle limit values
    x1, y1, z1, x2, y2, z2 = df_r.drop(columns=['metrics']).iloc[i,:].values.flatten()

    # Generate rectangle points
    X = np.array([[x1, x2, x2, x1, x1], [x1, x2, x2, x1, x1]])
    Y = np.array([[y1, y1, y2, y2, y1], [y1, y1, y2, y2, y1]])
    Z = np.array([[z1, z1, z1, z1, z1], [z2, z2, z2, z2, z2]])

    # Draw the rectangle with transparency
    return ax.plot_surface(X=X, Y=Y, Z=Z, alpha=0.2, color='gray')

  def plot_tridim(self, df_r, df, var_obj, interactive=False):
    """Plot three-dimensional regions together with the data.

    Parameters
    ----------
    df_r : pd.DataFrame
        DataFrame containing region limits with ``linf`` and ``lsup``.
    df : pd.DataFrame
        Original dataset used for plotting.
    var_obj : str
        Target variable used for coloring the points.
    interactive : bool, optional (default ``False``)
        If ``True``, enable Matplotlib's interactive mode by calling
        ``plt.ion()`` and displaying the figure with
        ``plt.show(block=False)``. Interactivity depends on the
        Matplotlib backend and may not be available in all
        environments.
    """

    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(df_r.shape[0]):
        self.plot_rect3d(df_r, i, ax)

    self.plot_scatter3d(df_r, df, ax, var_obj)

    dimension_columns = df_r['linf'].columns
    # Configure axis labels
    ax.set_xlabel(str(dimension_columns[0]))
    ax.set_ylabel(str(dimension_columns[1]))
    ax.set_zlabel(str(dimension_columns[2]))

    if interactive:
      plt.ion()
      plt.show(block=False)
    else:
      plt.show()
    
    
  def plot_multi_dims(self, df_sep_dm_agg, df, var_obj, interactive=False,
                      force=True):
    """Visualize the regions adapting to the number of dimensions.

    Parameters
    ----------
    df_sep_dm_agg : pd.DataFrame
        DataFrame containing aggregated region limits with ``linf`` and ``lsup``.
    df : pd.DataFrame
        Original dataset used for plotting.
    var_obj : str
        Target variable used for coloring the plots.
    interactive : bool, optional (default ``False``)
        If ``True``, enable Matplotlib's interactive mode by calling
        ``plt.ion()`` and displaying the figure with
        ``plt.show(block=False)``. Interactivity depends on the
        Matplotlib backend and may not be available in all
        environments.
    force : bool, optional (default ``True``)
        For datasets with more than three dimensions. If ``True``, all
        two-dimensional combinations are plotted as subplots. If
        ``False``, a :class:`ValueError` is raised with an explanatory
        message.

    Behavior
    --------
    - 1 dimension: Creates a temporary height axis and plots in 2D.
    - 2 dimensions: Plots regions and data in 2D.
    - 3 dimensions: Delegates plotting to ``plot_tridim``.
    - More than 3 dimensions: if ``force=True``, plots all 2D
      combinations as subplots; otherwise raises an error.

    Returns
    -------
    None
        This function generates plots and does not return a value.
    """

    # Work on a copy to avoid modifying the original data when the
    # function is called multiple times. Otherwise, auxiliary columns
    # like ``height`` added for the 1D case would persist and break
    # subsequent calls.
    df_sep_dm_agg = df_sep_dm_agg.copy()

    dimensions = df_sep_dm_agg['linf'].columns
    if len(dimensions) == 1:
      x_axis = dimensions.tolist()[0]
      y_axis = 'height'
      df_tmp = df.copy()
      df_tmp.loc[:, y_axis] = 1
      median_padding = abs((df_sep_dm_agg['linf'] - df_sep_dm_agg['lsup']).mean()[0])
      df_sep_dm_agg[('lsup', y_axis)] = 1 + median_padding
      df_sep_dm_agg[('linf', y_axis)] = 1 - median_padding

      # ``plot_bidim`` modifies the dataframe it receives (adding width
      # and height columns). Pass a copy so the caller's dataframe
      # remains untouched.
      self.plot_bidim(df_tmp, df_sep_dm_agg.copy(), x_axis, y_axis, var_obj,
                      interactive=interactive)
    elif len(dimensions) == 2:
      df_tmp = df.copy()
      x_axis, y_axis = dimensions.tolist()
      self.plot_bidim(df_tmp, df_sep_dm_agg.copy(), x_axis, y_axis, var_obj,
                      interactive=interactive)
    elif len(dimensions) == 3:
      self.plot_tridim(df_sep_dm_agg, df, var_obj, interactive=interactive)
    else:
      if not force:
        raise ValueError(
            "More than three dimensions detected. Set force=True to plot "
            "all 2D combinations as subplots.")

      dim_pairs = list(combinations(dimensions, 2))
      n_pairs = len(dim_pairs)
      n_cols = math.ceil(math.sqrt(n_pairs))
      n_rows = math.ceil(n_pairs / n_cols)
      fig, axes = plt.subplots(n_rows, n_cols,
                               figsize=(5 * n_cols, 4 * n_rows))
      axes = np.array(axes).reshape(n_rows, n_cols)

      for ax, (x_axis, y_axis) in zip(axes.flat, dim_pairs):
        self.plot_bidim(df.copy(), df_sep_dm_agg.copy(), x_axis, y_axis,
                        var_obj, interactive=False, ax=ax)
        ax.set_title(f"{x_axis} vs {y_axis}")

      for ax in list(axes.flat)[n_pairs:]:
        ax.axis('off')

      if interactive:
        plt.ion()
        plt.show(block=False)
      else:
        plt.show()


  def plot_experiments(self, df, table_row, interactive=False):
    """Plot experiment subsets defined by exclusive rules.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables to plot.
    table_row : Mapping or pd.Series
        Row from the experiments table. It must provide the following
        columns already parsed or as string representations of Python
        lists: ``intersection``, ``only_cluster_a``, ``only_cluster_b``,
        ``variables_a`` and ``variables_b``.
    interactive : bool, optional
        If ``True`` the plot is displayed in interactive mode. By
        default, ``False``.
    """

    def _parse_list(value):
      """Return a Python list from strings or other iterables."""
      if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
      if isinstance(value, str):
        try:
          return ast.literal_eval(value)
        except (ValueError, SyntaxError):
          return []
      if isinstance(value, (list, tuple, set)):
        return list(value)
      return [value]

    # Extract parameters from the table row
    get = table_row.get if hasattr(table_row, "get") else table_row.__getitem__
    intersection = _parse_list(get("intersection", []))
    only_cluster_a = _parse_list(get("only_cluster_a", []))
    only_cluster_b = _parse_list(get("only_cluster_b", []))
    variables_a = _parse_list(get("variables_a", []))
    variables_b = _parse_list(get("variables_b", []))

    def _apply_conditions(data, conds):
      for cond in conds:
        match = re.match(r"\s*([\-\d\.eE]+)\s*<=\s*([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*([\-\d\.eE]+)", str(cond))
        if match:
          low, var, high = float(match.group(1)), match.group(2), float(match.group(3))
          data = data[(data[var] >= low) & (data[var] <= high)]
      return data

    df_filtered = df.copy()
    if intersection:
      df_filtered = _apply_conditions(df_filtered, intersection)

    df_a = _apply_conditions(df_filtered.copy(), only_cluster_a)
    df_b = _apply_conditions(df_filtered.copy(), only_cluster_b)

    dims = sorted(set(variables_a) | set(variables_b))
    if len(dims) == 0:
      logger.warning('No dimensions to plot.')
      return

    if len(dims) == 2:
      ax = sns.scatterplot(data=df_a, x=dims[0], y=dims[1], color='blue', label='A')
      sns.scatterplot(data=df_b, x=dims[0], y=dims[1], color='red', label='B', ax=ax)
      ax.set_title('cluster A vs cluster B')
      plt.legend()
      if interactive:
        plt.ion()
        plt.show(block=False)
      else:
        plt.show()
    elif len(dims) == 3:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(df_a[dims[0]], df_a[dims[1]], df_a[dims[2]], color='blue', label='A')
      ax.scatter(df_b[dims[0]], df_b[dims[1]], df_b[dims[2]], color='red', label='B')
      ax.set_xlabel(dims[0])
      ax.set_ylabel(dims[1])
      ax.set_zlabel(dims[2])
      ax.legend()
      if interactive:
        plt.ion()
        plt.show(block=False)
      else:
        plt.show()
    else:
      dim_pairs = list(combinations(dims, 2))
      n_pairs = len(dim_pairs)
      n_cols = math.ceil(math.sqrt(n_pairs))
      n_rows = math.ceil(n_pairs / n_cols)
      fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
      axes = np.array(axes).reshape(n_rows, n_cols)
      for ax, (x_axis, y_axis) in zip(axes.flat, dim_pairs):
        sns.scatterplot(data=df_a, x=x_axis, y=y_axis, color='blue', label='A', ax=ax)
        sns.scatterplot(data=df_b, x=x_axis, y=y_axis, color='red', label='B', ax=ax)
        ax.set_title(f'{x_axis} vs {y_axis}')
      for ax in list(axes.flat)[n_pairs:]:
        ax.axis('off')
      if interactive:
        plt.ion()
        plt.show(block=False)
      else:
        plt.show()


  def asignar_ids_unicos(self, lista_reglas):
    """Assign a unique identifier to each rule in a list."""

    cluster_id = 0
    for df in lista_reglas:
      n_reglas = len(df)
      df = df.reset_index(drop=True)
      df['cluster_'] = range(cluster_id, cluster_id + n_reglas)
      cluster_id += n_reglas
    return lista_reglas

  def generar_descripcion_clusters(self, df_reglas):
      """
      Generate a DataFrame with textual descriptions and weights for each cluster
      based on the rules defined in ``df_reglas``.

      Assumes:
        - row['linf'] and row['lsup'] are Series indexed by variable names.
        - The weight is taken from ('metrics', 'ponderador').
        - The 'cluster' column holds the cluster identifier.
      """
      # import numpy as np
      # import pandas as pd

      cluster_descripciones = []

      for idx, row in df_reglas.iterrows():
          # Extraer el identificador del cluster; si viene encapsulado, extraer el valor escalar
          cluster_id = row['cluster']
          if hasattr(cluster_id, 'values'):
              cluster_id = cluster_id.values[0]

          # Extract lower and upper bounds
          linf = row['linf'].dropna()
          lsup = row['lsup'].dropna()

          # Build description: for each variable in linf create a string
          descripcion_partes = [f"{linf[var]} <= {var} <= {lsup[var]}" for var in linf.index]
          descripcion = " AND ".join(descripcion_partes)

          # Extraer el ponderador desde el grupo 'metrics'
          # This is the changed line. Accessing using the MultiIndex tuple.
          ponderador = row[('metrics', 'ponderador')]
          ef_sample = row[('metrics', 'ef_sample')]
          n_sample = row[('metrics', 'n_sample')]
          count = row[('metrics', 'count')]

          if hasattr(ponderador, '__iter__'):
              # print('media_ponderadore')
              ponderador = np.mean(ponderador)

          cluster_descripciones.append({
              'cluster': cluster_id,
              'cluster_descripcion': descripcion,
              'cluster_ponderador': ponderador,
              'cluster_ef_sample': ef_sample,
              'cluster_n_sample': n_sample,
              'cluster_count': count
          })

      df_clusters_descripcion = pd.DataFrame(cluster_descripciones)

      return df_clusters_descripcion


  def asignar_clusters_a_datos(self, df_datos, df_reglas, keep_all_clusters=True):
      """
      Assign clusters to data based on rules.

      Parameters
      ----------
      df_datos : pd.DataFrame
          DataFrame containing the data to assign to clusters.
      df_reglas : pd.DataFrame
          DataFrame defining the rules with MultiIndex columns where:
            - Lower limits are under first level 'linf'
            - Upper limits are under first level 'lsup'
            - Weight is stored at ('metrics', 'ponderador')
            - The 'cluster' column appears normally.
      keep_all_clusters : bool, optional (default=True)
          If True, four additional columns are added:
            - 'n_clusters': number of clusters the record belongs to.
            - 'clusters_list': list of all clusters the record belongs to.
            - 'ponderadores_list': list of weights of those clusters.
            - 'ponderador_mean': mean weight of those clusters.
          If False, only the cluster with highest weight is assigned (no extra columns).

      Returns
      -------
      df_datos_clusterizados : pd.DataFrame
          Input DataFrame with the 'cluster' column assigned.
          If keep_all_clusters=True, the additional columns are included.
      df_clusters_descripcion : pd.DataFrame
          DataFrame with description or metrics for each cluster.
      """
      # import numpy as np

      n_datos = df_datos.shape[0]
      # Array para almacenar el cluster "principal" (el que tiene mayor ponderador)
      clusters_datos = np.full(n_datos, -1, dtype=float)
      # Array to store the weight of the main cluster
      ponderador_datos = np.full(n_datos, -np.inf, dtype=float)

      # If we want to keep the full list of clusters per record, initialize structures
      if keep_all_clusters:
          clusters_datos_all = [[] for _ in range(n_datos)]
          ponderadores_datos_all = [[] for _ in range(n_datos)]

      # --- 1) Extract and normalize rule information ---
      reglas_info = []
      # Iterate each row of df_reglas; normally the number of rules is small
      for idx, row in df_reglas.iterrows():
          if row[('metrics', 'ponderador')]==0:
              continue
          # Extract lower and upper bounds; assumes row['linf'] and row['lsup'] are Series
          linf = row['linf'].dropna()
          lsup = row['lsup'].dropna()
          variables = linf.index.tolist()

          # Extract the weight from the 'metrics' group
          p_val = row[('metrics', 'ponderador')]
          # If for some reason it is iterable (e.g., a list), take its mean;
          # under normal conditions it's a scalar value.
          ponderador = p_val.mean() if hasattr(p_val, '__iter__') else p_val

          # Extract the assigned cluster; if encapsulated, get the scalar value.
          cluster_raw = row['cluster']
          if hasattr(cluster_raw, 'values') and len(cluster_raw.values) == 1:
              cluster_raw = float(cluster_raw.values[0])
          else:
              cluster_raw = float(cluster_raw)

          reglas_info.append({
              'variables': variables,
              'linf': linf.to_dict(),
              'lsup': lsup.to_dict(),
              'ponderador': ponderador,
              'cluster': cluster_raw,
          })

      # --- 2) Evaluate which rules each record satisfies ---
      for regla in reglas_info:
          variables = regla['variables']
          linf = regla['linf']
          lsup = regla['lsup']
          ponderador = regla['ponderador']
          cluster = regla['cluster']

          # Extract relevant columns and convert to arrays for vectorized operations
          X_datos = df_datos[variables]
          condiciones = [
              (X_datos[var].to_numpy() >= linf[var]) & (X_datos[var].to_numpy() <= lsup[var])
              for var in variables
          ]
          # If there are no variables, the rule can't be evaluated; assume no record satisfies it.
          if condiciones:
              cumple_regla = np.logical_and.reduce(condiciones)
          else:
              cumple_regla = np.zeros(n_datos, dtype=bool)

          if keep_all_clusters:
              indices_cumple = np.where(cumple_regla)[0]
              for i in indices_cumple:
                  clusters_datos_all[i].append(cluster)
                  ponderadores_datos_all[i].append(ponderador)

          # Update the "main" cluster if this rule's weight is higher
          actualizar = cumple_regla & (ponderador > ponderador_datos)
          clusters_datos[actualizar] = cluster
          ponderador_datos[actualizar] = ponderador

      # --- 3) Construir el DataFrame de salida ---
      df_datos_clusterizados = df_datos.copy()
      df_datos_clusterizados['cluster'] = clusters_datos  # Cluster final (mayor ponderador)

      if keep_all_clusters:
          df_datos_clusterizados['n_clusters'] = [len(lst) for lst in clusters_datos_all]
          df_datos_clusterizados['clusters_list'] = clusters_datos_all
          df_datos_clusterizados['ponderadores_list'] = ponderadores_datos_all
          df_datos_clusterizados['ponderador_mean'] = [
              np.mean(lst) if lst else np.nan for lst in ponderadores_datos_all
          ]

      # --- 4) Generar la descripción de clusters (usando el método propio)
      df_clusters_descripcion = self.generar_descripcion_clusters(df_reglas)

      return df_datos_clusterizados, df_clusters_descripcion

  def eliminar_reglas_redundantes(self, lista_reglas):
      """
      Remove only those rules strictly contained in others
      whose weight is lower than the containing rule.

      Comparison across all elements in the list:
        - Concatenate all DataFrames (each may have a different set of dimensions).
        - For each rule, evaluate whether it is contained in other rules with
          equal or greater number of dimensions.
        - If containment is total and the larger rule has strictly greater weight,
          remove the contained rule.

      Parameters
      ----------
      lista_reglas : List[pd.DataFrame]
          List of DataFrames, each with MultiIndex columns:
          - 'linf': lower bounds of each dimension
          - 'lsup': upper bounds of each dimension
          - ('metrics', 'ponderador'): weight of the rule
          - Other columns (e.g., 'cluster') preserved but unused in the comparison.

          Each DataFrame may have a different set of dimensions.

      Returns
      -------
      df_reglas_importantes : pd.DataFrame
          DataFrame with all non-redundant rules from all DataFrames.
      """
      # import numpy as np
      # import pandas as pd

      # 1. Concatenar todos los DataFrames en uno para comparar reglas de toda la lista
      df_reglas = pd.concat(lista_reglas, ignore_index=True)

      # Asegurar que los nombres de nivel de las columnas sean [None, 'dimension']
      if df_reglas.columns.names != [None, 'dimension']:
          df_reglas.columns.names = [None, 'dimension']

      # 2. Extract essential information from each rule
      reglas_info = []
      for idx, row in df_reglas.iterrows():
          linf = row['linf']        # Lower bounds (Series)
          lsup = row['lsup']        # Upper bounds (Series)
          p = row[('metrics', 'ponderador')]  # Weight (scalar value)

          # Set of dimensions defined by the rule (using linf.dropna())
          # Assume "no limit" => NaN and thus not an active dimension.
          vars_i = set(linf.dropna().index)

          reglas_info.append({
              'idx': idx,
              'variables': vars_i,
              'linf': linf.to_dict(),
              'lsup': lsup.to_dict(),
              'ponderador': p
          })

      # 3. Ordenar las reglas por la cantidad de dimensiones (de menor a mayor)
      #    This helps reduce the number of comparisons.
      reglas_info_sorted = sorted(reglas_info, key=lambda x: len(x['variables']))

      # 4. Recorrer las reglas y marcar las que resulten redundantes
      redundant_indices = set()
      num_reglas = len(reglas_info_sorted)

      for i in range(num_reglas):
          rule_i = reglas_info_sorted[i]
          if rule_i['idx'] in redundant_indices:
              # Already marked redundant in a previous comparison
              continue

          set_i = rule_i['variables']
          # Compare only with subsequent rules in the list,
          # since they are ordered and will have >= number of dimensions.
          for j in range(i + 1, num_reglas):
              rule_j = reglas_info_sorted[j]

              # 4.1. Requerimos que el conjunto de variables i sea un subconjunto del j
              if not set_i.issubset(rule_j['variables']):
                  continue

              # 4.2. Verify boundary containment: for each variable in i,
              #      linf_i >= linf_j y lsup_i <= lsup_j.
              contenido = True
              for var in set_i:
                  if (rule_i['linf'][var] < rule_j['linf'][var]) or \
                    (rule_i['lsup'][var] > rule_j['lsup'][var]):
                      contenido = False
                      break

              if not contenido:
                  continue

              # 4.3. El ponderador de j debe ser estrictamente mayor
              if rule_j['ponderador'] > rule_i['ponderador']:
                  # i es redundante, se marca y dejamos de compararla
                  redundant_indices.add(rule_i['idx'])
                  break  # No more comparisons needed for i

      # 5. Eliminar las reglas redundantes y reindexar
      df_reglas_importantes = df_reglas.drop(index=redundant_indices).reset_index(drop=True)
      return df_reglas_importantes


  def combinar_dataframes_por_columnas(self, lista_reglas):
      """
      Combine DataFrames in the list that have the same columns (MultiIndex) into one.

      The function groups DataFrames by their column structure and concatenates those
      sharing the same structure. If a group contains a single DataFrame, it is kept as is.

      Parameters
      ----------
      lista_reglas : list of pd.DataFrame
          List of DataFrames with MultiIndex columns.

      Returns
      -------
      nueva_lista : list of pd.DataFrame
          New list where each element is the result of concatenating (or keeping)
          DataFrames that share the same column structure.
      """
      grupos = {}
      for df in lista_reglas:
          cols_key = tuple(df.columns.tolist())
          grupos.setdefault(cols_key, []).append(df)

      nueva_lista = []
      for cols_key, dfs in grupos.items():
          if len(dfs) > 1:
              # Combina los DataFrames que comparten las mismas columnas
              df_combinado = pd.concat(dfs, ignore_index=True)
          else:
              df_combinado = dfs[0]
          nueva_lista.append(df_combinado)

      return nueva_lista


##-----Incorporar clusters simplificado


  def expandir_clusters_binario(self, df, columna_clusters, prefijo='cluster_'):
      """
      Expand a column of cluster lists into multiple binary columns.

      Parameters
      ----------
      df : pd.DataFrame
          DataFrame containing the column to expand.
      columna_clusters : str
          Name of the column with cluster lists.
      prefijo : str, optional
          Prefix for the new column names. Default is 'cluster_'.

      Returns
      -------
      pd.DataFrame
          Original DataFrame joined with the new binary cluster columns.
      """
      # Asegurar que la columna de clusters contiene listas o sets
      if not df[columna_clusters].apply(lambda x: isinstance(x, (list, set))).all():
          raise ValueError(f"Column '{columna_clusters}' must contain lists or sets of clusters.")
      
      # Inicializar el binarizador
      mlb = MultiLabelBinarizer()
      
      # Aplicar el binarizador a la columna de clusters
      clusters_binarizados = mlb.fit_transform(df[columna_clusters])
      
      # Crear nombres de columnas con el prefijo y el identificador del cluster
      nombres_columnas = [f"{prefijo}{cluster}" for cluster in mlb.classes_]
      
      # Crear un DataFrame con las columnas binarizadas
      df_clusters_bin = pd.DataFrame(clusters_binarizados, columns=nombres_columnas, index=df.index)
      
      # Unir las nuevas columnas binarias al DataFrame original
      df_final = pd.concat([df, df_clusters_bin], axis=1)
      
      return df_final

  def apply_clustering_and_similarity(self, df, cluster_columns, dbscan_params=None, kmeans_params=None):
      """
      Aplica DBSCAN y KMeans a las columnas de clusters seleccionadas, agrega las etiquetas
      de clustering al DataFrame y encuentra las columnas que más se parecen a cada
      conjunto de etiquetas de clustering.
      
      Parámetros
      ----------
      df : pd.DataFrame
          DataFrame que contiene las columnas de clusters seleccionadas.
      cluster_columns : list of str
          Lista de nombres de columnas binarias a usar para el clustering.
      dbscan_params : dict, optional
          Parámetros para el algoritmo DBSCAN. Si no se proporciona, se usarán los valores por defecto.
      kmeans_params : dict, optional
          Parámetros para el algoritmo KMeans. Si no se proporciona, se usarán los valores por defecto.
      
      Retorna
      -------
      df : pd.DataFrame
          DataFrame original con columnas adicionales:
              - 'db_labels': etiquetas de cluster de DBSCAN
              - 'km_labels': etiquetas de cluster de KMeans
              - 'db_most_similar_cluster': nombre de la columna más similar a 'db_labels'
              - 'km_most_similar_cluster': nombre de la columna más similar a 'km_labels'
      correlation_df : pd.DataFrame
          DataFrame con las correlaciones entre los cluster labels y las columnas de clusters.
      """
      # Validar que las columnas existen
      for col in cluster_columns:
          if col not in df.columns:
              raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
      
      # Preparar los datos para clustering
      X = df[cluster_columns].values
      
      # Aplicar DBSCAN
      if dbscan_params is None:
          dbscan_params = {'eps': 0.5, 'min_samples': 2}
      dbscan = DBSCAN(**dbscan_params)
      db_labels = dbscan.fit_predict(X)
      df['db_labels'] = db_labels
      
      # Aplicar KMeans
      if kmeans_params is None:
          # Elegir n_clusters de KMeans, aquí se usa 2 por defecto
          kmeans_params = {'n_clusters': 2, 'random_state': 42}
      kmeans = KMeans(**kmeans_params)
      km_labels = kmeans.fit_predict(X)
      df['km_labels'] = km_labels
      
      # Calcular correlaciones entre db_labels y cluster_columns
      # Usamos correlación de Pearson
      correlation_db = df[cluster_columns].apply(lambda col: df['db_labels'].corr(col))
      correlation_km = df[cluster_columns].apply(lambda col: df['km_labels'].corr(col))
      
      # Encontrar las columnas más similares
      db_most_similar_cluster = correlation_db.abs().idxmax()
      km_most_similar_cluster = correlation_km.abs().idxmax()
      
      # Agregar las columnas de similitud al DataFrame
      df['db_most_similar_cluster'] = db_most_similar_cluster
      df['km_most_similar_cluster'] = km_most_similar_cluster
      
      # Crear un DataFrame de correlaciones
      correlation_df = pd.DataFrame({
          'cluster_column': cluster_columns,
          'db_corr': correlation_db,
          'km_corr': correlation_km
      })
      
      return df, correlation_df

  def get_last_increasing_inflexion_point(self, data, bins=15):
      """
      Encuentra el último punto de inflexión creciente en un histograma.

      Args:
          data (list or array-like): Los datos originales.
          bins (int): Número de bins para el histograma.

      Returns:
          float: El valor del borde del bin donde ocurre el último punto de inflexión creciente.
      """
      # Generar el histograma
      hist, bin_edges = np.histogram(data, bins=bins)
      
      # Calcular la derivada de la frecuencia del histograma
      first_derivative = np.diff(hist)
      
      # Encontrar los índices donde la derivada es positiva (crecimiento)
      increasing_points = np.where(first_derivative > 0)[0]
      
      if len(increasing_points) == 0:
          raise ValueError("No se encontró un punto de inflexión creciente.")
      
      # Tomar el último punto de inflexión creciente
      last_increasing_index = increasing_points[-1]
      
      # Obtener el borde derecho del bin correspondiente al último punto de inflexión creciente
      last_inflexion_point = bin_edges[last_increasing_index + 1]
      
      return last_inflexion_point


  def get_first_decreasing_inflexion_point(self, data, bins=10):
      """
      Encuentra el primer punto de inflexión decreciente en un histograma.

      Args:
          data (list or array-like): Los datos originales.
          bins (int): Número de bins para el histograma.

      Returns:
          float: El valor del borde del bin donde ocurre el primer punto de inflexión decreciente.
      """
      # Generar el histograma
      hist, bin_edges = np.histogram(data, bins=bins)
      
      # Calcular la derivada de la frecuencia del histograma
      first_derivative = np.diff(hist)
      
      # Encontrar los índices donde la derivada es negativa (decrecimiento)
      decreasing_points = np.where(first_derivative < 0)[0]
      
      if len(decreasing_points) == 0:
          raise ValueError("No se encontró un punto de inflexión decreciente.")
      
      # Tomar el primer punto de inflexión decreciente
      first_decreasing_index = decreasing_points[0]
      
      # Obtener el borde derecho del bin correspondiente al primer punto de inflexión decreciente
      first_inflexion_point = bin_edges[first_decreasing_index + 1]
      
      return first_inflexion_point


  def add_active_clusters(self, df, cluster_prefix='cluster_', new_column='active_clusters'):
      """
      Agrega una nueva columna al DataFrame que contiene una lista de clusters activos (valor = 1) para cada fila.
      
      Parámetros:
      - df (pd.DataFrame): El DataFrame de entrada que contiene las columnas de clusters.
      - cluster_prefix (str, opcional): Prefijo que identifica las columnas de clusters. Por defecto es 'cluster_'.
      - new_column (str, opcional): Nombre de la nueva columna a crear. Por defecto es 'active_clusters'.
      
      Retorna:
      - pd.DataFrame: El DataFrame original con la nueva columna agregada.
      """
      
      # Identificar las columnas que corresponden a los clusters
      cluster_columns = [col for col in df.columns if col.startswith(cluster_prefix)]
      
      # Verificar si se encontraron columnas de clusters
      if not cluster_columns:
          raise ValueError(f"No se encontraron columnas que comiencen con el prefijo '{cluster_prefix}'.")
      
      # Extraer los números de los clusters y crear un mapeo de columna a número
      cluster_mapping = {}
      for col in cluster_columns:
          try:
              # Asumiendo que el nombre de la columna es algo como 'cluster_3.0'
              cluster_number = col.split('_')[1].split('.')[0]
              cluster_mapping[col] = int(cluster_number)
          except (IndexError, ValueError):
              raise ValueError(f"The format of column '{col}' is invalid. Expected 'cluster_<number>.0'.")
      
      # Utilizar una lista por comprensión para obtener los clusters activos por fila
      df[new_column] = [
          [cluster_mapping[col] for col in cluster_columns if row[col] == 1]
          for _, row in df.iterrows()
      ]
      
      return df


  def convert_list_to_string(self, df, list_column, sorted=False, delimiter=',', new_key_column='clusters_key'):
      """
      Convierte una columna de listas en cadenas de texto (opcionalmente ordenadas) para usarlas como llaves de unión.

      Parámetros:
      - df (pd.DataFrame): DataFrame de entrada.
      - list_column (str): Nombre de la columna que contiene las listas.
      - sorted (bool): Si se debe ordenar la lista antes de convertirla a cadena.
      - delimiter (str): Delimitador para concatenar los elementos de la lista.
      - new_key_column (str): Nombre de la nueva columna que contendrá las cadenas.

      Retorna:
      - pd.DataFrame: DataFrame con la nueva columna de llaves.
      """
      if list_column not in df.columns:
          raise KeyError(f"Column '{list_column}' does not exist in the DataFrame")

      if sorted:
          df[new_key_column] = df[list_column].apply(lambda x: delimiter.join(map(str, sorted(x))))
      else:
          df[new_key_column] = df[list_column].apply(lambda x: delimiter.join(map(str, x)))
      return df

  def get_clusters_importantes(self, df_clusterizado):
    """Identifica los clusters más representativos de un DataFrame."""

    if 'clusters_list' not in df_clusterizado.columns:
        raise KeyError("DataFrame must contain the 'clusters_list' column")

    try:
      df_clusterizado_diff = df_clusterizado[['clusters_list']].drop_duplicates()
      df_clusterizado_diff['n_ls'] = df_clusterizado_diff.apply(lambda x: len(x['clusters_list']), axis=1)
    except Exception as exc:
      logger.exception("Error procesando clusters_list: %s", exc)
      return df_clusterizado
    df_clusterizado_diff_sub = df_clusterizado_diff

    df_expanded = self.expandir_clusters_binario(df_clusterizado_diff_sub,'clusters_list','cluster_')
    cluster_cols = [x for x in df_expanded.columns if 'cluster_' in x]

    sample_size = int(np.sqrt(df_clusterizado_diff_sub.shape[0])*3)
    eps_su = self.get_eps_multiple_groups_opt(df_expanded[cluster_cols].sample(sample_size))

    # Obtener clusters
    df_clustered, _ = self.apply_clustering_and_similarity(df_expanded, cluster_cols,
                                                                dbscan_params={'eps': eps_su, 'min_samples': 2},
                                                                kmeans_params={'n_clusters': 6, 'random_state': 42})


    try:
      df_custers__vc = df_clustered[['db_labels','km_labels']].value_counts()
      values_up = np.sqrt(df_custers__vc.head(1)).values[0]
      df_custers__vc_ = df_custers__vc[df_custers__vc>values_up]
    except Exception as exc:
      logger.exception("Error calculando clusters importantes: %s", exc)
      return df_clusterizado

    # Extraer clsuters en el core
    pd_cluster_sun = []
    for db_, km_ in df_custers__vc_.index:

      df_clustered_subcluster = df_clustered[(df_clustered['db_labels']==db_)&(df_clustered['km_labels']==km_)].copy()#[cluster_cols]

      df_lista_ms_min = df_clustered_subcluster[df_clustered_subcluster['n_ls']==df_clustered_subcluster['n_ls'].min()]['clusters_list']
      lista_menor = df_lista_ms_min.values[0]

      df_clustered_subcluster_suma_clust = df_clustered_subcluster[cluster_cols].sum(axis=0)

      df_cl_inf = self.get_last_increasing_inflexion_point(df_clustered_subcluster_suma_clust)
      clusters_comunes = df_clustered_subcluster_suma_clust[df_clustered_subcluster_suma_clust>df_cl_inf]

      df_cl_value_counts = df_clustered[list(clusters_comunes.index)].value_counts()


      grupo_s_ = df_cl_value_counts[df_cl_value_counts>self.get_first_decreasing_inflexion_point(df_cl_value_counts)].reset_index()
      try:
          grupo_s_ = grupo_s_.rename(columns={0:'count'})
      except:
         pass
      grupo_s_ = self.add_active_clusters(grupo_s_).drop(columns='count')

      cols_keys = list(grupo_s_.columns[:-1])
      df_clustered_subcluster_agg = df_clustered_subcluster.merge(grupo_s_, on=cols_keys, how='left')


      pd_cluster_sun.append(df_clustered_subcluster_agg)

    df_clustered_subcluster_agg_all = pd.concat(pd_cluster_sun)

    df_clusterizado = self.convert_list_to_string(
        df_clusterizado, 'clusters_list', sorted=False, delimiter=',', new_key_column='clusters_key'
    )

    df_clustered_subcluster_agg_all = self.convert_list_to_string(
        df_clustered_subcluster_agg_all,
        'clusters_list',
        sorted=False,
        delimiter=',',
        new_key_column='clusters_key'
    )

    try:
        df_clusterizado_add = df_clusterizado.merge(
            df_clustered_subcluster_agg_all[['clusters_key', 'active_clusters']],
            on='clusters_key',
            how='left'
        )
    except Exception as exc:
        logger.exception("Error combining clusters: %s", exc)
        return df_clusterizado

    return df_clusterizado_add.drop(columns='clusters_key')


  def balance_lists_n_clusters(
    self,
    records: Sequence[Sequence[Any]],
    n_clusters: int | None = None,
    *,
    max_iter: int = 20_000,
    restarts: int = 4,
    T0: float = 1.0,
    alpha: float = 0.999,
    seed: int | None = None
  ) -> List[Any]:
    """
    Asigna **un único valor por fila** optimizando dos objetivos con *peso idéntico*:

    • |distinct - n_clusters| →   acercarse al nº deseado de clusters  
      (si `n_clusters` es `None`, se toma el mínimo posible de forma natural).

    • Desbalance absoluto    →   Σ |c_v – ideal| / n, donde `ideal = n / k`.

    Parameters
    ----------
    records : Sequence[Sequence[Any]]
      Lista de listas con los valores candidatos por fila.
    n_clusters : int | None
      Número de clusters deseado. Si `None` se minimiza automáticamente.
    max_iter : int
      Iteraciones de Simulated Annealing por reinicio.
    restarts : int
      Número de reinicios aleatorios.
    T0, alpha : float
      Temperatura inicial y factor de enfriamiento.
    seed : int | None
      Para reproducibilidad.
    """
    rng = random.Random(seed)
    records = [row if row else [-1] for row in records]
    n = len(records)

    # Utilidades internas -------------------------------------------------
    def imbalance(cnt: Counter):
      k = len(cnt)
      if k == 0:
        return 1.0
      ideal = n / k
      return sum(abs(c - ideal) for c in cnt.values()) / n

    def score(assign: List[Any]) -> float:
      cnt = Counter(assign)
      k = len(cnt)
      if n_clusters is None:
        cluster_term = k / n  # minimizar k
      else:
        cluster_term = abs(k - n_clusters) / n
      return cluster_term + imbalance(cnt)

    def neighbour(assign: List[Any]) -> List[Any]:
      """Mueve una fila a otra opción válida (aleatorio)."""
      i = rng.randrange(n)
      row = records[i]
      cur = assign[i]
      alt = [v for v in row if v != cur]
      if not alt:  # fila sin alternativas
        return assign
      new = assign[:]
      new[i] = rng.choice(alt)
      return new

    # Inicialización razonable ------------------------------------------
    val_rows = defaultdict(list)
    for idx, row in enumerate(records):
      for v in row:
        val_rows[v].append(idx)

    remaining = set(range(n))
    chosen: List[Any] = []
    while remaining:
      best = max(val_rows, key=lambda v: len(set(val_rows[v]) & remaining))
      chosen.append(best)
      remaining -= set(val_rows[best])

    if n_clusters is not None and len(chosen) < n_clusters:
      extras = sorted(
        (v for v in val_rows if v not in chosen),
        key=lambda v: -len(val_rows[v])
      )
      chosen.extend(extras[: n_clusters - len(chosen)])

    def initial_assignment() -> List[Any]:
      cnt: Counter = Counter()
      assign: List[Any] = [None] * n
      for i, row in enumerate(records):
        opts = [v for v in row if v in chosen] or row
        v = min(opts, key=lambda x: (cnt[x], x))
        assign[i] = v
        cnt[v] += 1
      return assign

    # Simulated Annealing -------------------------------------------------
    best_global, best_score = None, float("inf")
    for _ in range(restarts):
      cur = initial_assignment()
      cur_score = score(cur)
      best_local, best_local_score = cur[:], cur_score
      T = T0
      for _ in range(max_iter):
        nxt = neighbour(cur)
        if nxt is cur:
          continue
        nxt_score = score(nxt)
        accept = nxt_score < cur_score or rng.random() < math.exp((cur_score - nxt_score) / T)
        if accept:
          cur, cur_score = nxt, nxt_score
          if cur_score < best_local_score:
            best_local, best_local_score = cur[:], cur_score
        T *= alpha
      if best_local_score < best_score:
        best_global, best_score = best_local, best_local_score

    return best_global

  def max_prob_clusters(
    self,
    records: Sequence[Sequence[Any]],
    probs: Mapping[Any, float],
    n_clusters: int | None = None,
    *,
    max_iter: int = 20_000,
    restarts: int = 4,
    T0: float = 1.0,
    alpha: float = 0.999,
    seed: int | None = None
  ) -> List[Any]:
    """
    Selecciona **un valor por fila** cumpliendo:
      • Si `n_clusters` es `None`  →  minimiza el nº de valores distintos.
      • Si `n_clusters` es un entero:
          – intenta devolver EXACTAMENTE ese nº de clusters, maximizando la suma de probabilidades.
          – si es imposible, usa el valor factible más próximo (`k_min` o `k_max`).
    """
    rng = random.Random(seed)
    n = len(records)
    records = [row if row else [None] for row in records]

    # Paso 1: greedy set-cover para k_min -------------------------------
    value_rows = defaultdict(set)
    for i, row in enumerate(records):
      for v in row:
        value_rows[v].add(i)

    remaining = set(range(n))
    S: set[Any] = set()
    while remaining:
      best = max(
        value_rows,
        key=lambda v: (len(value_rows[v] & remaining), probs.get(v, 0.0))
      )
      S.add(best)
      remaining -= value_rows[best]

    k_min = len(S)
    k_max = len(value_rows)

    # Paso 2: determinar k_target ---------------------------------------
    if n_clusters is None:
      k_target = k_min
    else:
      k_target = max(k_min, min(n_clusters, k_max))

    if k_target > k_min:
      extras = sorted(
        (v for v in value_rows if v not in S),
        key=lambda v: probs.get(v, 0.0),
        reverse=True
      )
      S.update(extras[: k_target - k_min])

    S = set(list(S)[:k_target])  # asegura |S| == k_target

    # Paso 3: asignación greedy -----------------------------------------
    assign: List[Any] = []
    for row in records:
      opts = [v for v in row if v in S]
      if not opts:
        best = max(row, key=lambda v: probs.get(v, 0.0))
        if best not in S and len(S) == k_target:
          worst = min(S, key=lambda v: probs.get(v, 0.0))
          S.remove(worst)
          S.add(best)
        opts = [v for v in row if v in S]
      assign.append(max(opts, key=lambda v: probs.get(v, 0.0)))

    # Paso 4: Simulated Annealing ---------------------------------------
    B = n + 1  # peso que penaliza cambiar k

    def cost(ass: List[Any]) -> float:
      k = len(set(ass))
      return abs(k - k_target) * B - sum(probs.get(v, 0.0) for v in ass)

    def neighbour(ass: List[Any]) -> List[Any]:
      i = rng.randrange(n)
      cur_row = records[i]
      cur_v = ass[i]
      alt = [v for v in cur_row if v in S and v != cur_v]
      if not alt:
        return ass
      new = ass[:]
      new[i] = rng.choice(alt)
      return new

    best_global, best_c = assign[:], cost(assign)
    for _ in range(restarts):
      cur, cur_c = assign[:], best_c
      T = T0
      for _ in range(max_iter):
        nxt = neighbour(cur)
        if nxt is cur:
          T *= alpha
          continue
        nxt_c = cost(nxt)
        if nxt_c < cur_c or rng.random() < math.exp((cur_c - nxt_c) / T):
          cur, cur_c = nxt, nxt_c
          if cur_c < best_c:
            best_global, best_c = cur[:], cur_c
        T *= alpha

    return best_global

  def labels(self, df, df_reres, n_clusters = None,
             include_summary_cluster=False,
             balanced=False):
    """Assign cluster labels by selecting and applying rules.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to which the cluster labels will be assigned.
    df_reres : list[pd.DataFrame]
        List of DataFrames representing candidate rules (regions).
    n_clusters : int or None, optional
        Desired number of clusters. If ``None`` the number is inferred.
    include_summary_cluster : bool, default False
        When ``True`` additional summary metrics per cluster are kept in the
        output.
    balanced : bool, default False
        Use balanced assignment of cluster labels instead of probability based
        assignment.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple ``(df_datos_clusterizados, df_clusters_descripcion)`` where the
        first element contains the original data with an added ``cluster``
        column and the second describes each cluster's effectiveness and
        support metrics.
    """
    
    lista_reglas = copy.deepcopy(df_reres)

    # Asignar IDs únicos a las reglas
    lista_reglas = self.asignar_ids_unicos(lista_reglas)

    # Eliminar reglas redundantes
    nueva_lista_ = self.combinar_dataframes_por_columnas(lista_reglas)
    df_reglas_importantes = self.eliminar_reglas_redundantes(nueva_lista_)

    df_reglas_importantes = df_reglas_importantes.reset_index()
    df_reglas_importantes.rename(columns={'index': 'cluster'}, inplace=True)

    # Asignar clusters a los datos utilizando las reglas importantes
    df_datos_clusterizados, df_clusters_descripcion = self.asignar_clusters_a_datos(df, df_reglas_importantes)

    # if include_summary_cluster:
    #   df_datos_clusterizados = self.get_clusters_importantes(df_datos_clusterizados)
    
    records = df_datos_clusterizados['clusters_list'].tolist()
    probas = {int(a):float(b) for a, b in 
              df_clusters_descripcion[['cluster','cluster_ef_sample']].drop_duplicates().values}
    
    # Estandarizar la efectividad
    df_clusters_descripcion['cluster_ef_sample'] /= df_clusters_descripcion['cluster_ef_sample'].max()
    df_clusters_descripcion['cluster_ef_sample'] = abs(df_clusters_descripcion['cluster_ef_sample'])

    # Tratamiento de los clusters para agregar n_cluster
    if balanced:
      labels = self.balance_lists_n_clusters(records=records, 
                                             n_clusters=n_clusters, seed=1)
    else:
      labels = self.max_prob_clusters(records=records, probs=probas, 
                                      n_clusters=n_clusters, seed=1)
    
    if not include_summary_cluster:
       rem_cols = ['n_clusters','ponderadores_list','ponderador_mean']
       df_datos_clusterizados = df_datos_clusterizados.drop(columns=rem_cols)
    
    df_datos_clusterizados['cluster'] = labels

    return df_datos_clusterizados, df_clusters_descripcion

  
  def get_corr_clust(self, df_datos_clusterizados):
      """Compute the correlation matrix between cluster indicator columns.

      Parameters
      ----------
      df_datos_clusterizados : pd.DataFrame
          DataFrame containing a ``clusters_list`` column with cluster
          assignments for each observation.

      Returns
      -------
      pd.DataFrame
          Correlation matrix of the binary cluster indicator columns.
      """

      df_clusterizado_diff = df_datos_clusterizados[['clusters_list']].drop_duplicates()
      df_clusterizado_diff['n_ls'] = df_clusterizado_diff.apply(lambda x: len(x['clusters_list']), axis=1)
      df_expanded = self.expandir_clusters_binario(df_clusterizado_diff,'clusters_list','cluster_')
      cluster_cols = [x for x in df_expanded.columns if 'cluster_' in x]

      df_corr = df_expanded[cluster_cols].corr()

      return df_corr

  def obtener_clusters(self, df_clust, cluster_objetivo, n=5, direccion='ambos'):
      """Return clusters with highest or lowest correlation to a target.

      Parameters
      ----------
      df_clust : pd.DataFrame
          Correlation matrix between clusters.
      cluster_objetivo : str
          Name of the reference cluster.
      n : int, default 5
          Number of clusters to return.
      direccion : {'arriba', 'abajo', 'ambos', 'bottom'}, default 'ambos'
          Direction of correlation to consider: ``'arriba'`` most positive,
          ``'abajo'`` most negative, ``'ambos'`` both extremes, ``'bottom'``
          closest to zero.

      Returns
      -------
      pd.DataFrame
          DataFrame with selected clusters and their correlation to the target
          cluster.
      """
      corr_matrix = self.get_corr_clust(df_clust)

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
          raise ValueError("Parameter 'direccion' must be 'arriba', 'abajo', 'ambos' or 'bottom'.")
      
      return top_n.sort_values(ascending=False)

