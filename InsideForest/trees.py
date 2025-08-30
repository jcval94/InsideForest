import re
import pandas as pd
import numpy as np
import logging

from sklearn.tree import _tree
from tqdm import tqdm
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

class Trees:

  def __init__(self, lang='python',n_sample_multiplier=2, ef_sample_multiplier=3,
               percentil=95, low_frac=0.05):
    self.lang = lang
    self.n_sample_multiplier = n_sample_multiplier
    self.ef_sample_multiplier = ef_sample_multiplier
    self.percentil = percentil
    self.low_frac = low_frac


  def transform_tree_structure(self, tree_str):
    """Convert Spark-style tree string to scikit-learn format.

    Parameters
    ----------
    tree_str : str
        Representation of the ensemble produced by ``toDebugString`` in
        PySpark.

    Returns
    -------
    dict
        Mapping from tree name to a string resembling the output of
        ``sklearn.tree.export_text``.
    """
    # Split the string into individual trees
    trees = tree_str.strip().split('\n  Tree')
    tree_dict = {}

    for i, tree in enumerate(trees):
      if i == 0:
        # Ignore first line if it's the model description
        if tree.startswith('RandomForestClassificationModel'):
          continue

      if not tree.strip():
        continue

      lines = tree.strip().split('\n')
      tree_name = "Tree " + lines[0].split('(')[0].strip()

      # Determine the number of spaces to remove based on lines after "Tree"
      min_indent = min((len(line) - len(line.lstrip())) for line in lines[1:] if line.strip())

      # Initialize list to hold the transformed structure
      transformed_lines = []

      for line in lines[1:]:

        # Remove the minimum indentation
        adjusted_line = line[min_indent:]

        # Replace leading spaces with "|   "

        for i, s in enumerate(adjusted_line):
            if s != ' ':
                break

        indent = i * '|   '
        stripped_line = adjusted_line.strip()

        # Replace "If" and "Else" with "|---"
        if stripped_line.startswith('If') or stripped_line.startswith('Else'):
          condition = stripped_line.split('(')[1].split(')')[0]
          transformed_lines.append(f"{indent}|--- {condition}")
        elif stripped_line.startswith('Predict'):
          prediction = stripped_line.split(': ')[1]
          transformed_lines.append(f"{indent}|--- class: {prediction}")

      # Save the transformed tree in the dictionary
      tree_dict[tree_name] = '\n'.join(transformed_lines).replace('feature ','feature_')

    return tree_dict

  def get_path(self, estructura_iter):
    for i, valor in enumerate(estructura_iter):
      if not(('value: ' in valor) or ('class: ' in valor)):
        continue
      camino = [estructura_iter[0:i], estructura_iter[i]]
      estructura_iter.pop(i)
      estructura_iter.pop(i-1)
      return estructura_iter, camino


  def get_rangos(self, regr, data1, verbose=0, percentil=None, low_frac=None,
                 n_jobs=1, random_state=0):
    """Extract tree rules and importances.

    Parameters
    ----------
    regr : estimator
        Trained forest model.
    data1 : pandas.DataFrame
        Dataset used to replace feature names.
    verbose : int, default 0
        Verbosity level.
    percentil : float, optional
        Percentile to filter tree leaves.
    low_frac : float, optional
        Fraction of values below the percentile to sample.
    n_jobs : int, default 1
        Number of parallel jobs. Uses ``prefer="threads"`` to avoid
        pickling. Parallel execution only pays off when each tree is
        expensive to process; for small forests sequential mode reduces
        overhead.
    random_state : int, default 0
        Seed for random operations.
    """

    # This function may be slow; add tqdm to the main loop.

    if percentil is None:
      percentil = self.percentil
    if low_frac is None:
      low_frac = self.low_frac

    if self.lang == 'pyspark':
      arboles_estimadores = self.transform_tree_structure(regr.toDebugString)
      arboles_estimadores = [a for a in arboles_estimadores.values()]
    else:
      arboles_estimadores = regr.estimators_

    # Build feature name replacement once
    feature_names = list(map(str, data1.columns))
    pattern = re.compile(r'\bfeature_(\d+)\b')
    val_pat = re.compile(r'value:\s*\[?([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\]?', re.I)
    cls_pat = re.compile(r'class:\s*(\S+)', re.I)
    leaf_pat = re.compile(r'(?:value|class):', re.I)

    def process_tree(n_estimador, arbol_individual):
      if self.lang == 'pyspark':
        r = pattern.sub(lambda m: feature_names[int(m.group(1))], arbol_individual)
        estructura = r.split('\n')
        estructura_iter = estructura[:]
        paths = []
        leaves = []
        for valor in estructura:
          if not leaf_pat.search(valor):
            continue
          estructura_iter, path_ = self.get_path(estructura_iter)
          estructura_rep = [v.count('|') for v in path_[0]]
          if len(estructura_rep) != len(set(estructura_rep)):
            seen = set()
            new_path = []
            for elem in reversed(path_[0]):
              bc = elem.count('|')
              if bc not in seen:
                new_path.append(elem)
                seen.add(bc)
            new_path.reverse()
            path_[0] = new_path
          paths.append(path_[0])
          leaves.append(path_[1] if len(path_) > 1 else '')
        valores = []
        for texto in leaves:
          m = val_pat.search(texto)
          if m:
            valores.append(float(m.group(1).replace('_', '')))
          else:
            mc = cls_pat.search(texto)
            if mc:
              try:
                valores.append(float(mc.group(1).replace('_', '')))
              except ValueError:
                continue
      else:
        t = arbol_individual.tree_
        paths = []
        valores = []
        def recurse(node, path):
          if t.feature[node] != _tree.TREE_UNDEFINED:
            nombre = feature_names[t.feature[node]]
            umbral = t.threshold[node]
            recurse(t.children_left[node], path + [f"{nombre} <= {umbral:.6f}"])
            recurse(t.children_right[node], path + [f"{nombre} > {umbral:.6f}"])
          else:
            paths.append(path)
            val = t.value[node]
            if hasattr(arbol_individual, 'n_classes_'):
              class_idx = int(np.argmax(val))
              valores.append(float(class_idx))
            else:
              valores.append(float(val.ravel()[0]))
        recurse(0, [])
      if not valores:
        return pd.DataFrame(columns=['Regla', 'Importancia', 'N_regla', 'N_arbol', 'Va_Obj_minima'])
      rng = np.random.RandomState(random_state + n_estimador)
      if percentil is None:
        percent_ = np.nan
        estructuras_maximizadoras = [[pa, val] for pa, val in zip(paths, valores)]
      else:
        percent_ = np.percentile(valores, percentil)
        high_idx = [i for i, val in enumerate(valores) if val >= percent_]
        low_idx = [i for i, val in enumerate(valores) if val < percent_]
        sample_size = int(len(low_idx) * low_frac)
        sampled_low_idx = rng.choice(low_idx, size=sample_size, replace=False).tolist() if sample_size > 0 else []
        selected_idx = high_idx + sampled_low_idx
        estructuras_maximizadoras = [[paths[i], valores[i]] for i in selected_idx]

      importanc = []
      for n_path, (path, _) in enumerate(estructuras_maximizadoras):
        importanc += [
          [cond, 2 / (i + 2), n_path, n_estimador]
          for i, cond in enumerate(path)
        ]

      asdf = pd.DataFrame(importanc, columns=['Regla', 'Importancia', 'N_regla', 'N_arbol'])
      asdf['Va_Obj_minima'] = percent_
      return asdf

    def _run_parallel(items, func, n_jobs, use_tqdm, desc):
      it = tqdm(items, desc=desc) if use_tqdm else items
      try:
        if n_jobs == 1:
          return [func(i, a) for i, a in enumerate(it)]
        return Parallel(n_jobs=n_jobs, prefer="threads")(delayed(func)(i, a) for i, a in enumerate(it))
      except Exception:
        return [func(i, a) for i, a in enumerate(items)]

    resultados = _run_parallel(arboles_estimadores, process_tree, n_jobs, verbose > 0, "Processing trees")
    resultados = [r for r in resultados if not r.empty]
    return pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame(columns=['Regla', 'Importancia', 'N_regla', 'N_arbol', 'Va_Obj_minima'])


  def get_fro(self, df_full_arboles):
    if df_full_arboles.empty:
      return df_full_arboles.assign(feature=[], operador=[], rangos=[])

    # Regular expression that captures numbers with optional underscores and scientific notation
    num = r'[-+]?(?:\d+(?:_\d+)*)?(?:\.\d+(?:_\d+)*)?(?:e[-+]?\d+)?'
    pattern = re.compile(rf'^(\S+)\s*(<=|>=|<|>)\s*({num})$', re.I)

    def parse_regla(regla):
        match = pattern.search(regla.replace(' ', ''))
        if match:
            feature = match.group(1)
            operador = match.group(2)
            rangos = float(match.group(3).replace('_', ''))
            return feature, operador, rangos
        else:
            return None, None, None

    # Apply the parsing function to each 'Regla' row
    df_full_arboles[['feature', 'operador', 'rangos']] = df_full_arboles['Regla'].apply(
        lambda x: parse_regla(x)
    ).apply(pd.Series)

    # Remove rows that could not be parsed
    df_full_arboles = df_full_arboles.dropna(subset=['operador', 'rangos'])

    return df_full_arboles


  def get_summary(self, data1, df_full_arboles, var_obj, verbose=0):
    agrupacion = pd.pivot_table(
      df_full_arboles,
      index=['N_regla', 'N_arbol', 'feature', 'operador'],
      values=['rangos', 'Importancia'],
      aggfunc=['min', 'max', 'mean']
    )

    agrupacion_min = agrupacion['min'].reset_index()
    agrupacion_min = agrupacion_min[agrupacion_min['operador'] == '<=']
    agrupacion_max = agrupacion['max'].reset_index()
    agrupacion_max = agrupacion_max[agrupacion_max['operador'] == '>']
    agrupacion_mean = agrupacion['mean'].reset_index()

    agrupacion = pd.concat([agrupacion_min, agrupacion_max]).sort_values(['N_arbol', 'N_regla'])
    top_100_ramas = agrupacion.N_arbol.unique()[:100]

    reglas = []

    # Add tqdm over top_100_ramas to show progress
    for arbol_num in tqdm(top_100_ramas, disable=(verbose == 0), desc="Processing branches"):
      # Keep this log but make it conditional
      if arbol_num % 50 == 0 and verbose == 1:
        logger.info(f"Processing tree branch: {arbol_num}")

      ag_arbol = agrupacion[(agrupacion['N_arbol'] == arbol_num)]
      for regla_num in ag_arbol.N_regla.unique():
        data1_ = data1.copy()
        ag_regla = ag_arbol[(ag_arbol['N_regla'] == regla_num)]
        men_ = ag_regla[(ag_regla['operador'] == '<=')][['feature', 'rangos']].values
        may_ = ag_regla[(ag_regla['operador'] == '>')][['feature', 'rangos']].values
        if len(men_) > 0:
          for col, valor in men_:
            data1_ = data1_.loc[data1_[col] <= valor, :]
            for col2, valor2 in may_:
              data1_ = data1_.loc[data1_[col2] > valor2, :]
        else:
          for col, valor in may_:
            data1_ = data1_.loc[data1_[col] > valor, :]
            for col2, valor2 in men_:
              data1_ = data1_.loc[data1_[col2] <= valor2, :]

        ag_regla_copy = ag_regla.copy()
        ag_regla_copy.loc[:, 'n_sample'] = len(data1_)
        ag_regla_copy.loc[:, 'ef_sample'] = data1_[var_obj].mean()
        reglas.append(ag_regla_copy)

    agrupacion = pd.concat(reglas)
    agrupacion = agrupacion.sort_values(by=['ef_sample', 'n_sample'], ascending=False)
    return agrupacion

  def get_summary_optimizado(self, data1, df_full_arboles, var_obj,
                              no_branch_lim=None, verbose=0, n_jobs=1):
      """Resume las reglas evaluando las condiciones de manera vectorizada.

      Parameters
      ----------
      no_branch_lim : int | None, optional
          Maximum number of trees to process. If ``None`` all available
          trees are used.
      """

      # 1) Compute pivot summarizing by N_regla, N_arbol, feature, operator
      agrupacion = pd.pivot_table(
          df_full_arboles,
          index=['N_regla', 'N_arbol', 'feature', 'operador'],
          values=['rangos', 'Importancia'],
          aggfunc=['min', 'max', 'mean'],
      )

      # 2) Extract min and max values depending on the operator
      agrupacion_min = agrupacion['min'].reset_index()
      agrupacion_min = agrupacion_min[agrupacion_min['operador'] == '<=']
      agrupacion_max = agrupacion['max'].reset_index()
      agrupacion_max = agrupacion_max[agrupacion_max['operador'] == '>']

      # 3) Concatenate and sort
      agrupacion = pd.concat([agrupacion_min, agrupacion_max]).sort_values(['N_arbol', 'N_regla'])

      # 4) Select trees to process
      top_100_arboles = agrupacion['N_arbol'].unique()
      if no_branch_lim is not None:
          top_100_arboles = top_100_arboles[:no_branch_lim]

      # Convertir datos a matriz NumPy y preparar mapeo de columnas
      X = data1.to_numpy()
      col_to_idx = {col: i for i, col in enumerate(data1.columns)}
      y = data1[var_obj].to_numpy()

      def _process_tree(arbol_num):
          ag_arbol = agrupacion[agrupacion['N_arbol'] == arbol_num]
          reglas_info = []

          for regla_num in ag_arbol['N_regla'].unique():
              ag_regla = ag_arbol[ag_arbol['N_regla'] == regla_num]
              men_ = ag_regla[ag_regla['operador'] == '<='][['feature', 'rangos']].values
              may_ = ag_regla[ag_regla['operador'] == '>'][['feature', 'rangos']].values

              le_idx = np.array([col_to_idx[c] for c in men_[:, 0]]) if len(men_) else np.array([], dtype=int)
              le_val = men_[:, 1].astype(float) if len(men_) else np.array([], dtype=float)
              gt_idx = np.array([col_to_idx[c] for c in may_[:, 0]]) if len(may_) else np.array([], dtype=int)
              gt_val = may_[:, 1].astype(float) if len(may_) else np.array([], dtype=float)

              reglas_info.append((ag_regla.copy(), le_idx, le_val, gt_idx, gt_val))

          if not reglas_info:
              return pd.DataFrame()

          masks = []
          for _, le_idx, le_val, gt_idx, gt_val in reglas_info:
              conds = []
              if le_idx.size:
                  conds.append(X[:, le_idx] <= le_val)
              if gt_idx.size:
                  conds.append(X[:, gt_idx] > gt_val)
              if conds:
                  mask = np.logical_and.reduce(np.concatenate(conds, axis=1), axis=1)
              else:
                  mask = np.ones(X.shape[0], dtype=bool)
              masks.append(mask)

          mask_matrix = np.vstack(masks)
          n_sample = mask_matrix.sum(axis=1)
          sums = mask_matrix @ y
          ef_sample = np.divide(sums, n_sample, out=np.zeros_like(sums, dtype=float), where=n_sample > 0)

          res = []
          for (df_regla, _, _, _, _), ns, ef in zip(reglas_info, n_sample, ef_sample):
              df_regla['n_sample'] = ns
              df_regla['ef_sample'] = ef
              res.append(df_regla)

          return pd.concat(res, ignore_index=True)

      it = tqdm(top_100_arboles, disable=(verbose == 0), desc="Procesando ramas") if verbose else top_100_arboles
      try:
          if n_jobs == 1:
              resultados = [_process_tree(a) for a in it]
          else:
              resultados = Parallel(n_jobs=n_jobs)(delayed(_process_tree)(a) for a in it)
      except Exception:
          resultados = [_process_tree(a) for a in top_100_arboles]

      resultados = [r for r in resultados if r is not None and not r.empty]
      if not resultados:
          return pd.DataFrame(columns=['N_regla','N_arbol','feature','operador','rangos','Importancia','n_sample','ef_sample'])

      resultado = pd.concat(resultados, ignore_index=True)
      resultado = resultado.sort_values(by=['ef_sample', 'n_sample'], ascending=False)
      return resultado



  def get_rect_coords(self, df):
    limits = {}
    for i, row in df.iterrows():
        feature = row['feature']
        operador = row['operador']
        rango = row['rangos']
        if operador == '<=':
            if feature not in limits:
                limits[feature] = [float('-inf'), rango]
            else:
                limits[feature][1] = min(limits[feature][1], rango)
        elif operador == '>':
            if feature not in limits:
                limits[feature] = [rango, float('inf')]
            else:
                limits[feature][0] = max(limits[feature][0], rango)
    rectangle_coordinates = [(key, limits[key]) for key in sorted(limits.keys())]
    return rectangle_coordinates

  def rect_l_to_df(self, separacion_dim, llave):
    registros__ = []
    for i, sublista in enumerate(separacion_dim[llave]):
        rectangulo_m = sublista[:-3]
        ponde_ = sublista[-3]
        effe_ = sublista[-2]
        nsamp_ = sublista[-1]

        for rectangulo, intervalo in rectangulo_m:
            linf, lsup = intervalo
            registros__.append([i + 1, rectangulo, linf, lsup, ponde_[1], effe_[1], nsamp_[1]])
    df = pd.DataFrame(registros__, columns=['rectangulo', 'dimension', 'linf', 'lsup','ponderador','ef_sample','n_sample'])
    return df

  def generate_key(self, r):
    r = r[1]
    ndims_ = len(r)
    dimensiones__ = '*'.join(sorted([r[i][0] for i in range(ndims_)][:-1]))
    return dimensiones__

  def get_dfs_dim(self, rectangles_):
    l_rectangles_ = list(rectangles_.values())
    llaves = [(self.generate_key(r)) for  r in (rectangles_.items())]
    
    llaves_unicas = list(set(llaves))
    separacion_dim = {lun:[l_rectangles_[i] for i in
                           [i for i, llave in enumerate(llaves) if llave==lun]]
    for lun in llaves_unicas}
    separacion_dim = [self.rect_l_to_df(separacion_dim, k) for k, v in 
                      separacion_dim.items()]
    
    return separacion_dim

  def extract_rectangles(self, df_summ):
    
    grouped = df_summ[(df_summ['n_sample']>=0)&
                      (df_summ['ef_sample']>=0)].groupby(['N_arbol', 'N_regla'])

    rectangles_ = {}
    for name, group in grouped:
        rectangles_[name] = self.get_rect_coords(group)

    try:
      agrupacion_media = grouped.mean()
    except:
      agrupacion_media = grouped[['n_sample', 'ef_sample']].mean()

    agrupacion_media['ponderacion'] = ((agrupacion_media['n_sample']+1)\
    /(agrupacion_media['n_sample'].max()+1))*self.n_sample_multiplier\
      +((agrupacion_media['ef_sample']+1)/(agrupacion_media['ef_sample'].max()+1))*self.ef_sample_multiplier

    agrupacion_media.sort_values('ponderacion')
    for k in rectangles_.keys():
      rectangles_[k] += [('ponderador',agrupacion_media.loc[k,'ponderacion'])]
      rectangles_[k] += [('ef_sample',agrupacion_media.loc[k,'ef_sample'])]
      rectangles_[k] += [('n_sample',agrupacion_media.loc[k,'n_sample'])]

    separacion_dim = self.get_dfs_dim(rectangles_)
    return separacion_dim

  def get_branches(self, df, var_obj, regr, no_trees_search=None, verbose=0):
    """Extract rectangular rules from a tree ensemble.

    Parameters
    ----------
    df : pd.DataFrame
        Original training data including the target column ``var_obj``.
    var_obj : str
        Name of the target column.
    regr : object
        Trained estimator supporting ``estimators_`` (scikit-learn) or
        ``toDebugString`` (PySpark) interfaces.
    no_trees_search : int | None, optional
        Maximum number of trees to analyse when summarising the forest.
        If ``None`` all trees are used.
    verbose : int, default 0
        Verbosity level; ``1`` enables progress bars and logging.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames, one per dimension, describing the rectangles
        (rules) extracted from the forest.
    """
    if var_obj not in df.columns:
       raise KeyError(f"Target column '{var_obj}' does not exist in the DataFrame")

    # Separate X and ignore target column
    df_copy = df.copy()
    cat_cols = df_copy.select_dtypes(['category']).columns
    for col in cat_cols:
        df_copy[col] = df_copy[col].cat.add_categories([0])
    X = df_copy.drop(columns=[var_obj]).fillna(0)

    if verbose==1:
       logger.info("Calling get_rangos to extract tree limits")
    try:
       df_full_arboles = self.get_rangos(regr, X, verbose)
    except Exception as exc:
       logger.exception("Error obtaining tree ranges: %s", exc)
       raise

    if verbose==1:
       logger.info("Extract rules with regex")
    
    try:
       df_full_arboles = self.get_fro(df_full_arboles)
    except Exception as exc:
       logger.exception("Error applying regex to the trees: %s", exc)
       raise

    if verbose==1:
       logger.info("Obtaining a summary of the trees")
    
    try:
       df_summ = self.get_summary_optimizado(
           df,
           df_full_arboles,
           var_obj,
           no_branch_lim=no_trees_search,
           verbose=verbose,
       )
    except Exception as exc:
       logger.exception("Error generating tree summary: %s", exc)
       raise
    
    if verbose==1:
       logger.info("Generating the final rectangular DataFrame")
       
    try:
       separacion_dim = self.extract_rectangles(df_summ)
    except Exception as exc:
       logger.exception("Error extracting rectangles: %s", exc)
       raise

    return separacion_dim
  
  