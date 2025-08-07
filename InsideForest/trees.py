import re
import pandas as pd
import numpy as np
import logging

from sklearn.tree import export_text
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Trees:

  def __init__(self, lang='python',n_sample_multiplier=2, ef_sample_multiplier=3):
    self.lang = lang
    self.n_sample_multiplier = n_sample_multiplier
    self.ef_sample_multiplier = ef_sample_multiplier


  def transform_tree_structure(self, tree_str):
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


  def get_rangos(self, regr, data1, verbose=0):
    # This function may be slow; add tqdm to the main loop.

    if self.lang == 'pyspark':
      arboles_estimadores = self.transform_tree_structure(regr.toDebugString)
      arboles_estimadores = [a for a in arboles_estimadores.values()]
    else:
      arboles_estimadores = regr.estimators_

    df_info = []
    n_estimador = 0

    # Use tqdm in the loop; disable when verbose=0
    for arbol_individual in tqdm(arboles_estimadores, disable=(verbose == 0), desc="Processing trees"):
      if self.lang == 'pyspark':
        r = arbol_individual
      else:
        r = export_text(arbol_individual)

      columnas_nombres = list(data1.columns)
      columnas_nombres.reverse()

      # Replace feature indices with names
      for i, feat in enumerate(columnas_nombres):
        r = r.replace('feature_' + str(len(columnas_nombres) - i - 1), feat)

      estructura = r.split('\n')
      estructura_iter = estructura.copy()
      paths = []

      for i, valor in enumerate(estructura):
        if not (('value: ' in valor) or ('class: ' in valor)):
          continue
        estructura_iter, path_ = self.get_path(estructura_iter)
        estructura_rep = [v.count('|') for v in path_[0]]
        
        # if len(estructura_rep) != len(set(estructura_rep)):
        #   posiciones_ = []
        #   for i_pos, valor_pos in enumerate(estructura_rep):
        #     posiciones = [k for k, v in enumerate(estructura_rep) if v == valor_pos]
        #     posiciones_ += [x for x in posiciones if x != max(posiciones)]
        #   path_aux = [val for j, val in enumerate(path_[0]) if j not in set(posiciones_)]
        #   path_[0] = path_aux

        if len(estructura_rep) != len(set(estructura_rep)):
          seen = set()
          new_path = []
          # Traverse path_[0] in reverse order
          for elem in reversed(path_[0]):
            # Count '|' in the current element
            bc = elem.count('|')

            # If not seen yet, add it (last occurrence)
            if bc not in seen:
              new_path.append(elem)
              seen.add(bc)

          # new_path is reversed; restore natural order
          new_path.reverse()

          # Replace original path
          path_[0] = new_path

        paths.append([x for x in path_ if x != ''])

      valores = [float(path[1].split(': ')[1].replace(']', '')) for path in paths]
      percent_ = np.percentile(valores, 90)
      estructuras_maximizadoras = [[pa[0], val] for pa, val in zip(paths, valores) if val >= percent_]

      importanc = []
      for n_path in range(len(estructuras_maximizadoras)):
        importanc += [
          [
            v.replace('|---', '').replace('|   ', '')[1:],
            2 / ((v.count('|')) + 1),
            n_path,
            n_estimador
          ]
          for v in estructuras_maximizadoras[n_path][0]
        ]

      asdf = pd.DataFrame(importanc, columns=['Regla', 'Importancia', 'N_regla', 'N_arbol'])
      asdf['Va_Obj_minima'] = percent_
      df_info.append(asdf)

      n_estimador += 1

      if verbose == 1 and n_estimador % 10 == 0:
        logger.info(f"Processed {n_estimador} trees")

    return pd.concat(df_info)


  def get_fro(self, df_full_arboles): 
    # Regular expression that captures:
    # Group 1: a non-space text (the feature)
    # Group 2: an operator among <=, >=, < or >
    # Group 3: a number (possibly decimal)
    pattern = re.compile(r'^(\S+)\s*(<=|>=|<|>)\s*([0-9.]+)$')

    def parse_regla(regla):
        match = pattern.search(regla)
        if match:
            feature = match.group(1)
            operador = match.group(2)
            rangos = float(match.group(3))
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



  def get_summary_optimizado(self, data1, df_full_arboles, var_obj, no_branch_lim=500, verbose=0):
      # 1) Calculamos el pivot que resume por N_regla, N_arbol, feature, operador, etc.
      agrupacion = pd.pivot_table(
          df_full_arboles,
          index=['N_regla', 'N_arbol', 'feature', 'operador'],
          values=['rangos', 'Importancia'],
          aggfunc=['min', 'max', 'mean']
      )
      
      # 2) Extraemos los valores de min, max y mean
      agrupacion_min = agrupacion['min'].reset_index()
      agrupacion_min = agrupacion_min[agrupacion_min['operador'] == '<=']

      agrupacion_max = agrupacion['max'].reset_index()
      agrupacion_max = agrupacion_max[agrupacion_max['operador'] == '>']

      # (We could use agrupacion_mean later; not reused in this example)
      agrupacion_mean = agrupacion['mean'].reset_index()

      # 3) Concatenate rows with operator <= and >, and sort
      agrupacion = pd.concat([agrupacion_min, agrupacion_max]).sort_values(['N_arbol', 'N_regla'])

      # 4) Select the top 100 trees
      top_100_arboles = agrupacion['N_arbol'].unique()[:no_branch_lim]

      # 5) Iterate over each tree and rule to build a single boolean mask per rule
      reglas = []

      for arbol_num in tqdm(top_100_arboles, disable=(verbose == 0), desc="Procesando ramas"):
          # Log (optional) depending on verbose
          if arbol_num % 50 == 0 and verbose == 1:
              logger.info(f"Processing tree branch: {arbol_num}")

          # Subset of the pivot for this tree
          ag_arbol = agrupacion[agrupacion['N_arbol'] == arbol_num]

          # Traverse each rule of that tree
          for regla_num in ag_arbol['N_regla'].unique():
              ag_regla = ag_arbol[ag_arbol['N_regla'] == regla_num]

              # Obtain (feature, value) pairs by operator
              men_ = ag_regla[ag_regla['operador'] == '<='][['feature', 'rangos']].values
              may_ = ag_regla[ag_regla['operador'] == '>'][['feature', 'rangos']].values

              # Build a boolean mask to filter data1 in a single step
              mask = np.ones(len(data1), dtype=bool)

              # Add <= conditions
              for col, val in men_:
                  mask &= (data1[col] <= val)

              # Add > conditions
              for col, val in may_:
                  mask &= (data1[col] > val)

              # Calculate n_sample and ef_sample
              n_sample = mask.sum()  # number of rows meeting all conditions
              # Avoid error when n_sample = 0
              ef_sample = data1.loc[mask, var_obj].mean() if n_sample > 0 else 0

              # Creamos una copia para esa regla, asignando los valores calculados
              ag_regla_copy = ag_regla.copy()
              ag_regla_copy['n_sample'] = n_sample
              ag_regla_copy['ef_sample'] = ef_sample

              reglas.append(ag_regla_copy)

      # 6) Concatenate all results and sort by requested metrics
      resultado = pd.concat(reglas, ignore_index=True)
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

  def get_branches(self, df, var_obj, regr, no_trees_search=500, verbose=0):
    """
    Main function to extract rectangles (rules) from trees.
    :param df: Original DataFrame
    :param var_obj: Name of target column
    :param regr: Trained model (RandomForest or others)
    :param verbose: 0 = no prints/progress bar, 1 = prints and tqdm
    :return: List of DataFrames with rectangles separated by dimension
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
       df_summ = self.get_summary_optimizado(df, df_full_arboles, var_obj, no_trees_search, verbose)
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
  
  