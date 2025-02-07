import re
import pandas as pd
import numpy as np

from sklearn.tree import export_text
from tqdm import tqdm

class trees:

  def __init__(self, lang='python',n_sample_multiplier=2, ef_sample_multiplier=3):
    self.lang = lang
    self.n_sample_multiplier = n_sample_multiplier
    self.ef_sample_multiplier = ef_sample_multiplier


  def transform_tree_structure(self, tree_str):
    # Separar el string en árboles individuales
    trees = tree_str.strip().split('\n  Tree')
    tree_dict = {}

    for i, tree in enumerate(trees):
      if i == 0:
        # Ignorar la primera línea si es la descripción del modelo
        if tree.startswith('RandomForestClassificationModel'):
          continue

      if not tree.strip():
        continue

      lines = tree.strip().split('\n')
      tree_name = "Tree " + lines[0].split('(')[0].strip()

      # Determinar el número de espacios a eliminar basado en las líneas después de la línea "Tree"
      min_indent = min((len(line) - len(line.lstrip())) for line in lines[1:] if line.strip())

      # Inicializar la lista que contendrá la estructura transformada
      transformed_lines = []

      for line in lines[1:]:

        # Eliminar la indentación mínima determinada
        adjusted_line = line[min_indent:]

        # Reemplazar los espacios iniciales por "|   "

        for i, s in enumerate(adjusted_line):
            if s != ' ':
                break

        indent = i * '|   '
        stripped_line = adjusted_line.strip()

        # print(adjusted_line, i)

        # Reemplazar "If" y "Else" por "|---"
        if stripped_line.startswith('If') or stripped_line.startswith('Else'):
          condition = stripped_line.split('(')[1].split(')')[0]
          transformed_lines.append(f"{indent}|--- {condition}")
        elif stripped_line.startswith('Predict'):
          prediction = stripped_line.split(': ')[1]
          transformed_lines.append(f"{indent}|--- class: {prediction}")

      # Guardar el árbol transformado en el diccionario
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
    # Esta función puede tardar mucho, añadimos tqdm para el bucle principal.

    if self.lang == 'pyspark':
      arboles_estimadores = self.transform_tree_structure(regr.toDebugString)
      arboles_estimadores = [a for a in arboles_estimadores.values()]
    else:
      arboles_estimadores = regr.estimators_

    df_info = []
    n_estimador = 0

    # Usamos tqdm en el for, si verbose=0, disable=True, si verbose=1, disable=False
    for arbol_individual in tqdm(arboles_estimadores, disable=(verbose == 0), desc="Procesando árboles"):
      if self.lang == 'pyspark':
        r = arbol_individual
      else:
        r = export_text(arbol_individual)

      columnas_nombres = list(data1.columns)
      columnas_nombres.reverse()

      # Reemplazo de feature indices por nombres
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
          # Recorremos path_[0] en orden inverso
          for elem in reversed(path_[0]):
            # Conteo de '|' en el elemento actual
            bc = elem.count('|')
            
            # Si no lo hemos visto aún, lo añadimos (porque es la última vez que aparece)
            if bc not in seen:
              new_path.append(elem)
              seen.add(bc)
          
          # new_path está invertido, lo devolvemos a su orden natural
          new_path.reverse()
          
          # Reemplazamos el path original
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
        print(f"Procesados {n_estimador} árboles")

    return pd.concat(df_info)


  def get_fro(self, df_full_arboles): 
    # Expresión regular que busca:
    # Grupo 1: un texto sin espacios (la feature)
    # Grupo 2: un operador entre <=, >=, < o >
    # Grupo 3: un número (posiblemente decimal)
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

    # Aplicamos la función de parseo a cada fila de 'Regla'
    df_full_arboles[['feature', 'operador', 'rangos']] = df_full_arboles['Regla'].apply(
        lambda x: parse_regla(x)
    ).apply(pd.Series)

    # Remover las filas que no se pudieron parsear
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

    # Añadimos tqdm sobre top_100_ramas para ver progreso
    for arbol_num in tqdm(top_100_ramas, disable=(verbose == 0), desc="Procesando ramas"):
      # Mantenemos este print pero lo hacemos condicional
      if arbol_num % 50 == 0 and verbose == 1:
        print(f"Procesando rama del árbol: {arbol_num}")

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

      # (Podríamos usar agrupacion_mean si lo necesitas luego; en el ejemplo no se reusa directamente)
      agrupacion_mean = agrupacion['mean'].reset_index()

      # 3) Concatenamos las filas con operador <= y >, y ordenamos
      agrupacion = pd.concat([agrupacion_min, agrupacion_max]).sort_values(['N_arbol', 'N_regla'])

      # 4) Seleccionamos los top 100 árboles
      top_100_arboles = agrupacion['N_arbol'].unique()[:no_branch_lim]

      # 5) Iteramos por cada árbol y regla para construir una única máscara booleana por regla
      reglas = []

      for arbol_num in tqdm(top_100_arboles, disable=(verbose == 0), desc="Procesando ramas"):
          # Imprimimos (opcional) según el valor de verbose
          if arbol_num % 50 == 0 and verbose == 1:
              print(f"Procesando rama del árbol: {arbol_num}")

          # Subconjunto del pivot para este árbol
          ag_arbol = agrupacion[agrupacion['N_arbol'] == arbol_num]

          # Recorremos cada regla de ese árbol
          for regla_num in ag_arbol['N_regla'].unique():
              ag_regla = ag_arbol[ag_arbol['N_regla'] == regla_num]

              # Obtenemos pares (feature, valor) según operador
              men_ = ag_regla[ag_regla['operador'] == '<='][['feature', 'rangos']].values
              may_ = ag_regla[ag_regla['operador'] == '>'][['feature', 'rangos']].values

              # Construimos la máscara booleana para filtrar data1 en un único paso
              mask = np.ones(len(data1), dtype=bool)

              # Agregamos condiciones de <=
              for col, val in men_:
                  mask &= (data1[col] <= val)

              # Agregamos condiciones de >
              for col, val in may_:
                  mask &= (data1[col] > val)

              # Calculamos n_sample y ef_sample
              n_sample = mask.sum()  # número de filas que cumplen todas las condiciones
              # Evitamos error en caso de n_sample = 0
              ef_sample = data1.loc[mask, var_obj].mean() if n_sample > 0 else 0

              # Creamos una copia para esa regla, asignando los valores calculados
              ag_regla_copy = ag_regla.copy()
              ag_regla_copy['n_sample'] = n_sample
              ag_regla_copy['ef_sample'] = ef_sample

              reglas.append(ag_regla_copy)

      # 6) Concatenamos todos los resultados y ordenamos por las métricas solicitadas
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
    Función principal para extraer los rectángulos (reglas) de los árboles.
    :param df: DataFrame original
    :param var_obj: Nombre de la columna objetivo
    :param regr: Modelo (RandomForest u otro) entrenado
    :param verbose: 0 = sin prints ni barra de progreso, 1 = con prints y tqdm
    :return: Lista de DataFrames con los rectángulos separados por dimensión
    """
    # Separamos X e ignoramos la columna objetivo
    X = df.drop(columns=[var_obj]).fillna(0)

    if verbose==1:
       print("Llamamos a get_rangos para extraer limites de los arboles")
    df_full_arboles = self.get_rangos(regr, X, verbose)

    if verbose==1:
       print("Extraer las reglas con regex")
    
    df_full_arboles = self.get_fro(df_full_arboles)

    if verbose==1:
       print("Obtenemos un resumen de los  árboles")
    
    df_summ = self.get_summary_optimizado(df, df_full_arboles, var_obj, no_trees_search, verbose)
    
    if verbose==1:
       print("Generamos el df final con forma de rectángulo")
       
    separacion_dim = self.extract_rectangles(df_summ)

    return separacion_dim
  
  