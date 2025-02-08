from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import DBSCAN, KMeans

import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import Rectangle
from collections import Counter

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster


class regions:
    
  def search_original_tree(self,df_clusterizado, separacion_dim):
            
    dims_after_clus = list(df_clusterizado['linf'].columns)

    for i in range(len(separacion_dim)):
      dims = list(set(separacion_dim[i].dimension.values))
      if len(dims)==len(dims_after_clus):
        if all([a==b for a, b in zip(sorted(dims),
                                     sorted(dims_after_clus))]):
          break

    return separacion_dim[i]
    
  def mean_distance_ndim(self, df_sep_dm_agg):
    df_p1 = df_sep_dm_agg.xs('linf', axis=1, level=0)
    df_p2 = df_sep_dm_agg.xs('lsup', axis=1, level=0)
    m_medios = [(df_p1.iloc[i] + df_p2.iloc[i]) / 2 for i in range(len(df_p1))]
    return pd.DataFrame(m_medios)
  
  def mean_distance_ndim_fast(self, df_sep_dm_agg, verbose):
      """
      Versión optimizada de mean_distance_ndim que calcula (linf + lsup)/2
      en forma vectorizada usando NumPy.

      Parámetros:
      - df_sep_dm_agg: DataFrame con índices multi-nivel que permiten extraer
        'linf' y 'lsup' usando xs.

      Retorna:
      - DataFrame con la media de linf y lsup por fila y dimensión.
      """

      # Extraemos linf y lsup
      df_p1 = df_sep_dm_agg.xs('linf', axis=1, level=0)
      df_p2 = df_sep_dm_agg.xs('lsup', axis=1, level=0)

      # Operación vectorizada con NumPy:
      # (df_p1 + df_p2) / 2
      m_medios_values = (df_p1.values + df_p2.values) / 2.0

      # Reconstruimos el DataFrame con el mismo índice y columnas que df_p1
      df_result = pd.DataFrame(
          m_medios_values,
          index=df_p1.index,
          columns=df_p1.columns
      )

      return df_result

  def posiciones_valores_frecuentes(self, lista):
    frecuentes = Counter(lista).most_common()
    if len(set(lista)) == len(lista):
      resultado = list(range(len(lista)))
    else:
      frecuencia_maxima = frecuentes[0][1]
      resultado = [i for i, v in enumerate(lista) if v in dict(frecuentes).keys() and 
             dict(frecuentes)[v] == frecuencia_maxima]
    return resultado
  
  def get_eps_multiple_groups_opt(self, data, eps_min=1e-5, eps_max=None):
    if len(data)==1:
      return 1e-2
    elif len(data)==2:
      return .5
    if eps_max is None:
      eps_max = np.max(np.sqrt(np.sum((data - np.mean(data, axis=0)) ** 2, axis=1)))
      if eps_max<=1e-10:
        eps_max=0.1
    eps_values = np.linspace(eps_min, eps_max, num=75)
    n_groups = []
    last_unique_labels = None
    was_multiple_groups = False
    for eps in eps_values:
      if eps <= 0:
        continue
      dbscan = DBSCAN(eps=eps, min_samples=2)
      labels = dbscan.fit_predict(data)
      unique_labels = np.unique(labels)
      if unique_labels.size > 1:
        n_groups.append(unique_labels.size)
        last_unique_labels = unique_labels.size
        was_multiple_groups = True
      elif unique_labels.size == 1 and was_multiple_groups:
        break
    if len(n_groups)==0:
      return (eps_min + eps_max) / 2
    mode_indices = self.posiciones_valores_frecuentes(n_groups)
    if len(mode_indices) == 1:
      return eps_values[mode_indices[0]]
    else:
      mean_grupos = [n_groups[i] for i in mode_indices]
      dist_to_mean = [np.abs(x - np.mean(mean_grupos)) for x in mean_grupos]
      return eps_values[mode_indices[np.argmin(dist_to_mean)]]
  
    
  def fill_na_pond(self, df_sep_dm, df, features_val):
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
      """
      Versión ultra-optimizada de fill_na_pond para reemplazar -inf e inf usando operaciones vectorizadas avanzadas.
      
      Parámetros:
      - df_sep_dm: DataFrame con columnas multi-nivel ('linf', 'lsup', 'ponderador', etc.).
      - df: DataFrame original para extraer límites de cada dimensión.
      - features_val: Lista de características/dimensiones presentes en df.
      
      Retorna:
      - DataFrame con los mismos valores que el original, pero reemplazando -inf e inf
        por los límites correspondientes en las columnas 'linf' y 'lsup'.
        Incluye la columna 'ponderador'.
      """
      # Extraer las columnas 'linf' y 'lsup'
      df_lilu = df_sep_dm[['linf', 'lsup']].copy()
      
      # Calcular los límites de reemplazo para cada dimensión
      lsup_limit = df[features_val].max() + 1  # Límite superior
      linf_limit = df[features_val].min() - 1  # Límite inferior
      
      # Asegurarse de que el orden de features_val coincide con el orden de las columnas
      # Obtener los nombres de las dimensiones desde las columnas MultiIndex
      linf_features = df_lilu['linf'].columns.tolist()
      lsup_features = df_lilu['lsup'].columns.tolist()
      
      # Para 'linf' columns
      linf_repl_df = pd.DataFrame(
          np.tile(linf_limit.values, (df_lilu['linf'].shape[0], 1)),
          columns=df_lilu['linf'].columns,
          index=df_lilu.index
      )
      
      # Para 'lsup' columns
      lsup_repl_df = pd.DataFrame(
          np.tile(lsup_limit.values, (df_lilu['lsup'].shape[0], 1)),
          columns=df_lilu['lsup'].columns,
          index=df_lilu.index
      )
      
      # Crear máscaras para identificar dónde están los -inf y inf
      mask_linf = np.isinf(df_lilu['linf'].values)
      mask_lsup = np.isinf(df_lilu['lsup'].values)
      
      # Aplicar las máscaras y reemplazar los valores
      # Usamos donde para asignar los valores de reemplazo donde la máscara es True
      df_lilu['linf'] = np.where(mask_linf, linf_repl_df.values, df_lilu['linf'].values)
      df_lilu['lsup'] = np.where(mask_lsup, lsup_repl_df.values, df_lilu['lsup'].values)
      
      # Concatenar la columna 'ponderador' de vuelta al DataFrame
      df_replaced = pd.concat([df_lilu, df_sep_dm[['ponderador','ef_sample','n_sample']]], axis=1)
      
      return df_replaced

  def group_by_cluster(self, df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
      """
      Agrupa un DataFrame por la columna 'cluster', conservando el primer valor de las demás columnas.
      Para las columnas que contienen 'ef_sample' o 'n_sample', solo se utiliza la primera columna encontrada.
      Se maneja el caso en el que la columna de agrupación se repita, evitando duplicados en la selección final.
      
      :param df: DataFrame de entrada.
      :param cluster_col: Nombre de la columna por la que se agrupará (por defecto 'cluster').
      :return: DataFrame agrupado.
      """
      
      # Verificar si la columna 'cluster' contiene valores no escalares (listas, sets, dicts, etc.)
      if df[cluster_col].apply(lambda x: isinstance(x, (list, tuple, set, dict))).any():
          df = df.copy()  # Para evitar SettingWithCopyWarning
          df[cluster_col] = df[cluster_col].apply(
              lambda x: tuple(x) if isinstance(x, (list, set)) else x
          )
      
      # Identificar las columnas que contienen 'ef_sample' y 'n_sample'
      ef_sample_cols = [col for col in df.columns if "ef_sample" in col]
      n_sample_cols = [col for col in df.columns if "n_sample" in col]
      ponderador_cols = [col for col in df.columns if "ponderador" in col]

      # Seleccionar la primera columna de cada tipo, si existen
      first_ef_sample = ef_sample_cols[0] if ef_sample_cols else None
      first_n_sample = n_sample_cols[0] if n_sample_cols else None
      ponderador_sample = ponderador_cols[0] if ponderador_cols else None

      # Construir la lista de columnas a conservar:
      # - Se incluye la columna de cluster de forma explícita.
      # - Se añaden todas las columnas que no sean parte de las listas de ef_sample o n_sample
      #   ni la propia columna cluster (para evitar duplicados).
      cols_to_keep = [cluster_col] + [
          col for col in df.columns if col not in (ef_sample_cols + n_sample_cols + ponderador_cols + [cluster_col])
      ]
      
      # Agregar la primera columna 'ef_sample' y 'n_sample', si existen
      if first_ef_sample:
          cols_to_keep.append(first_ef_sample)
      if first_n_sample:
          cols_to_keep.append(first_n_sample)
      if ponderador_sample:
          cols_to_keep.append(ponderador_sample)
      
      # Eliminar duplicados en 'cols_to_keep' preservando el orden
      cols_to_keep = list(dict.fromkeys(cols_to_keep))
      
      # Agrupar por la columna 'cluster' y conservar el primer valor de cada columna seleccionada
      df_grouped = df[cols_to_keep].groupby(cluster_col, as_index=False).first()
      df_grouped_c = df[cols_to_keep[:2]].groupby(cluster_col, as_index=False).count()
      df_grouped_c = df_grouped_c.rename(columns={cols_to_keep[1]:'count'})
      
      return df_grouped.merge(df_grouped_c, how='left', on='cluster')


  def get_agg_regions_j(self, df_eval, df):

    features_val = sorted(df_eval['dimension'].unique())
    df_sep_dm = pd.pivot_table(df_eval, index='rectangulo', columns='dimension')
    df_sep_dm = self.fill_na_pond_fastest(df_sep_dm, df, features_val,None)

    # Si ya tienes el MultiIndex, puedes aplanarlo:
    df_sep_dm.columns = [f"{col[1].strip()}&&{col[0]}" for col in df_sep_dm.columns]
    df_raw = df_sep_dm

    # Lista de dimensiones (extraídas de los nombres de columnas)
    dims = sorted(set(col.split('&&')[0] for col in df_raw.columns))

    # Función para calcular IoU entre dos hipercubos
    def iou_hypercube(row1, row2, dims):
        inter_vol = 1
        vol1 = 1
        vol2 = 1

        for dim in dims:
            low1, high1 = row1[f"{dim}&&linf"], row1[f"{dim}&&lsup"]
            low2, high2 = row2[f"{dim}&&linf"], row2[f"{dim}&&lsup"]

            inter_low = max(low1, low2)
            inter_high = min(high1, high2)

            if inter_low >= inter_high:
                return 0.0  # No hay intersección

            inter_vol *= (inter_high - inter_low)
            vol1 *= (high1 - low1)
            vol2 *= (high2 - low2)

        union_vol = vol1 + vol2 - inter_vol
        return inter_vol / union_vol if union_vol != 0 else 0.0

    # Crear matriz de distancias basada en IoU (1 - IoU para que sea una distancia)
    n = len(df_raw)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i, j] = 1 - iou_hypercube(df_raw.iloc[i], df_raw.iloc[j], dims)
            distance_matrix[j, i] = distance_matrix[i, j]  # Simétrica

    # Convertir matriz en formato de lista de distancias para clustering
    dist_vector = squareform(distance_matrix)

    # Aplicar clustering jerárquico
    Z = linkage(dist_vector, method='average')  # Método de enlace promedio

    # Determinar número de clusters (ajustar umbral según necesites)

    for tr in [tr/100 for tr in range(65,5,-1)]:
      clusters = fcluster(Z, tr, criterion='distance')
      if len(set(clusters))*2>len(distance_matrix):
        break

    # Agregar la asignación de clusters al DataFrame
    df_raw["cluster"] = clusters

    n_sample_col = [col for col in df_raw.columns if 'n_sample' in col][0]
    eff_sample_col = [col for col in df_raw.columns if 'ef_sample' in col][0]

    df_raw.sort_values(by=['cluster', n_sample_col, eff_sample_col], ascending=False, inplace=True)

    df_raw_agg = self.group_by_cluster(df_raw)

    return df_raw_agg


  def set_multiindex(self, df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
      """
      Transforma el DataFrame para:
        - Usar la columna `cluster` como índice.
        - Convertir las demás columnas en un MultiIndex a partir del separador "&&".
      
      Reglas:
        1. Si una columna contiene "&&", se separa en:
            left  = nombre_medida
            right = identificador
        2. Si right es una palabra especial (ef_sample, n_sample, ponderador, count):
              primer nivel = "metrics"
              segundo nivel = right
        3. Si right no es especial:
              primer nivel = right
              segundo nivel = left
        4. Si la columna no contiene "&&":
            - Si contiene alguna palabra especial, se asigna ("metrics", <nombre completo>).
            - De lo contrario, se asigna (None, <nombre completo>).
      
      El MultiIndex resultante tiene nombres de niveles: [None, 'dimension'].
      
      :param df: DataFrame de entrada.
      :param cluster_col: Nombre de la columna que se usará como índice (por defecto "cluster").
      :return: DataFrame con índice `cluster` y columnas con MultiIndex.
      """
      # Definir las palabras especiales.
      special_words = {"ef_sample", "n_sample", "ponderador", "count"}
      new_cols = []
      
      # Procesar cada columna (exceptuando la columna de cluster)
      for col in df.columns:
          if col == cluster_col:
              continue  # Se usará como índice.
          if "&&" in col:
              # Separamos en left y right
              left, right = col.split("&&", 1)
              left = left.strip()
              right = right.strip()
              if right in special_words:
                  # Para columnas especiales: ("metrics", <nombre especial>), es decir, right.
                  new_label = ("metrics", right)
              else:
                  # Para columnas no especiales: (identificador, nombre_medida)
                  new_label = (right, left)
          else:
              # Columnas sin separador "&&"
              if any(sw in col for sw in special_words):
                  new_label = ("metrics", col)
              else:
                  new_label = (None, col)
          new_cols.append(new_label)
      
      # Crear el MultiIndex con nombres de niveles [None, 'dimension']
      multi_cols = pd.MultiIndex.from_tuples(new_cols, names=[None, "dimension"])
      
      # Establecer la columna cluster como índice y asignar el nuevo MultiIndex a las columnas.
      df_new = df.set_index(cluster_col).copy()
      df_new.columns = multi_cols
      return df_new

  def prio_ranges(self, separacion_dim, df):

    df_res = [self.set_multiindex(self.get_agg_regions_j(df_, df)) 
              for df_ in separacion_dim]

    df_res = [df.sort_values(by=[('metrics', 'n_sample'), 
                                ('metrics', 'count'), 
                                ('metrics', 'ef_sample')], ascending=False) for df in df_res]

    return df_res


  def plot_bidim(self, df,df_sep_dm_agg, eje_x, eje_y, var_obj):
    df_sep_dm_agg['derecha'] = df_sep_dm_agg[('lsup',eje_x)]-\
    df_sep_dm_agg[('linf',eje_x)]
    df_sep_dm_agg['arriba'] = df_sep_dm_agg[('lsup',eje_y)]-\
    df_sep_dm_agg[('linf',eje_y)] 

    eis_ = df_sep_dm_agg['linf'].values
    ders_ = df_sep_dm_agg['derecha'].values
    arrs_ = df_sep_dm_agg['arriba'].values

    ax = sns.scatterplot(x=eje_x, y=eje_y, hue=var_obj, palette='RdBu', data=df)
    norm = plt.Normalize(df[var_obj].min(), df[var_obj].max())
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])
    for i in range(len(eis_)):
      if i>25:
        break
      # i = len(eis_)-i-1
      ax.add_patch(Rectangle(eis_[i], ders_[i], arrs_[i], alpha=0.15, color='#0099FF'))
    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm, ax=ax)

    
  def plot_scatter3d(self,df_r,df, ax, var_obj):
    dimesniones_fd = df_r['linf'].columns
    df_scatter = df[list(dimesniones_fd)+[var_obj]].copy()
    df_scatter.replace(dict(zip([True,False],[1,0])), inplace=True)

    valores_target = list(df_scatter[var_obj].unique())
    colores_disc = ['red','blue','green','yellow','orange']
    replace_var = dict(zip(valores_target,colores_disc[:len(valores_target)]))

    # Generamos los puntos del scatter plot
    xs = df_scatter[dimesniones_fd[0]].values
    ys = df_scatter[dimesniones_fd[1]].values
    zs = df_scatter[dimesniones_fd[2]].values
    colors = df_scatter[var_obj].replace(replace_var).values
    hex_colors = [mcolors.to_hex(c) for c in colors]
    # Dibujamos los puntos
    return ax.scatter(xs, ys, zs, c=hex_colors, s=1)
    

  def plot_rect3d(self,df_r,i, ax):
    # Obtenemos los valores de los límites del rectángulo
    x1, y1, z1, x2, y2, z2 = df_r.drop(columns=['ponderador']).iloc[i,:].values.flatten()

    # Generamos los puntos del rectángulo
    X = np.array([[x1, x2, x2, x1, x1], [x1, x2, x2, x1, x1]])
    Y = np.array([[y1, y1, y2, y2, y1], [y1, y1, y2, y2, y1]])
    Z = np.array([[z1, z1, z1, z1, z1], [z2, z2, z2, z2, z2]])

    # Dibujamos el rectángulo con transparencia
    return ax.plot_surface(X=X, Y=Y, Z=Z, alpha=0.2, color='gray')

  def plot_tridim(self,df_r,df,var_obj):
    # Graficar figura
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(df_r.shape[0]):
        self.plot_rect3d(df_r,i, ax)

    self.plot_scatter3d(df_r,df, ax, var_obj)
    
    dimesniones_fd = df_r['linf'].columns
    # Configuramos las etiquetas de los ejes
    ax.set_xlabel(str(dimesniones_fd[0]))
    ax.set_ylabel(str(dimesniones_fd[1]))
    ax.set_zlabel(str(dimesniones_fd[2]))

    plt.show()
    
    
  def plot_multi_dims(self,df_sep_dm_agg, df,var_obj):
    dimensiss = df_sep_dm_agg['linf'].columns
    # print(len(dimensiss), dimensiss)
    if len(dimensiss)==1:
      eje_x = dimensiss.tolist()[0]
      eje_y = 'altura'
      df_a = df.copy()
      df_a.loc[:,eje_y] = 1
      meddd_p = abs((df_sep_dm_agg['linf']-df_sep_dm_agg['lsup']).mean()[0])
      df_sep_dm_agg[('lsup', eje_y)] = 1+meddd_p
      df_sep_dm_agg[('linf', eje_y)] = 1-meddd_p
    
      self.plot_bidim(df_a,df_sep_dm_agg, eje_x, eje_y, var_obj)
    elif len(dimensiss)==2:
      df_a = df.copy()
      eje_x, eje_y = dimensiss.tolist()
      self.plot_bidim(df_a,df_sep_dm_agg, eje_x, eje_y, var_obj)
    elif len(dimensiss)==3:
      self.plot_tridim(df_sep_dm_agg,df,var_obj)
    else:
      ddimee = dimensiss.tolist()
      eje_x, eje_y, eje_z = ddimee[0], ddimee[1], ddimee[1]


  
  def asignar_ids_unicos(self, lista_reglas):
    cluster_id = 0
    for df in lista_reglas:
      n_reglas = len(df)
      df = df.reset_index(drop=True)
      df['cluster_'] = range(cluster_id, cluster_id + n_reglas)
      cluster_id += n_reglas
    return lista_reglas

  def generar_descripcion_clusters(self, df_reglas):
      """
      Genera un DataFrame con la descripción textual y el ponderador de cada cluster,
      basado en las reglas definidas en df_reglas.

      Se asume que:
        - row['linf'] y row['lsup'] son Series con índices correspondientes a los nombres de las variables.
        - El ponderador se extrae de ('metrics', 'ponderador').
        - La columna 'cluster' contiene el identificador del cluster.
      """
      # import numpy as np
      # import pandas as pd

      cluster_descripciones = []

      for idx, row in df_reglas.iterrows():
          # Extraer el identificador del cluster; si viene encapsulado, extraer el valor escalar
          cluster_id = row['cluster']
          if hasattr(cluster_id, 'values'):
              cluster_id = cluster_id.values[0]

          # Extraer los límites inferiores y superiores
          linf = row['linf'].dropna()
          lsup = row['lsup'].dropna()

          # Construir la descripción: para cada variable definida en linf, se crea un string
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
      Asigna clusters a los datos en base a reglas.

      Parámetros
      ----------
      df_datos : pd.DataFrame
          DataFrame con los datos a asignar a clusters.
      df_reglas : pd.DataFrame
          DataFrame con la definición de las reglas, con columnas MultiIndex en las que:
            - Las columnas de límites inferiores están bajo el primer nivel 'linf'
            - Las columnas de límites superiores están bajo el primer nivel 'lsup'
            - El ponderador se encuentra en ('metrics', 'ponderador')
            - La columna 'cluster' se encuentra de forma normal.
      keep_all_clusters : bool, optional (default=True)
          Si es True, se añadirán cuatro columnas adicionales:
            - 'n_clusters': número de clusters en los que cae el registro.
            - 'clusters_list': lista de todos los clusters a los que pertenece ese registro.
            - 'ponderadores_list': lista de los ponderadores de los clusters a los que pertenece.
            - 'ponderador_mean': media de los ponderadores de los clusters a los que pertenece.
          Si es False, se asigna sólo el cluster de mayor ponderador (sin columnas extras).

      Returns
      -------
      df_datos_clusterizados : pd.DataFrame
          DataFrame de entrada con la columna 'cluster' asignada.
          Si keep_all_clusters=True, se añaden las columnas adicionales.
      df_clusters_descripcion : pd.DataFrame
          DataFrame con la descripción (o métricas) de cada cluster.
      """
      # import numpy as np

      n_datos = df_datos.shape[0]
      # Array para almacenar el cluster "principal" (el que tiene mayor ponderador)
      clusters_datos = np.full(n_datos, -1, dtype=float)
      # Array para almacenar el ponderador del cluster principal
      ponderador_datos = np.full(n_datos, -np.inf, dtype=float)

      # Si se desea conservar el listado completo de clusters por registro, inicializamos estructuras
      if keep_all_clusters:
          clusters_datos_all = [[] for _ in range(n_datos)]
          ponderadores_datos_all = [[] for _ in range(n_datos)]

      # --- 1) Extraer y normalizar la información de las reglas ---
      reglas_info = []
      # Se recorre cada fila de df_reglas; normalmente el número de reglas es pequeño
      for idx, row in df_reglas.iterrows():
          if row[('metrics', 'ponderador')]==0:
              continue
          # Extraer los límites inferiores y superiores; se asume que row['linf'] y row['lsup'] son Series
          linf = row['linf'].dropna()
          lsup = row['lsup'].dropna()
          variables = linf.index.tolist()

          # Extraer el ponderador desde el grupo 'metrics'
          p_val = row[('metrics', 'ponderador')]
          # Si por alguna razón fuese iterable (por ejemplo, una lista), se toma su media;
          # en condiciones normales es un valor escalar.
          ponderador = p_val.mean() if hasattr(p_val, '__iter__') else p_val

          # Extraer el cluster asignado; si viene encapsulado, se extrae el valor escalar.
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

      # --- 2) Evaluar para cada registro qué reglas cumple ---
      for regla in reglas_info:
          variables = regla['variables']
          linf = regla['linf']
          lsup = regla['lsup']
          ponderador = regla['ponderador']
          cluster = regla['cluster']

          # Extraer las columnas relevantes y convertir a arrays para operaciones vectorizadas
          X_datos = df_datos[variables]
          condiciones = [
              (X_datos[var].to_numpy() >= linf[var]) & (X_datos[var].to_numpy() <= lsup[var])
              for var in variables
          ]
          # Si no hay variables, la regla no se puede evaluar; se asume que ningún registro la cumple.
          if condiciones:
              cumple_regla = np.logical_and.reduce(condiciones)
          else:
              cumple_regla = np.zeros(n_datos, dtype=bool)

          if keep_all_clusters:
              indices_cumple = np.where(cumple_regla)[0]
              for i in indices_cumple:
                  clusters_datos_all[i].append(cluster)
                  ponderadores_datos_all[i].append(ponderador)

          # Actualizar el cluster "principal" si el ponderador de esta regla es mayor
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
      Elimina únicamente aquellas reglas que estén estrictamente contenidas en otras
      y cuyo ponderador sea menor que el de la regla "contenedora".

      Comparación entre todos los elementos de la lista:
        - Se concatenan todos los DataFrames (cada uno puede tener un conjunto de dimensiones distinto).
        - Para cada regla, se evalúa su contención en otras reglas que tengan
          igual o mayor número de dimensiones.
        - Si la contención es total y el ponderador de la regla 'grande' es
          estrictamente mayor, se elimina la regla contenida.

      Parámetros
      ----------
      lista_reglas : List[pd.DataFrame]
          Lista de DataFrames, cada uno con columnas MultiIndex:
          - 'linf': límites inferiores de cada dimensión
          - 'lsup': límites superiores de cada dimensión
          - ('metrics', 'ponderador'): ponderador de la regla
          - Otras columnas (p.ej. 'cluster') que se conservarán sin usarse en la comparación.

          Cada DataFrame puede tener un conjunto de dimensiones distinto.

      Retorna
      -------
      df_reglas_importantes : pd.DataFrame
          DataFrame con todas las reglas (de todos los DataFrames) que no resultaron redundantes.
      """
      # import numpy as np
      # import pandas as pd

      # 1. Concatenar todos los DataFrames en uno para comparar reglas de toda la lista
      df_reglas = pd.concat(lista_reglas, ignore_index=True)

      # Asegurar que los nombres de nivel de las columnas sean [None, 'dimension']
      if df_reglas.columns.names != [None, 'dimension']:
          df_reglas.columns.names = [None, 'dimension']

      # 2. Extraer la información esencial de cada regla
      reglas_info = []
      for idx, row in df_reglas.iterrows():
          linf = row['linf']        # Límites inferiores (Series)
          lsup = row['lsup']        # Límites superiores (Series)
          p = row[('metrics', 'ponderador')]  # Ponderador (valor escalar)

          # Conjunto de dimensiones que la regla define (usando linf.dropna())
          # Asumimos que "sin límite" => NaN y, por tanto, no cuenta como dimensión activa.
          vars_i = set(linf.dropna().index)

          reglas_info.append({
              'idx': idx,
              'variables': vars_i,
              'linf': linf.to_dict(),
              'lsup': lsup.to_dict(),
              'ponderador': p
          })

      # 3. Ordenar las reglas por la cantidad de dimensiones (de menor a mayor)
      #    Esto ayuda a reducir el número de comparaciones.
      reglas_info_sorted = sorted(reglas_info, key=lambda x: len(x['variables']))

      # 4. Recorrer las reglas y marcar las que resulten redundantes
      redundant_indices = set()
      num_reglas = len(reglas_info_sorted)

      for i in range(num_reglas):
          rule_i = reglas_info_sorted[i]
          if rule_i['idx'] in redundant_indices:
              # Ya se marcó como redundante en alguna comparación anterior
              continue

          set_i = rule_i['variables']
          # Comparamos únicamente con las reglas que siguen en la lista,
          # ya que están ordenadas y tendrán >= número de dimensiones.
          for j in range(i + 1, num_reglas):
              rule_j = reglas_info_sorted[j]

              # 4.1. Requerimos que el conjunto de variables i sea un subconjunto del j
              if not set_i.issubset(rule_j['variables']):
                  continue

              # 4.2. Verificar contención de límites: para cada variable en i,
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
                  break  # No se necesitan más comparaciones para i

      # 5. Eliminar las reglas redundantes y reindexar
      df_reglas_importantes = df_reglas.drop(index=redundant_indices).reset_index(drop=True)
      return df_reglas_importantes


  def combinar_dataframes_por_columnas(self, lista_reglas):
      """
      Combina los DataFrames de la lista que tienen exactamente las mismas columnas (MultiIndex) en uno solo.

      La función agrupa los DataFrames por la estructura de columnas y concatena aquellos que compartan la misma estructura.
      Si un grupo contiene un único DataFrame, se conserva tal cual.

      Parámetros
      ----------
      lista_reglas : list of pd.DataFrame
          Lista de DataFrames con columnas MultiIndex.

      Retorna
      -------
      nueva_lista : list of pd.DataFrame
          Nueva lista en la que cada elemento es el resultado de concatenar (o conservar) los DataFrames
          que comparten la misma estructura de columnas.
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
      Expande una columna de listas de clusters en múltiples columnas binarias.
      
      Parámetros
      ----------
      df : pd.DataFrame
          DataFrame que contiene la columna a expandir.
      columna_clusters : str
          Nombre de la columna que contiene las listas de clusters.
      prefijo : str, opcional
          Prefijo para los nombres de las nuevas columnas. Por defecto es 'cluster_'.
      
      Retorna
      -------
      pd.DataFrame
          DataFrame original unido con las nuevas columnas binarias de clusters.
      """
      # Asegurar que la columna de clusters contiene listas o sets
      if not df[columna_clusters].apply(lambda x: isinstance(x, (list, set))).all():
          raise ValueError(f"La columna '{columna_clusters}' debe contener listas o sets de clusters.")
      
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
              raise ValueError(f"La columna '{col}' no existe en el DataFrame.")
      
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
              raise ValueError(f"El formato de la columna '{col}' no es válido. Esperado 'cluster_<número>.0'.")
      
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
      if sorted: # Changed variable name
          df[new_key_column] = df[list_column].apply(lambda x: delimiter.join(map(str, sorted(x))))
      else:
          df[new_key_column] = df[list_column].apply(lambda x: delimiter.join(map(str, x)))
      return df

  def get_clusters_importantes(self, df_clusterizado):
    df_clusterizado_diff = df_clusterizado[['clusters_list']].drop_duplicates()
    df_clusterizado_diff['n_ls'] = df_clusterizado_diff.apply(lambda x: len(x['clusters_list']), axis=1)
    df_clusterizado_diff_sub = df_clusterizado_diff

    df_expanded = self.expandir_clusters_binario(df_clusterizado_diff_sub,'clusters_list','cluster_')
    cluster_cols = [x for x in df_expanded.columns if 'cluster_' in x]

    sample_size = int(np.sqrt(df_clusterizado_diff_sub.shape[0])*3)
    eps_su = self.get_eps_multiple_groups_opt(df_expanded[cluster_cols].sample(sample_size))

    # Obtener clusters
    df_clustered, _ = self.apply_clustering_and_similarity(df_expanded, cluster_cols,
                                                                dbscan_params={'eps': eps_su, 'min_samples': 2},
                                                                kmeans_params={'n_clusters': 6, 'random_state': 42})


    df_custers__vc = df_clustered[['db_labels','km_labels']].value_counts()
    values_up = np.sqrt(df_custers__vc.head(1)).values[0]
    df_custers__vc_ = df_custers__vc[df_custers__vc>values_up]

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
      grupo_s_ = self.add_active_clusters(grupo_s_).drop(columns='count')

      cols_keys = list(grupo_s_.columns[:-1])
      df_clustered_subcluster_agg = df_clustered_subcluster.merge(grupo_s_, on=cols_keys, how='left')


      pd_cluster_sun.append(df_clustered_subcluster_agg)

    df_clustered_subcluster_agg_all = pd.concat(pd_cluster_sun)

    df_clusterizado = self.convert_list_to_string(df_clusterizado, 'clusters_list',
                                            sorted=False, delimiter=',', new_key_column='clusters_key')

    df_clustered_subcluster_agg_all = self.convert_list_to_string(df_clustered_subcluster_agg_all, 'clusters_list',
                                                            sorted=False, delimiter=',', new_key_column='clusters_key')

    df_clusterizado_add = df_clusterizado.merge(df_clustered_subcluster_agg_all[['clusters_key','active_clusters']],
                                                on='clusters_key', how='left')

    return df_clusterizado_add.drop(columns='clusters_key')


  def labels(self, df, df_reres, include_summary_cluster=True):
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

    if include_summary_cluster:
      df_datos_clusterizados = self.get_clusters_importantes(df_datos_clusterizados)

    return df_datos_clusterizados, df_clusters_descripcion

  
  def get_corr_clust(self, df_datos_clusterizados):
      df_clusterizado_diff = df_datos_clusterizados[['clusters_list']].drop_duplicates()
      df_clusterizado_diff['n_ls'] = df_clusterizado_diff.apply(lambda x: len(x['clusters_list']), axis=1)
      df_expanded = self.expandir_clusters_binario(df_clusterizado_diff,'clusters_list','cluster_')
      cluster_cols = [x for x in df_expanded.columns if 'cluster_' in x]

      df_corr=df_expanded[cluster_cols].corr()

      return df_corr

  def obtener_clusters(self, df_clust, cluster_objetivo, n=5, direccion='ambos'):
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
          raise ValueError("El parámetro 'direccion' debe ser 'arriba', 'abajo', 'ambos' o 'bottom'.")
      
      return top_n.sort_values(ascending=False)

