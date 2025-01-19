from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import DBSCAN, KMeans

import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import Rectangle
from collections import Counter



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

    
  def get_agg_regions(self, df_eval, df, verbose=False):
    features_val = sorted(df_eval['dimension'].unique())
    aleatorio1 = features_val[0]
    df_sep_dm = pd.pivot_table(df_eval, index='rectangulo', columns='dimension')

    df_sep_dm = self.fill_na_pond(df_sep_dm, df, features_val)
    
    df_m_medios = self.mean_distance_ndim(df_sep_dm)
    scaler = StandardScaler()
    X_feat = scaler.fit_transform(df_m_medios.values)
    epsil = self.get_eps_multiple_groups_opt(X_feat)
    if epsil<=0:
        epsil = 0.05
    if verbose:
      print(epsil)
    dbscan = DBSCAN(eps=epsil*.95, min_samples=2)
    df_sep_dm['cluster'] = dbscan.fit_predict(X_feat)
    df_m_medios['cluster'] = dbscan.fit_predict(X_feat)
    col_dim_names = {x:([len,'mean','std']) if i == 0 else ['mean','std'] 
                    for i, x in enumerate(df_sep_dm.columns) if x[0] not in ['cluster',
                                                                              'ponderador']}
    col_dim_names[('ponderador', aleatorio1)] = ['mean']
    df_sep_outl = df_sep_dm[df_sep_dm['cluster']==-1]
    df_sep_noutl = df_sep_dm[df_sep_dm['cluster']!=-1]
    if df_sep_noutl.shape[0]==0:
      df_sep_dm_agg = df_sep_outl.drop(columns=['cluster'])
      return df_sep_dm_agg
    else:
      df_sep_dm_agg = pd.pivot_table(df_sep_noutl,
                                    index='cluster',
                                    values = [x for x in df_sep_noutl.columns if x[0]!='cluster'],
                                    aggfunc=col_dim_names)
      ln_col_ = [x for x in df_sep_dm_agg.columns if 'len' in list(x)]
      df_sep_dm_aggl = df_sep_dm_agg[ln_col_[0]]
      max_l_val = 1/df_sep_dm_aggl.max()
      df_sep_dm_aggl = df_sep_dm_aggl*max_l_val
      df_sep_dm_agg['ponderador'] = df_sep_dm_agg['ponderador'].values*1
      df_sep_dm_agg = df_sep_dm_agg.sort_values(by=[('ponderador', aleatorio1,'mean'),
                                    ('linf', aleatorio1,'len')], ascending=False)
      df_sep_dm_agg = df_sep_dm_agg.xs('mean', axis=1, level=2)
      df_sep_dm_aggl = df_sep_dm_aggl/df_sep_dm_aggl.max()
      df_sep_outl = df_sep_outl[df_sep_dm_agg.columns]
      df_sep_outl.loc[:,'ponderador'] = df_sep_outl.loc[:,'ponderador'].values*max_l_val
      return pd.concat([df_sep_dm_agg,df_sep_outl])
  
  def prio_ranges(self, separacion_dim, df):
    # aquí se usa DBS
    df_res = [self.get_agg_regions(df_, df) for df_ in separacion_dim]
    prio_ = [df_['ponderador'].values[0][0] for df_ in df_res]
    df_reres = [x[0] for x in sorted([(a, b) for a,b in zip(df_res,prio_)],
                     key=lambda x: -x[1])]
    cols_ = [df_['linf'].columns.tolist() for df_ in df_reres]
    return df_reres


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
      df['cluster'] = range(cluster_id, cluster_id + n_reglas)
      cluster_id += n_reglas
    return lista_reglas

  def generar_descripcion_clusters(self, df_reglas):
    cluster_descripciones = []

    for idx, row in df_reglas.iterrows():
      cluster_id = row['cluster']
      linf = row['linf'].dropna()
      lsup = row['lsup'].dropna()

      # Generar descripción de la regla
      descripcion_partes = []
      for var in linf.index:
        linf_val = linf[var]
        lsup_val = lsup[var]
        descripcion_partes.append(f"{linf_val} <= {var} <= {lsup_val}")
      descripcion = " AND ".join(descripcion_partes)

      # Añadir ponderador si es relevante
      ponderador = row['ponderador'].mean()

      cluster_descripciones.append({
        'cluster': cluster_id.values[0],
        # 'id': cluster_id,
        'cluster_descripcion': descripcion,
        'cluster_ponderador': ponderador
      })

    df_clusters_descripcion = pd.DataFrame(cluster_descripciones)

    return df_clusters_descripcion



  def asignar_clusters_a_datos(self, df_datos, df_reglas, keep_all_clusters=True):
      """
      Asigna clusters a los datos en base a reglas.

      Parámetros
      ----------
      df_datos : pd.DataFrame
          Dataframe con los datos a asignar a clusters.
      df_reglas : pd.DataFrame
          Dataframe con la definición de las reglas (linf, lsup, ponderador, etc.)
      keep_all_clusters : bool, optional (default=True)
          Si es True, se añadirán cuatro columnas adicionales:
            - 'n_clusters': número de clusters en los que cae el registro.
            - 'clusters_list': lista de todos los clusters a los que pertenece ese registro.
            - 'ponderadores_list': lista de los ponderadores de los clusters a los que pertenece.
            - 'ponderador_mean': media de los ponderadores de los clusters a los que pertenece.
          Si es False, se comportará como la función original, asignando sólo el cluster
          de mayor ponderador (sin columnas extras).

      Returns
      -------
      df_datos_clusterizados : pd.DataFrame
          Dataframe de entrada con la columna 'cluster' asignada.
          Si keep_all_clusters=True, además se incluyen 'n_clusters', 'clusters_list',
          'ponderadores_list' y 'ponderador_mean'.
      df_clusters_descripcion : pd.DataFrame
          Dataframe con la descripción (o métricas) de cada cluster.
      """

      import numpy as np

      n_datos = len(df_datos)
      # Array para almacenar el cluster "principal" (mayor ponderador)
      clusters_datos = np.full(n_datos, -1, dtype=float)
      # Array para almacenar el ponderador del cluster principal
      ponderador_datos = np.full(n_datos, -np.inf, dtype=float)

      # Si vamos a guardar todos los clusters por registro, inicializamos estructuras
      if keep_all_clusters:
          clusters_datos_all = [[] for _ in range(n_datos)]
          ponderadores_datos_all = [[] for _ in range(n_datos)]

      # --- 1) Extraer información de las reglas y normalizarla ---
      reglas_info = []
      for idx, row in df_reglas.iterrows():
          linf = row['linf'].dropna()
          lsup = row['lsup'].dropna()

          variables = linf.index.tolist()  # las mismas variables en linf y lsup
          # En caso de que 'ponderador' sea un valor o iterable, asumimos la media como en tu código
          ponderador = row['ponderador'].mean() if hasattr(row['ponderador'], '__iter__') else row['ponderador']

          # row['cluster'] a veces viene como Serie de un solo valor. Convertirlo a float si es el caso.
          # Por ejemplo: dimension    0.0
          #             Name: 0, dtype: float64
          cluster_raw = row['cluster']
          # Si cluster_raw es Serie con un solo valor, obtenemos ese valor:
          if hasattr(cluster_raw, 'values') and len(cluster_raw.values) == 1:
              cluster_raw = float(cluster_raw.values[0])
          else:
              # Asumir que es un float/entero directamente
              cluster_raw = float(cluster_raw)

          reglas_info.append({
              'variables': variables,
              'linf': linf.to_dict(),
              'lsup': lsup.to_dict(),
              'ponderador': ponderador,
              'cluster': cluster_raw,
          })

      # --- 2) Verificar para cada dato qué reglas cumple ---
      for regla in reglas_info:
          variables = regla['variables']
          linf = regla['linf']
          lsup = regla['lsup']
          ponderador = regla['ponderador']
          cluster = regla['cluster']

          # Extraer las columnas relevantes para evaluar la regla
          X_datos = df_datos[variables]

          # Verificar fila a fila si cumple la regla
          cumple_regla = np.ones(n_datos, dtype=bool)
          for var in variables:
              cumple_regla &= (X_datos[var] >= linf[var]) & (X_datos[var] <= lsup[var])

          # Si queremos mantener el listado de todos los clusters que cumple
          if keep_all_clusters:
              # Para cada fila que cumpla la regla, agregamos el cluster y su ponderador a sus listas
              indices_cumple = np.where(cumple_regla)[0]
              for i in indices_cumple:
                  clusters_datos_all[i].append(cluster)
                  ponderadores_datos_all[i].append(ponderador)

          # Independiente de keep_all_clusters, mantenemos la lógica de "un solo cluster final"
          # usando el de mayor ponderador
          actualizar = (cumple_regla) & (ponderador > ponderador_datos)
          clusters_datos[actualizar] = cluster
          ponderador_datos[actualizar] = ponderador

      # --- 3) Construir dataframe de salida ---
      df_datos_clusterizados = df_datos.copy()
      df_datos_clusterizados['cluster'] = clusters_datos  # cluster final (mayor ponderador)

      # Si se solicitó conservar todos los clusters, agregamos las 4 columnas
      if keep_all_clusters:
          # Número de clusters por registro
          df_datos_clusterizados['n_clusters'] = [len(lst) for lst in clusters_datos_all]
          # Lista de clusters por registro
          df_datos_clusterizados['clusters_list'] = clusters_datos_all
          # Lista de ponderadores por registro
          df_datos_clusterizados['ponderadores_list'] = ponderadores_datos_all
          # Media de los ponderadores por registro (maneja registros sin clusters)
          df_datos_clusterizados['ponderador_mean'] = [
              np.mean(lst) if lst else np.nan for lst in ponderadores_datos_all
          ]

      # --- 4) Generar la descripción de clusters (tal como en tu código original) ---
      df_clusters_descripcion = self.generar_descripcion_clusters(df_reglas)
      # return df_datos_clusterizados
      return df_datos_clusterizados, df_clusters_descripcion



  def eliminar_reglas_redundantes(self, lista_reglas):
    # Concatenar todos los dataframes en uno solo
    df_reglas = pd.concat(lista_reglas, ignore_index=True)

    # Asegurarse de que los nombres de los niveles sean consistentes
    if df_reglas.columns.names != [None, 'dimension']:
      df_reglas.columns.names = [None, 'dimension']

    n_reglas = len(df_reglas)

    # Inicializar listas para almacenar información de cada regla
    reglas_info = []

    for idx, row in df_reglas.iterrows():
      # Obtener las variables para las que esta regla tiene límites
      linf = row['linf']
      lsup = row['lsup']
      ponderador = row['ponderador'].mean()  # Promedio si hay varios ponderadores

      variables = linf.dropna().index.tolist()  # Variables con linf definido

      # Crear diccionarios para linf y lsup
      linf_dict = linf.to_dict()
      lsup_dict = lsup.to_dict()

      reglas_info.append({
        'idx': idx,
        'variables': set(variables),
        'linf': linf_dict,
        'lsup': lsup_dict,
        'ponderador': ponderador,
      })

    # Marcar reglas redundantes
    reglas_redundantes = set()

    for i in range(n_reglas):
      regla_i = reglas_info[i]
      if regla_i['idx'] in reglas_redundantes:
        continue  # Ya marcada como redundante

      for j in range(n_reglas):
        if i == j:
          continue

        regla_j = reglas_info[j]

        # Condición 1: variables_j ⊆ variables_i
        if not regla_j['variables'].issubset(regla_i['variables']):
          continue

        # Condición 2: linf_i >= linf_j y lsup_i <= lsup_j para variables comunes
        es_contenida = True
        for var in regla_j['variables']:
          if (regla_i['linf'][var] >= regla_j['linf'][var]) and (regla_i['lsup'][var] <= regla_j['lsup'][var]):
            continue
          else:
            es_contenida = False
            break

        if not es_contenida:
          continue

        # Condición 3: ponderador_j >= ponderador_i
        if regla_j['ponderador'] >= regla_i['ponderador']:
          # Marcar regla_i como redundante
          reglas_redundantes.add(regla_i['idx'])
          break  # No es necesario comparar con más reglas

    # Conservar solo las reglas no redundantes
    df_reglas_importantes = df_reglas.drop(index=reglas_redundantes).reset_index(drop=True)

    return df_reglas_importantes


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




  def labels(self, df, df_reres, include_desc=False):
    lista_reglas = copy.deepcopy(df_reres)

    # Asignar IDs únicos a las reglas
    lista_reglas = self.asignar_ids_unicos(lista_reglas)

    # Eliminar reglas redundantes
    df_reglas_importantes = self.eliminar_reglas_redundantes(lista_reglas)

    df_reglas_importantes = df_reglas_importantes.reset_index()
    df_reglas_importantes.rename(columns={'index': 'cluster'}, inplace=True)
    
    # Asignar clusters a los datos utilizando las reglas importantes
    df_datos_clusterizados, df_clusters_descripcion = self.asignar_clusters_a_datos(df, df_reglas_importantes)
    
    df_datos_clusterizados = self.get_clusters_importantes(df_datos_clusterizados)

    if include_desc:
      df_datos_clusterizados = df_datos_clusterizados.merge(df_clusters_descripcion, on='cluster', how='left')
      
      df_datos_clusterizados = df_datos_clusterizados.rename(columns={'cluster_descripcion':'best_cluster_descripcion','cluster_ponderador':'best_cluster_ponderador','cluster':'best_cluster'})
      
      return df_datos_clusterizados

    return df_datos_clusterizados, df_clusters_descripcion
  