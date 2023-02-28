from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import pandas as pd
import numpy as np
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
    ax.figure.colorbar(sm)

    
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

