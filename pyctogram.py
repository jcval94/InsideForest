# De ley
import numpy as np
import pandas as pd
import random
import datetime

# Manejo de docs
import os
import glob
import zipfile

# Data man
import re
import time
import typer
import seaborn as sns

## Necesario para la extracción de los árboles
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from collections import Counter

#ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import export_text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import pairwise_distances
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


class models:
  def get_knn_rows(self, df, target_col, criterio_fp=True, min_obs = 5):
    X = df.drop(columns=[target_col]).values
    y = df.loc[:, target_col].values
    for k in range(1,int(len(df))):
      knn = KNeighborsClassifier(n_neighbors=k)
      knn.fit(X, y)
      y_pred = knn.predict(X)
      cm = confusion_matrix(y, y_pred)
      tn, fp, fn, tp = cm.ravel()
      if criterio_fp:
        if fp>min_obs:
          break
      else:
        if fn>min_obs:
          break
    if fn>0:
      false_negatives = (y == 1) & (y_pred == 0)
      return df[false_negatives], df[~false_negatives]
    if fp>0:
      false_positives = (y == 0) & (y_pred == 1)
      return df[false_positives], df[~false_positives]
  
  def get_cvRF(self, X_train, y_train, param_grid):
    rf = RandomForestClassifier(random_state=semilla)
    cv = GridSearchCV(rf,param_grid=param_grid,cv=5,n_jobs=-1)
    cv.fit(X_train,y_train)
    return cv


class trees:
  def get_path(self, estructura_iter):
    for i, valor in enumerate(estructura_iter):
      if not(('value: ' in valor) or ('class: ' in valor)):
        continue
      camino = [estructura_iter[0:i], estructura_iter[i]]
      estructura_iter.pop(i)
      estructura_iter.pop(i-1)
      return estructura_iter, camino
  
  def get_rangos(self, regr, data1):
    df_info = []
    for n_estimador in range(len(regr.estimators_)):
      r = export_text(regr.estimators_[n_estimador])
      columnas_nombres = list(data1.columns)
      columnas_nombres.reverse()
      for i, feat in enumerate(columnas_nombres):
        r = r.replace('feature_'+str(len(columnas_nombres)-i-1), feat)
      estructura = r.split('\n')
      estructura_iter = estructura.copy()
      paths = []
      for i, valor in enumerate(estructura):
        if not(('value: ' in valor) or ('class: ' in valor)):
          continue
        estructura_iter, path_ = self.get_path(estructura_iter)
        estructura_rep = [v.count('|') for v in path_[0]]
        if len(estructura_rep)!=len(set(estructura_rep)):
          posiciones_ = []
          for i, valor in enumerate(estructura_rep):
            posiciones = [k for k, v in enumerate(estructura_rep) if valor == v]
            posiciones_ += [x for x in posiciones if x != max(posiciones)]
          path_aux = [i for j, i in enumerate(path_[0]) if j not in set(posiciones_)]
          path_[0] = path_aux
        paths.append([x for x in path_ if x != ''])
      valores = [float(path[1].split(': ')[1].replace(']','')) for path in paths]
      percent_ = np.percentile(valores,90)
      estructuras_maximizadoras = [[pa[0],val] for pa, val in zip(paths,valores) if val>= percent_]
      importanc = []
      for n_path in range(len(estructuras_maximizadoras)):
        importanc += [[v.replace('|---','').replace('|   ','')[1:], 
                      2/((v.count('|'))+1), n_path, n_estimador] 
                      for v in estructuras_maximizadoras[n_path][0]]
      asdf = pd.DataFrame(importanc, columns = ['Regla', 'Importancia', 'N_regla','N_arbol'])
      asdf['Va_Obj_minima'] = percent_
      df_info.append(asdf)
    return pd.concat(df_info)
  
  def get_fro(self, df_full_arboles):
    df_full_arboles['feature'] = df_full_arboles['Regla'].\
    apply(lambda x: re.split(', |>|<=', x)[0][:-1])
    df_full_arboles['rangos'] = df_full_arboles['Regla'].\
    apply(lambda x: float(re.split(', |>|<=', x)[1]))
    df_full_arboles['operador'] = [a[len(b):(len(b)+3)].replace(' ','') for a, b in zip(df_full_arboles['Regla'], df_full_arboles['feature'])]
    return df_full_arboles
  
  def get_summary(self, data1, df_full_arboles, var_obj, verbose):
    agrupacion = pd.pivot_table(df_full_arboles, 
                                index=['N_regla','N_arbol','feature', 'operador'],
                                values=['rangos','Importancia'], aggfunc=['min','max','mean'])
    agrupacion_min = agrupacion['min'].reset_index()
    agrupacion_min = agrupacion_min[agrupacion_min['operador']=='<=']
    agrupacion_max = agrupacion['max'].reset_index()
    agrupacion_max = agrupacion_max[agrupacion_max['operador']=='>']
    agrupacion_mean = agrupacion['mean'].reset_index()
    agrupacion = pd.concat([agrupacion_min,agrupacion_max]).sort_values(['N_arbol','N_regla'])
    top_100_ramas = agrupacion.N_arbol.unique()[:100]
    reglas = []
    for arbol_num in top_100_ramas:
      if arbol_num%50==0 and verbose:
        print(arbol_num)
      ag_arbol = agrupacion[(agrupacion['N_arbol']==arbol_num)]
      for regla_num in ag_arbol.N_regla.unique():
        data1_ = data1.copy()
        ag_regla = ag_arbol[(ag_arbol['N_regla']==regla_num)]
        men_ = ag_regla[(ag_regla['operador']=='<=')][['feature','rangos']].values
        may_ = ag_regla[(ag_regla['operador']=='>')][['feature','rangos']].values
        if len(men_)>0:
          for col, valor in men_:
            data1_ = data1_.loc[data1_[col]<=valor,:]
            for col, valor in may_:
              data1_ = data1_.loc[data1_[col]>valor,:]
        else:
          for col, valor in may_:
              data1_ = data1_.loc[data1_[col]>valor,:]
              for col, valor in men_:
                data1_ = data1_.loc[data1_[col]<=valor,:]
        ag_regla_copy = ag_regla.copy()
        ag_regla_copy.loc[:, 'n_sample'] = len(data1_)
        ag_regla_copy.loc[:, 'ef_sample'] = data1_[var_obj].mean()
        reglas.append(ag_regla_copy)
    agrupacion = pd.concat(reglas)
    agrupacion = agrupacion.sort_values(by=['ef_sample','n_sample'], ascending=False)
    return agrupacion
  
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
        rectangulo_m = sublista[:-1]
        ponde_ = sublista[-1]
        for rectangulo, intervalo in rectangulo_m:
            linf, lsup = intervalo
            registros__.append([i + 1, rectangulo, linf, lsup,ponde_[1]])
    df = pd.DataFrame(registros__, columns=['rectangulo', 'dimension', 'linf', 'lsup','ponderador'])
    return df
  
  def generate_key(self, r):
    r = r[1]
    ndims_ = len(r)
    dimensiones__ = '*'.join(sorted([r[i][0] for i in range(ndims_)][:-1]))
    return dimensiones__
  
  def get_dfs_dim(self, rectangles_):
    l_rectangles_ = list(rectangles_.values())
    llaves = [(self.generate_key(r)) for  r in (rectangles_.items())]
    ponderadores = [(self.generate_key(r)) for  r in (rectangles_.items())]
    llaves_unicas = list(set(llaves))
    separacion_dim = {lun:[l_rectangles_[i] for i in 
                           [i for i, llave in enumerate(llaves) if llave==lun]]
    for lun in llaves_unicas}
    separacion_dim = [self.rect_l_to_df(separacion_dim, k) for k, v in separacion_dim.items()]
    return separacion_dim
  
  def extract_rectangles(self, df_summ):
    grouped = df_summ[(df_summ['n_sample']>0)&
                      (df_summ['ef_sample']>0)].groupby(['N_arbol', 'N_regla'])
    rectangles_ = {}
    for name, group in grouped:
        rectangles_[name] = self.get_rect_coords(group)
    agrupacion_media = grouped.mean()
    agrupacion_media['ponderacion'] = (agrupacion_media['n_sample']\
    /agrupacion_media['n_sample'].max())+agrupacion_media['ef_sample']*5
    agrupacion_media.sort_values('ponderacion')
    for k in rectangles_.keys():
      rectangles_[k] += [('ponderador',agrupacion_media.loc[k,'ponderacion'])]
    separacion_dim = self.get_dfs_dim(rectangles_)
    return separacion_dim



  def get_branches(self,df, var_obj, regr):
    X = df.drop(columns=[var_obj]).fillna(0)
    # y = df[var_obj]
    df_full_arboles = self.get_rangos(regr, X)

    # La variable N_regla indica el número de regla que se está utilizando para realizar la clasificación. 
    # Si N_regla es igual a 0, significa que se está utilizando la primera regla, y si es igual a 1, significa que se está utilizando la segunda regla.
    df_full_arboles = self.get_fro(df_full_arboles)

    df_summ = self.get_summary(df, df_full_arboles,var_obj, False)

    separacion_dim = self.extract_rectangles(df_summ)

    return separacion_dim

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


class labels:
  def round_values(self, lst):
    variance = np.var(lst)
    if variance >= 0.01 and len(lst)>1:
      return [round(val, 2) for val in lst]
    else:
      return ['{:.2e}'.format(val) for val in lst]
  
  def custom_round(self, x):
    if abs(x) > 100 or abs(x - int(x)) < 1e-10:
      return int(x)
    elif abs(x) < 0.01:
      return "{:.2e}".format(x)
    else:
      return round(x, 3)
  
  def get_intervals(self, df__):
    df__ = self.drop_altura_columns(df__)
    df__ = df__.applymap(self.custom_round)
    resuld = []
    for i in range(len(df__)):
      row_result = []
      for col in df__[['linf']].columns:
        valor1 = df__.iloc[i][('linf',col[-1])]
        valor2 = df__.iloc[i][('lsup',col[-1])]
        if valor1 == valor2:
          continue
        row_result.append(f"{col[-1]} entre {valor1} y {valor2}")
      vvva = " | ".join(row_result)
      resuld.append(vvva)
    return resuld
  
  def drop_altura_columns(self, df):
    altura_columns = [col for col in df.columns if 'altura' in col[1]]
    df = df.drop(altura_columns, axis=1)
    return df


