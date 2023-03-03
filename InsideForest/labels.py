import numpy as np


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

  def get_branch(self,df, df_sub, i):
    
    df_sub.reset_index(inplace=True,drop=True)

    if i >= len(df_sub):
      return None
    limitador_inf = df_sub.loc[i,'linf'].copy()
    limitador_sup = df_sub.loc[i,'lsup'].copy()
    vars_lm = list(limitador_sup.index)

    conds = [(df[va]<=limitador_sup[va])&(df[va]>limitador_inf[va]) for va in vars_lm]

    # Inicializar la variable con la única condición de la lista
    variable_cd = conds[0]

    # Verificar si hay más de una condición en la lista
    if len(conds) > 1:
        # Si hay más de una condición, concatenarlas con &
        for condicion in conds[1:]:
            variable_cd = variable_cd & condicion

    return df[variable_cd]


  def get_labels(self,df_reres, df, var_obj, etq_max = 9,ramas = 10):
    
    etiquetas_list = []
    for j in range(ramas-1):
      if j>len(df_reres):
        continue
      df_ppr = df_reres[j].copy()
      df_ppr = df_ppr[[(a, b) for a, b in df_ppr.columns if 'altura' != b]]
      descripcion_vrs = self.get_intervals(df_ppr.head(etq_max))
      ramas_ = [self.get_branch(df, df_ppr, i) for i in range(1,etq_max+1)]
      scores_pob = [(x[var_obj].mean(), x[var_obj].count()) for x in ramas_ if x is not None]
      poblacion_objetivo = [x[x[var_obj]==0] for x in ramas_ if x is not None]
      if len(poblacion_objetivo)==0:
        continue
      dicci = {etq_:[sc_, po_]for po_, sc_, etq_ in zip(poblacion_objetivo, scores_pob, descripcion_vrs) if po_.shape[0]>0}
      etiquetas_list.append(dicci)

    return etiquetas_list
  
