import numpy as np
import logging

logger = logging.getLogger(__name__)


class Labels:
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
    df__ = self.drop_height_columns(df__)
    df__ = df__.applymap(self.custom_round)
    resuld = []
    for i in range(len(df__)):
      row_result = []
      for col in df__[['linf']].columns:
        valor1 = df__.iloc[i][('linf',col[-1])]
        valor2 = df__.iloc[i][('lsup',col[-1])]
        if valor1 == valor2:
          continue
        row_result.append(f"{col[-1]} between {valor1} and {valor2}")
      vvva = " | ".join(row_result)
      resuld.append(vvva)
    return resuld
  
  def drop_height_columns(self, df):
    height_columns = [col for col in df.columns if 'altura' in col[1]]
    df = df.drop(height_columns, axis=1)
    return df

  def get_branch(self,df, df_sub, i):
    
    df_sub.reset_index(inplace=True,drop=True)

    if not set(df_sub.columns.get_level_values(1)).issubset(df.columns):
      missing = set(df_sub.columns.get_level_values(1)) - set(df.columns)
      raise KeyError(f"Columns {missing} do not exist in the main DataFrame")

    if i >= len(df_sub):
      return None
    limitador_inf = df_sub.loc[i,'linf'].copy()
    limitador_sup = df_sub.loc[i,'lsup'].copy()
    vars_lm = list(limitador_sup.index)

    conds = [(df[va]<=limitador_sup[va])&(df[va]>limitador_inf[va]) for va in vars_lm]

    # Initialize variable with the only condition from the list
    variable_cd = conds[0]

    # If there is more than one condition, combine them with &
    if len(conds) > 1:
        for condicion in conds[1:]:
            variable_cd = variable_cd & condicion

    return df[variable_cd]


  def get_labels(self,df_reres, df, var_obj, etq_max = 9,ramas = 10):

    labels_list = []
    for j in range(ramas-1):
      if j>len(df_reres):
        continue
      df_ppr = df_reres[j].copy()
      df_ppr = df_ppr[[(a, b) for a, b in df_ppr.columns if 'altura' != b]]
      desc_vars = self.get_intervals(df_ppr.head(etq_max))
      try:
        ramas_ = [self.get_branch(df, df_ppr, i) for i in range(0,etq_max+1)]
        scores_pop = [
            (x[var_obj].mean(), x[var_obj].count()) for x in ramas_ if x is not None
        ]
        target_population = [x[x[var_obj]==0] for x in ramas_ if x is not None]
      except KeyError as exc:
        logger.exception("Missing columns when obtaining labels: %s", exc)
        continue
      if len(target_population)==0:
        continue
      labels_dict = {etq_:[sc_, po_]for po_, sc_, etq_ in zip(target_population, scores_pop, desc_vars) if po_.shape[0]>0}
      labels_list.append(labels_dict)

    return labels_list
  
