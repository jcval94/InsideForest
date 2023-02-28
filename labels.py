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
