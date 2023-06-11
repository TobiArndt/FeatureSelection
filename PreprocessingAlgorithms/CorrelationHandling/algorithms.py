from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class VifColumns(BaseEstimator, TransformerMixin):
  def __init__(self, vif_thresh):
    self.vif_thresh = vif_thresh
    self.columns_to_drop = None

  def _compute_vif(self, df):
    X = add_constant(df)
    vif = pd.Series([variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])],
                    index = X.columns)
    return vif

  def transform(self, df):
    df_out = df.copy(deep = True)
    for i in self.columns_to_drop:
      df_out = df_out.drop([i], axis=1)
    return df_out

  def fit(self, X, y = None):
    X_out = X.copy(deep = True)
    X_out = X_out.fillna(0)
    self.columns_to_drop = []

    while True:
      vif = self._compute_vif(X_out)
      col = vif.where(vif > self.vif_thresh).sort_values(ascending=False).dropna()
      if len(col) <= 1:
        break
      self.columns_to_drop.append(col.index[1])
      X_out = X_out.drop([col.index[1]], axis=1)
      print(10*'*')
      print(col)

    return self