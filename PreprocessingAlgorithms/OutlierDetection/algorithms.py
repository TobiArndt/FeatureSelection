from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class OutlierIQR(BaseEstimator, TransformerMixin):
  def fit(self, X, y = None):
    return self

  def transform(self, X):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    X_out = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    return X_out

class OutlierLOF(BaseEstimator, TransformerMixin):
  def __init__(self, n_neighbours=20, contamination='auto'):
    self.n_neighbours = n_neighbours
    self.contaminatin = contamination
    self.lof = LocalOutlierFactor(
        n_neighbors=self.n_neighbours,
        contamination=self.contamination
    )

  def fit(self, X, y = None):
    return self

  def transform(self, X):
    X_out = X.copy(deep=True)
    X_out = X_out.fillna(0)
    y_pred = self.lof.fit_predict(X_out)
    X_out = X_out[np.where(y_pred == 1, True, False)]
    return X_out
    '''
    X_out = X.copy(deep=True)
    X_out = X_out.fillna(0)
    y_pred = self.estimator.predict(X_out)
    X_out = X_out[np.where(y_pred == 1), True, False]
    return X_out
    '''
