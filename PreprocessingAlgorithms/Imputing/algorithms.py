from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge

import pandas as pd


class ImputerIterative(BaseEstimator, TransformerMixin):
  def __init__(self, estimator, max_iter, tolerance, random_state = 42):
    self.tolerance = tolerance
    self.random_state = random_state
    self.max_iter = max_iter
    #self.estimator_ = estimator
    if estimator == 'BayesianRidge':
      estimator = BayesianRidge(verbose=False)

    self.estimator = IterativeImputer(
      random_state=random_state,
      estimator=estimator,
      max_iter = max_iter,
      tol=tolerance
    )

  def fit(self, X, y = None):
    self.estimator.fit(X)
    return self

  def transform(self, X):
    X_out = X.copy(deep=True)
    imputed = self.estimator.transform(X_out)
    X_out = pd.DataFrame(imputed, columns=X_out.columns)
    return X_out

class ImputerStupid(BaseEstimator, TransformerMixin):
  def __init__(self, fill_value):
    self.fill_value = fill_value

  def fit(self, X, y = None):
    return self

  def transform(self, X):
    return X.fillna(self.fill_value)
