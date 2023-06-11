from sklearn.feature_selection import VarianceThreshold
import numpy as np

from PreprocessingAlgorithms.preprocessing_base import PreprocessingBase


class ColumnDropper(PreprocessingBase):
  def __init__(self, columns):
    self.columns = columns
    self.columns_left = None

  def fit(self, X, y = None):
    return self

  def transform(self, X):
    X_dropped = X.drop(self.columns, axis = 1)
    self.columns_left = X_dropped.columns
    return X_dropped
  

class VarianceThresholdHandler(PreprocessingBase):
  def __init__(self, thresh):
    self.thresh = thresh
    self.vt = VarianceThreshold(threshold=thresh)
    self.mask = None
    self.columns_left = None

  def fit(self, X, y = None):
    X_copy = X.copy(deep=True)
    X_copy = X_copy.fillna(0)
    X_normalized = X_copy / X_copy.mean()
    _ = self.vt.fit(X_normalized)
    self.mask = self.vt.get_support()
    return self

  def transform(self, X):
    X_dropped = X.loc[:, self.mask]
    self.columns_left = X_dropped.columns
    return X_dropped