from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

class BoxCoxTransform(BaseEstimator, TransformerMixin):
  def __init__(self, skewing_threshold):
    self.skewing_threshold = skewing_threshold
    self.skew_index = None

  def fit(self, X, y = None):
    X_out = X.copy(deep=True)
    X_out = X_out.fillna(0)
    skew_features = X_out.apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skew_features[skew_features > self.skewing_threshold]
    self.skew_index = high_skew.index
    return self

  def transform(self, X):
    X_out = X.copy(deep=True)
    X_out = X_out.fillna(0)
    for i in self.skew_index:
      X_out[i] = boxcox1p(X_out[i], boxcox_normmax(X_out[i] + 1))
    return X_out
