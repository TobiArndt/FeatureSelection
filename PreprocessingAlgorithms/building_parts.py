from sklearn.base import BaseEstimator, TransformerMixin

def build(cls, **kwargs):
  return lambda: cls(**kwargs)

class Identity(BaseEstimator, TransformerMixin):
  def fit(self, X, y = None):
    return self

  def transform(self, X):
    return X