from sklearn.base import BaseEstimator, TransformerMixin
from PreprocessingAlgorithms.Utils.optimizer_utils import SuggestBase
from PreprocessingAlgorithms.Utils.pte_optimizer import OptimizerWrapper


class BuildingPart:
  def __init__(self, cls_, kwargs = None):

    self.kwargs = kwargs
    if self.kwargs == None:
      self.kwargs = {}

    self.cls_ = cls_
    pass

  def __call__(self):
    is_to_optimize = False
    for k in self.kwargs.keys():
      if isinstance(self.kwargs[k], SuggestBase):
        is_to_optimize = True

    if is_to_optimize:
      return OptimizerWrapper(self.cls_, self.kwargs)

    #return lambda: self.cls_(**self.kwargs)
    return self.cls_(**self.kwargs)

def build(cls, **kwargs):
  print(kwargs)
  is_to_optimize = False
  for k in kwargs.keys():
    if isinstance(kwargs[k], SuggestBase):
      is_to_optimize = True
  
  if is_to_optimize:
    pass

  return lambda: cls(**kwargs)

class Identity(BaseEstimator, TransformerMixin):
  def fit(self, X, y = None):
    return self

  def transform(self, X):
    return X