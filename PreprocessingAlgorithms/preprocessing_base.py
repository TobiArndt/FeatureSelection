from sklearn.base import BaseEstimator, TransformerMixin
from abc import abstractmethod
class PreprocessingBase(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.experiment_handler = None

    def set_experiment_handler(self, experiment_handler):
        self.experiment_handler = experiment_handler

    @abstractmethod
    def fit(self, X, y=None):
        return self
    
    @abstractmethod
    def transform(self, X):
        return X