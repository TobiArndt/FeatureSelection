import operator
import numpy as np
import optuna
from PreprocessingAlgorithms.Utils.optimizer_utils import SuggestBase
from PreprocessingAlgorithms.preprocessing_base import PreprocessingBase
class EarlyStoppingCallback:
    def __init__(self, early_stopping_rounds, direction = "minimize"):
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == 'minimize':
            self._operator = operator.lt
            self._score = np.inf
        elif direction == 'maximize':
            self._operator = operator.gt
            self._score = -np.inf
       

    def __call__(self, study, trial):
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


class OptimizerWrapper(PreprocessingBase):
    def __init__(self, cls_, kwargs):
        self.kwargs = kwargs
        self.cls_ = cls_

        args = self.create_optimizer_args()
        self.args_to_optimize = args['to_optimize']
        self.args_to_keep = args['to_keep']

        self.dataset = None
        self.y = None
        self.experiment_handler = None
        
        self.obj = None
        self.objs_by_params = {}

        self.trial_num = 0
        self.optimized = False

    def create_optimizer_args(self):
        args_to_optimize = {}
        args_to_keep = {}
        for k in self.kwargs.keys():
            if isinstance(self.kwargs[k], SuggestBase):
                args_to_optimize[k] = self.kwargs[k]
            else:
                args_to_keep[k] = self.kwargs[k]


        return  {
                'to_optimize': args_to_optimize,
                'to_keep': args_to_keep
                }
    
    def objective(self, trial):
        self.trial_num += 1

        args_out = {}
        sugessted_args = {}
        for k in self.args_to_optimize.keys():
            arg = self.args_to_optimize[k]
            suggested = arg.suggest(trial, k)

            args_out[k] = suggested
            sugessted_args[k] = suggested

        for k in self.args_to_keep.keys():
            arg = self.args_to_keep[k]
            args_out[k] = arg


        preprocess_stage = self.cls_(**args_out)
        preprocess_stage.fit(self.dataset, self.y)

        self.objs_by_params[tuple(sugessted_args.values())] = preprocess_stage

        df_out = preprocess_stage.transform(self.dataset.copy())

        best_score = self.experiment_handler.next_experiment([f"{self.trial_num}"], df_out)
        return best_score
    

    def start_optimizing(self):
        study = optuna.create_study(direction='minimize')
        early_stopping = EarlyStoppingCallback(5, direction='minimize')
        study.optimize(self.objective, callbacks=[early_stopping], n_trials=5)
        return self.objs_by_params[tuple(study.best_params.values())]

    def fit(self, X, y=None):
        if self.optimized:
            return self
        
        self.experiment_handler = self.experiment_handler.copy()
        self.dataset = X.copy(deep=True)
        self.y = y
        self.obj = self.start_optimizing()
        self.optimized = True
        return self
            
    def transform(self, X):
        return self.obj.transform(X)