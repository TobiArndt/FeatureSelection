import pandas as pd

class ExperimentExecuter:
  def __init__(self, target, objective_fn, model_generator_fn, models_to_check):
    self.objective_fn = objective_fn
    self.target = target
    self.results = pd.DataFrame()
    self.scores = []
    self.model_generator_fn = model_generator_fn
    self.models_to_check = models_to_check
    



  def next_experiment(self, path, dataset):
    X = dataset.drop([self.target], axis=1) 
    y = dataset[self.target]
    models = self.model_generator_fn()
    run_scores = []
    for k in self.models_to_check:
      m = models[k]
      X = X.fillna(0)
      print(f'\t Starting {k}...')
      score = self.objective_fn(m, X, y)
      name = '-'.join(path)
      result = {'experiment': name, 'model_name': k, 'mean': score.mean(),  'std': score.std(), 'num_rows': dataset.shape[0], 'cols': dataset.shape[1]}
      run_scores.append(result)
      self.scores.append(result)
      print(f'\t Done {k}.')
      print(f'\t{result}')
      print(20*'-')
    
    run_scores.sort(key=lambda x: x['mean'])
    print(f"\t Best mean: {run_scores[0]['mean']}, model: {run_scores[0]['model_name']}")
    print(40*'*')
    return run_scores[0]['mean']

  def to_df(self):
    return pd.DataFrame(self.scores)