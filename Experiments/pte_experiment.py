import pandas as pd

class ExperimentExecuter:
  def __init__(self, target, objective_fn):
    self.objective_fn = objective_fn
    self.target = target
    self.results = pd.DataFrame()
    self.scores = []

    
  def copy(self):
    return ExperimentExecuter(
      target = self.target,
      objective_fn= self.objective_fn,
    )


  def next_experiment(self, model, path, dataset, model_params = None):
    X = dataset.drop([self.target], axis=1) 
    y = dataset[self.target]
    
    X = X.fillna(0)
    print(f'\t Starting...')
    score = self.objective_fn(model, X, y)
    name = '-'.join(path)
    result = {'experiment': name, 'mean': score.mean(),  'std': score.std(), 'num_rows': dataset.shape[0], 'cols': dataset.shape[1]}
    if model_params is not None:
        result.update(model_params)
    
    dataset.to_csv(f'/content/{str(score.mean())[:5].replace(".", "_")}_{name}_cols{dataset.shape[1]}.csv')

    self.scores.append(result)
    print(f'\t Done.')
    print(f'\t{result}')
    print(20*'-')

    
    return score.mean()

  def to_df(self):
    return pd.DataFrame(self.scores)
