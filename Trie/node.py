from sklearn.pipeline import Pipeline, FeatureUnion

class NodeIDHandler:
  def __init__(self):
    self._id = 0

  def get_next(self):
    self._id += 1
    return self._id

node_id = NodeIDHandler()

class PreprocessNode:
  def __init__(self, key, obj = None):
    self.children_nodes = {}
    self.key = key
    self.obj = obj
    self.id = node_id.get_next()
    self.processed = False
    self.df_out = None
    
  def __str__(self):
    return f"{self.key}_{self.id}"

  def __repr__(self):
    return str(self)

  def add_child(self, key, obj):
    if isinstance(obj, PreprocessNode):
      self.children_nodes[key] = obj
    elif not key in self.children_nodes.keys():
      n = PreprocessNode(key, obj)
      self.children_nodes[key] = n
    
    return self.children_nodes[key]

  def process(self, df, experiment_handler):
    df_out = df.copy(deep=True)
    if self.processed == False:
      print(f'\t Start fitting...')
      self.obj.set_experiment_handler(experiment_handler)
      self.obj.fit(df_out)
      self.processed = True
      print(f'\t Fitting done!')
    else:
      print('\tCACHED!')
    df_out = self.obj.transform(df_out)
    print(f'\t{df_out.shape}\n')
    return df_out

  def add_path(self, key_fn_pairs):
    #key_fn_pairs = list(filter(lambda x: x[0] != 'none', key_fn_pairs))
    path_out = []
    n = self
    for key, fn in key_fn_pairs:
      if isinstance(fn, PreprocessNode):
        n = n.add_child(key, fn)
      elif fn is None:
        n = n.add_child(key, None)
      else:

        n = n.add_child(key, fn())
      path_out.append(n)

    return path_out

  def process_path(self, path_as_strings, df, experiment_handler):
    n = self 
   
    df_processed = df.copy(deep=True)
    for idx, p in enumerate(path_as_strings):
      n = n.children_nodes[p]
      print(f'({idx + 1} / {len(path_as_strings)}) Preprocess: {n.obj.__class__.__name__}')
      df_processed = n.process(df_processed, experiment_handler)

    return df_processed


  def get_path(self, path_as_strings):
    path = []
    n = self
    for p in path_as_strings:
      n = self.children_nodes[p]
      path.append(n)
    return path

  def path_to_pipeline(self, path_as_strings):
    steps = []
    n = self
    print(n.children_nodes.keys())
    for p in path_as_strings:
      n = n.children_nodes[p]
      steps.append((p, n.obj))

    return Pipeline(steps=steps)


    