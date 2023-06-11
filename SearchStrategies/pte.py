from construction import A
import optuna

class Search:
    def __init__(self, trie, experiment_handler):
        self.trie = trie
        self.all_building_blocks = trie.building_blocks
        self.experiment_handler = experiment_handler
        self.potential_stages = self.create_potential_stages()
        self.study = optuna.create_study(direction='minimize', study_name='regression')
        self.dataset = None
        self.cache = {}


    def create_potential_stages(self):
        all_stages = {}
        for idx, stage in enumerate(self.all_building_blocks):
            elements = []
            for element in stage:
                elements.append(A(element[0]).name)

            all_stages[f"stage_{idx}"] = elements
        
        return all_stages

    def get_stages(self, trial):
        path = []
        for k in self.potential_stages:
            next_stage = trial.suggest_categorical(k, self.potential_stages[k])
            if next_stage != 'Optional':
                path.append(next_stage)

        return path
    
    def objective(self, trial):
        path = self.get_stages(trial)
        if tuple(path) in self.cache.keys():
            return self.cache[tuple(path)]
        
        print(f'Path to check: {path}')
        df = self.trie.preprocess(path, self.dataset)
        best_score = self.experiment_handler.next_experiment(path, df)
        self.cache[tuple(path)] = best_score
        return best_score


    
    def start(self, dataset, n_trials = 50):
        self.dataset = dataset.copy(deep=True)
        self.study.optimize(self.objective, n_trials=n_trials)
        print('Best Path', self.study.best_params)
        print('Best value', self.study.best_value)



