from abc import abstractmethod
class SuggestBase:
    def __init__(self):
        pass

    @abstractmethod
    def suggest(self, trial, name):
        return None

class SuggestNumeric(SuggestBase):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    @abstractmethod
    def suggest(self, trial, name):
        return super().suggest()

class SuggestFloat(SuggestNumeric):
    def __init__(self, lower, upper):
        super().__init__(lower, upper)

    def suggest(self, trial, name):
        return trial.suggest_float(name, self.lower, self.upper)

class SuggestLogUniform(SuggestNumeric):
    def __init__(self, lower, upper):
        super().__init__(lower, upper)
    
    def suggest(self, trial, name):
        return trial.suggest_loguniform(name, self.lower, self.upper)