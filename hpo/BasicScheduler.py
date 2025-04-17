from hpo.HPOScheduler import HPOScheduler
from hpo.HPOSearcher import HPOSearcher

class BasicScheduler(HPOScheduler):
    def __init__(self, searcher: HPOSearcher):
        self.save_hyperparameters()

    def suggest(self) -> dict:
        return self.searcher.sample_configuration()
    
    def update(self, config: dict, error: float, info=None):
        self.searcher.update(config, error, additional_info=info)