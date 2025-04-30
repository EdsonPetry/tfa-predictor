from hpo.base.HPOScheduler import HPOScheduler
from hpo.base.HPOSearcher import HPOSearcher

class BasicScheduler(HPOScheduler):
    """Basic scheduler for hyperparameter optimization.
    
    This scheduler simply forwards requests to a searcher.
    """
    
    def __init__(self, searcher: HPOSearcher):
        """Initialize the basic scheduler.
        
        Args:
            searcher (HPOSearcher): The searcher to use for sampling configurations
        """
        self.save_hyperparameters()
        self.searcher = searcher
    
    def suggest(self) -> dict:
        """Suggest a new configuration to evaluate.
        
        Returns:
            dict: A configuration to evaluate
        """
        return self.searcher.sample_configuration()
    
    def update(self, config: dict, error: float, info=None):
        """Update the scheduler with results from a trial.
        
        Args:
            config (dict): The configuration that was evaluated
            error (float): The validation error/score
            info (dict, optional): Additional information from the trial
        """
        self.searcher.update(config, error, info)