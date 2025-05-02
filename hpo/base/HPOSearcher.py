from utils.HyperParameters import HyperParameters

class HPOSearcher(HyperParameters):
    """Base abstract class for hyperparameter search strategies.
    
    This class defines the interface for hyperparameter searchers
    that sample configurations from a search space.
    """
    
    def sample_configuration(self) -> dict:
        """Sample a new configuration from the search space.
        
        Returns:
            dict: A sampled configuration
        """
        raise NotImplementedError
    
    def update(self, config: dict, error: float, additional_info=None):
        """Update the searcher with results from a completed trial.
        
        Args:
            config (dict): The configuration that was evaluated
            error (float): The validation error/score
            additional_info (dict, optional): Additional information from the trial
        """
        pass