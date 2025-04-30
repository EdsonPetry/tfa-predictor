from hpo.base.HPOSearcher import HPOSearcher

class RandomSearcher(HPOSearcher):
    """Random search implementation for hyperparameter optimization.
    
    This class implements random sampling from a configuration space.
    """
    
    def __init__(self, config_space: dict, initial_config=None):
        """Initialize the random searcher.
        
        Args:
            config_space (dict): Dictionary mapping parameter names to their search domains
            initial_config (dict, optional): Initial configuration to evaluate
        """
        self.save_hyperparameters()
        self.config_space = config_space
        self.initial_config = initial_config

    def sample_configuration(self) -> dict:
        """Sample a new configuration using random search.
        
        Returns:
            dict: A sampled configuration
        """
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }

        return result