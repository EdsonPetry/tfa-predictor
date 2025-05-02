from utils.HyperParameters import HyperParameters

class HPOScheduler(HyperParameters):
    """Base abstract class for hyperparameter optimization schedulers.
    
    This class defines the interface for schedulers that manage
    the execution of hyperparameter optimization trials.
    """
    
    def suggest(self) -> dict:
        """Suggest a new configuration to evaluate.
        
        Returns:
            dict: A configuration to evaluate
        """
        raise NotImplementedError

    def update(self, config: dict, error: float, info=None):
        """Update the scheduler with results from a trial.
        
        Args:
            config (dict): The configuration that was evaluated
            error (float): The validation error/score
            info (dict, optional): Additional information from the trial
        """
        raise NotImplementedError