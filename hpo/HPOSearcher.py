from utils.HyperParameters import HyperParameters

class HPOSearcher(HyperParameters):
    def sample_configuration() -> dict:
        raise NotImplementedError
    
    def update(self, config: dict, error: float, additional_info=None):
        pass