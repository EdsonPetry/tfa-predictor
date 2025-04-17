from utils.HyperParameters import HyperParameters

class HPOScheduler(HyperParameters):
    def suggest(self) -> dict:
        raise NotImplementedError

    def update(self, config: dict, error: float, info=None):
        raise NotImplementedError
