import HPOSearcher

class RandomSearcher(HPOSearcher):
    def __init(self, config_space: dict, inital_config=None):
        self.save_hyperparameters()

    def sample_configuration(self) -> dict:
        if self.inital_config is not None:
            result = self.initial_config
            self.inital_config = None
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }

        return result