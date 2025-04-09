import torch
import torch.nn as nn
from utils.HyperParameters import HyperParameters

class BaseModule(nn.Module, HyperParameters):
    """The base class of models."""
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        raise NotImplementedError