import torch
import torch.nn as nn
from base.HyperParameters import HyperParameters

class Module(nn.Module, HyperParameters):
    """The base class of models."""
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches
        

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        raise NotImplementedError