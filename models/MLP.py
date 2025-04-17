import torch.nn as nn
from base.BaseModule import BaseModule
import torch.optim as optim

class MLP(BaseModule):
    def __init__(self, input_size, output_size, hidden_sizes=[1024, 512, 256], lr=0.0005, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.Tanh(),
                  nn.BatchNorm1d(hidden_sizes[0]), nn.Dropout(0.3)]

        for i in range(len(hidden_sizes) - 1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.Tanh(),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.Dropout(0.3)
            ])

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.net = nn.Sequential(*layers)

    def loss(self, y_hat, y):
        return nn.MSELoss()(y_hat, y)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
