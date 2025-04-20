import torch.nn as nn
from base.BaseModule import BaseModule
import torch.optim as optim

class MLP(BaseModule):
    """Multi-Layer Perceptron model for TF activity prediction.
    
    This model maps gene expression data to transcription factor activity.
    It consists of fully connected layers with tanh activation,
    batch normalization, and dropout for regularization.
    """
    def __init__(self, input_size, output_size, hidden_sizes=[1024, 512, 256], lr=0.0005, weight_decay=1e-4):
        """Initialize the MLP model.
        
        Args:
            input_size: Number of input features (gene expression dimensions)
            output_size: Number of output features (TF activity dimensions)
            hidden_sizes: List of hidden layer sizes in descending order
            lr: Learning rate for Adam optimizer
            weight_decay: L2 regularization parameter
        """
        super().__init__()
        self.save_hyperparameters()

        # Create the network architecture with fully connected layers
        # First layer: input to first hidden layer
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.Tanh(),
                  nn.BatchNorm1d(hidden_sizes[0]), nn.Dropout(0.3)]

        # Hidden layers: connect each hidden layer to the next one
        for i in range(len(hidden_sizes) - 1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.Tanh(),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.Dropout(0.3)
            ])

        # Output layer: final hidden layer to output
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        # Build the entire network as a sequential model
        self.net = nn.Sequential(*layers)

    def loss(self, y_hat, y):
        """Calculate the mean squared error loss.
        
        Args:
            y_hat: Model predictions
            y: Ground truth values
            
        Returns:
            MSE loss between predictions and ground truth
        """
        return nn.MSELoss()(y_hat, y)

    def configure_optimizers(self):
        """Configure the Adam optimizer with learning rate and weight decay.
        
        Returns:
            Configured Adam optimizer instance
        """
        return optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
