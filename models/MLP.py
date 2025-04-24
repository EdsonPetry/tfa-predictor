import torch
import torch.nn as nn
from base.BaseModule import BaseModule
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class MLP(BaseModule):
    """Multi-Layer Perceptron model for TF activity prediction.
    
    This model maps gene expression data to transcription factor activity.
    It consists of fully connected layers with configurable activation,
    optional batch normalization, and dropout for regularization.
    """
    def __init__(self, input_size, output_size, hidden_sizes=[1024, 512, 256], 
                 lr=0.0005, weight_decay=1e-4, dropout_rate=0.3, 
                 activation='relu', optimizer='adam', 
                 learning_rate_schedule=None, batch_norm=True):
        """Initialize the MLP model.
        
        Args:
            input_size: Number of input features (gene expression dimensions)
            output_size: Number of output features (TF activity dimensions)
            hidden_sizes: List of hidden layer sizes
            lr: Learning rate for optimizer
            weight_decay: L2 regularization parameter
            dropout_rate: Dropout probability for regularization
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'selu', 'tanh')
            optimizer: Optimizer type ('adam', 'adamw', 'sgd_momentum', 'rmsprop')
            learning_rate_schedule: LR scheduler (None, 'step', 'cosine', 'exponential')
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.save_hyperparameters()

        # Set activation function based on configuration
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'selu':
            act_fn = nn.SELU()
        else:  # Default to tanh for backward compatibility
            act_fn = nn.Tanh()

        # Create the network architecture with fully connected layers
        # First layer: input to first hidden layer
        layers = [nn.Linear(input_size, hidden_sizes[0]), act_fn]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Hidden layers: connect each hidden layer to the next one
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(act_fn)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer: final hidden layer to output
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Build the entire network as a sequential model
        self.net = nn.Sequential(*layers)
        
        # Store the optimizer type and lr schedule for configure_optimizers
        self.optimizer_type = optimizer
        self.lr_schedule = learning_rate_schedule

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
        """Configure optimizer and optional learning rate scheduler.
        
        Returns:
            Configured optimizer or (optimizer, scheduler) tuple
        """
        # Select optimizer based on configuration
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(self.parameters(), self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd_momentum':
            optimizer = optim.SGD(self.parameters(), self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), self.lr, weight_decay=self.weight_decay)
        else:
            # Default to Adam for backward compatibility
            optimizer = optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
            
        # Return optimizer only if no learning rate schedule specified
        if self.lr_schedule is None:
            return optimizer
            
        # Configure learning rate scheduler if specified
        if self.lr_schedule == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif self.lr_schedule == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        elif self.lr_schedule == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            return optimizer  # No scheduler if unrecognized type
            
        return [optimizer], [scheduler]
