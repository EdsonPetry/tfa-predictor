import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models.MLP import MLP
from data_loader.Data import Data
from base.BaseTrainer import BaseTrainer
from utils.visualization import LearningVisualizer

class Trainer(BaseTrainer):
    """Trainer class for managing the model training and evaluation process.
    
    This class handles:
    1. Training loop implementation
    2. Optimization and gradient clipping
    3. Loss tracking and visualization
    4. Model evaluation and performance metrics
    """
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0, 
                 visualize=True, save_plots=True, plots_dir='plots'):
        """Initialize the trainer.
        
        Args:
            max_epochs: Maximum number of training epochs
            num_gpus: Number of GPUs to use (0 for CPU)
            gradient_clip_val: Value for gradient clipping (0 for no clipping)
            visualize: Whether to create visualizations during training
            save_plots: Whether to save plots to disk
            plots_dir: Directory where plots will be saved
        """
        super().__init__(max_epochs, num_gpus, gradient_clip_val)
        self.save_hyperparameters(ignore=['visualize', 'save_plots', 'plots_dir'])
        
        # Visualization settings
        self.visualize = visualize
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        
        # Initialize visualizer if visualization is enabled
        if self.visualize:
            self.visualizer = LearningVisualizer()
            
            # Create plots directory if it doesn't exist
            if self.save_plots and not os.path.exists(self.plots_dir):
                os.makedirs(self.plots_dir)
    
    def fit_epoch(self):
        """Run a single training epoch with validation.
        
        This method:
        1. Trains the model on all training batches
        2. Evaluates on validation data if available
        3. Updates visualizations with metrics
        """
        # Training phase
        self.model.train()
        train_losses = []
        for batch in self.train_dataloader:
            # Zero gradients before forward pass
            self.optim.zero_grad()
            # Forward pass and loss calculation
            loss = self.model.training_step(batch)
            # Backward pass
            loss.backward()
            # Gradient clipping (if configured)
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            # Update parameters
            self.optim.step()
            # Record loss
            train_losses.append(loss.item())
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_losses)

        # Validation phase
        if self.val_dataloader is not None:
            self.model.eval()
            with torch.no_grad():  # No gradients needed for validation
                val_losses = []
                for batch in self.val_dataloader:
                    loss = self.model.validation_step(batch)
                    val_losses.append(loss.item())
                avg_val_loss = np.mean(val_losses)
                print(f"Epoch {self.epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_val_loss:.4f}")
                
                # Update visualizations if enabled
                if self.visualize:
                    self.visualizer.update(self.epoch+1, avg_train_loss, avg_val_loss)
    
    def fit(self, model, data):
        """Train the model on the provided data.
        
        Args:
            model: The neural network model to train
            data: Data module containing training and validation data
            
        Returns:
            r2_scores: R-squared scores if visualization is enabled
        """
        # Call parent class fit method to run training loop
        super().fit(model, data)
        
        # After training, create visualizations
        if self.visualize:
            # Plot learning curves (train and validation loss over epochs)
            save_path = f"{self.plots_dir}/learning_curve.png" if self.save_plots else None
            self.visualizer.plot_learning_curve(save_path=save_path)
            
            # Plot predictions vs true values for selected TFs
            save_path = f"{self.plots_dir}/predictions_vs_true.png" if self.save_plots else None
            r2_scores = self.visualizer.plot_prediction_vs_true(
                self.model, self.val_dataloader, num_tfs=5, save_path=save_path
            )
            
            # Plot R² distribution across all TFs
            save_path = f"{self.plots_dir}/r2_distribution.png" if self.save_plots else None
            self.visualizer.plot_r2_distribution(r2_scores, save_path=save_path)
            
            return r2_scores

if __name__ == "__main__":
    # For Amarel, use the correct data path
    data_dir = '/home/elp95/tfa-predictor/data'
    print(f"Using data directory: {data_dir}")
    data = Data(data_dir=data_dir)
    model = MLP(input_size=3883, output_size=214)
    trainer = Trainer(max_epochs=100, visualize=True, save_plots=True)
    trainer.fit(model, data)
