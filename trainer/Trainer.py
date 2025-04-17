import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models.MLP import MLP
from data_loader.Data import Data
from base.BaseTrainer import BaseTrainer
from utils.visualization import LearningVisualizer

class Trainer(BaseTrainer):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0, 
                 visualize=True, save_plots=True, plots_dir='plots'):
        super().__init__(max_epochs, num_gpus, gradient_clip_val)
        self.save_hyperparameters(ignore=['visualize', 'save_plots', 'plots_dir'])
        
        self.visualize = visualize
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        
        if self.visualize:
            self.visualizer = LearningVisualizer()
            
            if self.save_plots and not os.path.exists(self.plots_dir):
                os.makedirs(self.plots_dir)
    
    def fit_epoch(self):
        self.model.train()
        train_losses = []
        for batch in self.train_dataloader:
            self.optim.zero_grad()
            loss = self.model.training_step(batch)
            loss.backward()
            self.optim.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)

        # Evaluation
        if self.val_dataloader is not None:
            self.model.eval()
            with torch.no_grad():
                val_losses = []
                for batch in self.val_dataloader:
                    loss = self.model.validation_step(batch)
                    val_losses.append(loss.item())
                avg_val_loss = np.mean(val_losses)
                print(f"Epoch {self.epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_val_loss:.4f}")
                
                if self.visualize:
                    self.visualizer.update(self.epoch+1, avg_train_loss, avg_val_loss)
    
    def fit(self, model, data):
        super().fit(model, data)
        
        # After training, create visualizations
        if self.visualize:
            # Plot learning curves
            save_path = f"{self.plots_dir}/learning_curve.png" if self.save_plots else None
            self.visualizer.plot_learning_curve(save_path=save_path)
            
            # Plot predictions vs true values
            save_path = f"{self.plots_dir}/predictions_vs_true.png" if self.save_plots else None
            r2_scores = self.visualizer.plot_prediction_vs_true(
                self.model, self.val_dataloader, num_tfs=5, save_path=save_path
            )
            
            # Plot R² distribution
            save_path = f"{self.plots_dir}/r2_distribution.png" if self.save_plots else None
            self.visualizer.plot_r2_distribution(r2_scores, save_path=save_path)
            
            return r2_scores

if __name__ == "__main__":
    data = Data()
    model = MLP(input_size=3883, output_size=214)
    trainer = Trainer(max_epochs=100, visualize=True, save_plots=True)
    trainer.fit(model, data)
