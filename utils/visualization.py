import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import r2_score
import pandas as pd

class LearningVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
    
    def update(self, epoch, train_loss, val_loss):
        """Update history with new loss values"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    
    def plot_learning_curve(self, save_path=None):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Training Loss', marker='o')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss', marker='o')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_prediction_vs_true(self, model, dataloader, num_tfs=5, save_path=None):
        """Plot predictions vs true values for a subset of TFs"""
        model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch
                preds = model(X)
                all_preds.append(preds)
                all_true.append(y)
        
        # Concatenate batches
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        all_true = torch.cat(all_true, dim=0).cpu().numpy()
        
        # Calculate R² for each TF
        r2_scores = [r2_score(all_true[:, i], all_preds[:, i]) for i in range(all_true.shape[1])]
        
        # Get indices of top TFs by R²
        top_indices = np.argsort(r2_scores)[-num_tfs:]
        
        # Plot for top TFs
        fig, axes = plt.subplots(1, num_tfs, figsize=(num_tfs*4, 4))
        if num_tfs == 1:
            axes = [axes]
            
        for i, idx in enumerate(top_indices):
            ax = axes[i]
            ax.scatter(all_true[:, idx], all_preds[:, idx], alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(all_true[:, idx].min(), all_preds[:, idx].min())
            max_val = max(all_true[:, idx].max(), all_preds[:, idx].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_title(f'TF #{idx}, R² = {r2_scores[idx]:.3f}')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        return r2_scores
    
    def plot_r2_distribution(self, r2_scores, save_path=None):
        """Plot histogram of R² scores across all TFs"""
        plt.figure(figsize=(10, 6))
        sns.histplot(r2_scores, kde=True)
        plt.axvline(np.mean(r2_scores), color='r', linestyle='--', 
                   label=f'Mean R² = {np.mean(r2_scores):.3f}')
        plt.title('Distribution of R² Scores Across TFs')
        plt.xlabel('R² Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()