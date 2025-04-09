import torch
import torch.nn as nn
import numpy as np
from utils.Trainer import Trainer

class TBTrainer(Trainer):
    """Custom trainer for TB models with k-fold cross-validation support."""
    
    def __init__(self, max_epochs=10, num_gpus=0, gradient_clip_val=0):
        super().__init__(max_epochs=max_epochs, num_gpus=num_gpus, 
                         gradient_clip_val=gradient_clip_val)
        self.train_losses = []
        self.val_losses = []
        
    def fit_epoch(self):
        """Train the model for one epoch."""
        # Training phase
        self.model.train()
        epoch_train_loss = 0.0
        for batch_idx, batch in enumerate(self.train_dataloader):
            self.optim.zero_grad()
            loss = self.model.training_step(batch)
            loss.backward()
            
            if self.gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val)
                
            self.optim.step()
            epoch_train_loss += loss.item()
            self.train_batch_idx += 1
        
        # Average training loss
        epoch_train_loss /= len(self.train_dataloader)
        self.train_losses.append(epoch_train_loss)
        
        # Validation phase
        if self.val_dataloader is None:
            return
        
        self.model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                # Since validation_step may not return loss in base Module
                # We need to compute it directly
                y_hat = self.model(*batch[:-1])
                loss = self.model.loss(y_hat, batch[-1])
                epoch_val_loss += loss.item()
                self.val_batch_idx += 1
        
        # Average validation loss
        epoch_val_loss /= len(self.val_dataloader)
        self.val_losses.append(epoch_val_loss)
        
        print(f"Epoch {self.epoch+1}/{self.max_epochs} - "
              f"Train loss: {epoch_train_loss:.4f}, Val loss: {epoch_val_loss:.4f}")
    
    def fit_with_kfold(self, model_class, model_args, data, k=5, random_state=42):
        """Train and evaluate model using k-fold cross-validation."""
        kfold_indices = data.get_kfold_indices(k=k, random_state=random_state)
        fold_val_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold_indices):
            print(f"\nFold {fold+1}/{k}")
            
            # Create model instance for this fold
            model = model_class(**model_args)
            
            # Prepare data loaders for this fold
            self.train_dataloader = data.get_dataloader(train=True, indices=train_idx)
            self.val_dataloader = data.get_dataloader(train=False, indices=val_idx)
            self.num_train_batches = len(self.train_dataloader)
            self.num_val_batches = len(self.val_dataloader)
            
            # Prepare model
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            
            # Train
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            self.train_losses = []
            self.val_losses = []
            
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()
            
            # Get final validation loss for this fold
            fold_val_loss = self.val_losses[-1]
            fold_val_losses.append(fold_val_loss)
            
            print(f"Fold {fold+1} final validation loss: {fold_val_loss:.4f}")
        
        # Calculate and return average validation loss across all folds
        avg_val_loss = np.mean(fold_val_losses)
        std_val_loss = np.std(fold_val_losses)
        print(f"\nAverage validation loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        
        return avg_val_loss, fold_val_losses