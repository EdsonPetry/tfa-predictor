import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from base.BaseDataModule import BaseDataModule

class Data(BaseDataModule):
    """Data module for loading and preprocessing gene expression and TF activity data.
    
    This class handles:
    1. Loading gene expression and transcription factor activity data
    2. Preprocessing and scaling the data
    3. Splitting into train/validation sets
    4. Creating PyTorch DataLoaders
    """
    def __init__(self, data_dir="/home/elp95/tfa-predictor/data", batch_size=32):
        """Initialize the data module.
        
        Args:
            data_dir: Directory containing gene expression and TF activity data
            batch_size: Batch size for DataLoader
        """
        super().__init__()
        self.save_hyperparameters()

        # Define file paths for gene expression and TF activity data
        xprs_path = f"{data_dir}/gene-xprs/processed/xprs-data.csv"
        tfa_path = f"{data_dir}/tfa/processed/tfa-labels.csv"

        print(f"xprs_path: {xprs_path}")
        print(f"tfa_path: {tfa_path}")

        # Load data from CSV files
        xprs_df = pd.read_csv(xprs_path)  # Gene expression data
        tfa_df = pd.read_csv(tfa_path)    # TF activity data

        # Extract feature matrices
        # For gene expression, skip the first column if it contains sample IDs
        X = xprs_df.iloc[:, 1:].values if xprs_df.columns[0].lower() in ['unnamed: 0', 'index', 'sample', 'sample_id'] else xprs_df.values.T
        y = tfa_df.values.T  # Transpose TF activity data to match samples

        # Standardize data (mean=0, std=1) for better model convergence
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)  # Scale gene expression data
        y_scaled = scaler_y.fit_transform(y)  # Scale TF activity data

        # Split data into training and validation sets (80% train, 20% validation)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        # Create PyTorch datasets
        self.train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                           torch.tensor(y_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                         torch.tensor(y_val, dtype=torch.float32))

    def get_dataloader(self, train=True):
        """Get data loader for either training or validation data.
        
        Args:
            train: If True, return training data loader, else validation
            
        Returns:
            DataLoader: PyTorch DataLoader for the requested dataset
        """
        dataset = self.train_dataset if train else self.val_dataset
        return DataLoader(dataset, batch_size=self.batch_size)
