import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from base.BaseDataModule import BaseDataModule

class Data(BaseDataModule):
    def __init__(self, data_dir="/home/edson/Desktop/yang_lab/tfa-predictor/data", batch_size=32):
        super().__init__()
        self.save_hyperparameters()

        xprs_path = f"{data_dir}/gene-xprs/processed/xprs-data.csv"
        tfa_path = f"{data_dir}/tfa/processed/tfa-labels.csv"

        print(f"xprs_path: {xprs_path}")
        print(f"tfa_path: {tfa_path}")

        xprs_df = pd.read_csv(xprs_path)
        tfa_df = pd.read_csv(tfa_path)

        X = xprs_df.iloc[:, 1:].values if xprs_df.columns[0].lower() in ['unnamed: 0', 'index', 'sample', 'sample_id'] else xprs_df.values.T
        y = tfa_df.values.T

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        self.train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                           torch.tensor(y_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                         torch.tensor(y_val, dtype=torch.float32))

    def get_dataloader(self, train=True):
        dataset = self.train_dataset if train else self.val_dataset
        return DataLoader(dataset, batch_size=self.batch_size)
