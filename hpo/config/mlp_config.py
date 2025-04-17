"""
Configuration for hyperparameter optimization of MLP models on Amarel.
"""

import os
import numpy as np
from scipy.stats import uniform, loguniform

# Define the hyperparameter search space
def get_config_space():
    """Define the hyperparameter search space for MLP model."""
    return {
        # Model architecture
        'input_size': 3883,  # Fixed input size based on data
        'output_size': 214,  # Fixed output size based on data
        'hidden_sizes': [
            [1024, 512, 256],       # Default architecture
            [2048, 1024, 512],      # Wider architecture
            [2048, 1024, 512, 256], # Deeper architecture
            [512, 256, 128],        # Narrower architecture
            [1024, 256],            # Two-layer architecture
        ],
        
        # Optimization parameters
        'lr': loguniform(1e-4, 1e-2),          # Learning rate
        'weight_decay': loguniform(1e-5, 1e-3), # L2 regularization
        'batch_size': [32, 64, 128],            # Batch size options
        'max_epochs': [50, 100, 200],           # Training epochs options
    }

# SLURM configuration for Amarel
def get_slurm_config():
    """Define the SLURM configuration for Amarel HPO jobs."""
    return {
        'partition': 'main',          # Use main partition
        'time': '4:00:00',            # Time limit of 4 hours per job
        'mem': '16G',                 # Memory request
        'cpus_per_task': 4,           # CPU cores per task
        'gpus': 1,                    # Request 1 GPU per job
        'env_setup': 'source activate tfa-predictor', # Environment setup command
        
        # Additional SLURM options
        'mail-user': 'your.email@rutgers.edu',
        'mail-type': 'END,FAIL',
    }

# HPO campaign configuration
def get_hpo_config():
    """Define the configuration for the HPO campaign."""
    return {
        'n_trials': 100,                   # Total number of trials to run
        'max_concurrent_jobs': 8,          # Maximum concurrent jobs
        'output_dir': os.path.join('hpo_results', 'mlp'),  # Results directory
        'checkpoint_interval': 5,          # Save checkpoint every 5 trials
        'seed': 42,                        # Random seed for reproducibility
    }