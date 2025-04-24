"""
Configuration for hyperparameter optimization of MLP models on Amarel.
"""

import os
import numpy as np
from scipy.stats import uniform, loguniform, randint

# Define the hyperparameter search space
def get_config_space():
    """Define the hyperparameter search space for MLP model."""
    return {
        # Model architecture
        'input_size': 3883,  # Fixed input size based on data
        'output_size': 214,  # Fixed output size based on data
        'hidden_sizes': [
            # Previous architectures
            [1024, 512, 256],       
            [2048, 1024, 512],      
            [2048, 1024, 512, 256], 
            [512, 256, 128],        
            [1024, 256],            
            # Additional architectures
            [4096, 2048, 1024],     # Extra wide
            [4096, 2048, 1024, 512], # Extra wide and deep
            [3072, 1536, 768, 384], # Alternative wide and deep
            [2048, 1024, 512, 256, 128], # Very deep
            [1024, 1024, 1024],     # Constant width
            [2048, 1024, 1024, 512], # Wide with plateau
            [1024, 1024, 512, 256], # Plateau with taper
            [768, 512, 256],        # Medium-sized
        ],
        
        # Optimization parameters
        'lr': loguniform(5e-5, 2e-2),          # Extended learning rate range
        'weight_decay': loguniform(1e-6, 1e-3), # Extended weight decay range
        'batch_size': [16, 32, 64, 128, 256],   # More batch size options
        'max_epochs': [50, 100, 150, 200, 300], # More epoch options
        
        # Additional parameters
        'dropout_rate': uniform(0, 0.5),        # Dropout regularization
        'activation': ['relu', 'leaky_relu', 'elu', 'selu'], # Activation functions
        'optimizer': ['adam', 'adamw', 'sgd_momentum', 'rmsprop'], # Optimizer types
        'learning_rate_schedule': [None, 'step', 'cosine', 'exponential'], # LR schedulers
        'batch_norm': [True, False],            # Whether to use batch normalization
    }

# SLURM configuration for Amarel
def get_slurm_config():
    """Define the SLURM configuration for Amarel HPO jobs."""
    return {
        'partition': 'main',          # Use main partition
        'time': '4:00:00',            # Time limit of 4 hours per job
        'mem': '16G',                 # Memory request
        'cpus_per_task': 4,           # CPU cores per task
        # 'gpus': 1,                  # GPU request removed - not needed for MLP model
        
        # Additional SLURM options
        'mail-user': 'edson.petry@rutgers.edu',
        'mail-type': 'END,FAIL',
    }

# HPO campaign configuration
def get_hpo_config():
    """Define the configuration for the HPO campaign."""
    return {
        'n_trials': 100,                   # Increased number of trials for larger space
        'max_concurrent_jobs': 16,         # Increased concurrency
        'output_dir': os.path.join('/home/elp95/tfa-predictor/hpo_results', 'mlp_extended'),  # New results directory
        'checkpoint_interval': 5,          # Save checkpoint every 5 trials
        'seed': 42,                        # Random seed for reproducibility
    }