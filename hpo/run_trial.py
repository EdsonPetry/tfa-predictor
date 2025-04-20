#!/usr/bin/env python
"""
Script to run a single hyperparameter optimization trial on the Amarel cluster.
This script is executed by SLURM for each trial.
"""

import os
import json
import argparse
import time
import torch
import numpy as np
from datetime import datetime

from models.MLP import MLP
from data_loader.Data import Data
from trainer.Trainer import Trainer


def run_trial(config, output_path):
    """Run a single trial with the given hyperparameter configuration.
    
    Args:
        config (dict): Hyperparameter configuration to evaluate
        output_path (str): Path to save the results
    
    Returns:
        float: Validation error (lower is better)
    """
    # Print configuration for debugging
    print(f"Running trial with config: {config}")
    start_time = time.time()
    
    # Ensure configs are the right type (SLURM job might convert them)
    processed_config = {}
    for key, value in config.items():
        if key in ['input_size', 'output_size', 'max_epochs', 'batch_size']:
            processed_config[key] = int(value)
        elif key in ['lr', 'weight_decay', 'dropout']:
            processed_config[key] = float(value)
        elif key == 'hidden_sizes':
            if isinstance(value, list):
                processed_config[key] = [int(x) for x in value]
            else:
                # Handle string representation from JSON
                processed_config[key] = [int(x) for x in json.loads(value.replace("'", '"'))]
        else:
            processed_config[key] = value
    
    # Create data loader
    data_config = {k: v for k, v in processed_config.items() 
                  if k in ['batch_size', 'data_dir']}
    # Make sure data_dir is set properly for Amarel
    if 'data_dir' not in data_config:
        data_config['data_dir'] = '/home/elp95/tfa-predictor/data'
    print(f"Using data directory: {data_config['data_dir']}")
    data = Data(**data_config)
    
    # Create model
    model_config = {k: v for k, v in processed_config.items() 
                   if k in ['input_size', 'output_size', 'hidden_sizes', 'lr', 'weight_decay']}
    model = MLP(**model_config)
    
    # Create trainer
    trainer_config = {k: v for k, v in processed_config.items() 
                     if k in ['max_epochs']}
    trainer_config.setdefault('max_epochs', 10)  # Default to 10 epochs if not specified
    trainer = Trainer(**trainer_config, visualize=False)  # Disable visualization on cluster
    
    # Train model
    trainer.fit(model, data)
    
    # Calculate validation error
    model.eval()
    with torch.no_grad():
        val_losses = []
        for batch in data.val_dataloader():
            loss = model.validation_step(batch)
            val_losses.append(loss.item())
        val_error = np.mean(val_losses)
    
    elapsed_time = time.time() - start_time
    
    # Save results
    results = {
        'error': float(val_error),
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().isoformat(),
        'additional_info': {
            'training_time': elapsed_time,
            'epochs': trainer_config['max_epochs'],
            'final_train_loss': float(trainer.visualizer.train_losses[-1]) 
                               if hasattr(trainer, 'visualizer') else None
        }
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results to output file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Trial completed in {elapsed_time:.2f}s with error: {val_error:.4f}")
    
    return val_error


def main():
    """Parse arguments and run the trial."""
    parser = argparse.ArgumentParser(description='Run a hyperparameter optimization trial')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to save results JSON')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Run the trial
    error = run_trial(config, args.output)
    
    return error


if __name__ == '__main__':
    main()