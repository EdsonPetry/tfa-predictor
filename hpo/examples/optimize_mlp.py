#!/usr/bin/env python
"""
Example script demonstrating how to use the asynchronous random search
implementation to optimize an MLP model locally before running on Amarel.
"""

import os
import sys
import json
import torch
import numpy as np
from scipy.stats import uniform, loguniform
import matplotlib.pyplot as plt

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.MLP import MLP
from data_loader.Data import Data
from trainer.Trainer import Trainer
from hpo.RandomSearcher import RandomSearcher
from hpo.BasicScheduler import BasicScheduler
from hpo.HPOTuner import HPOTuner


def objective(input_size, output_size, hidden_sizes, lr, weight_decay, batch_size=32, max_epochs=10):
    """Objective function to minimize.
    
    Args:
        input_size (int): Input dimension
        output_size (int): Output dimension
        hidden_sizes (list): List of hidden layer sizes
        lr (float): Learning rate
        weight_decay (float): Weight decay
        batch_size (int): Batch size
        max_epochs (int): Number of training epochs
        
    Returns:
        float: Validation error (lower is better)
    """
    # Create data loader
    data = Data(batch_size=batch_size)
    
    # Create model
    model = MLP(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Create trainer
    trainer = Trainer(max_epochs=max_epochs, visualize=False)
    
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
    
    return val_error


def main():
    """Run a simple hyperparameter optimization example."""
    # Define the search space
    config_space = {
        'input_size': 3883,  # Fixed
        'output_size': 214,  # Fixed
        'hidden_sizes': [
            [1024, 512, 256],       # Default
            [2048, 1024, 512],      # Wider
            [1024, 256],            # Two-layer
        ],
        'lr': loguniform(1e-4, 1e-2),
        'weight_decay': loguniform(1e-5, 1e-3),
        'batch_size': [32, 64],
        'max_epochs': [5, 10],  # Keep low for example purposes
    }
    
    # Create searcher and scheduler
    searcher = RandomSearcher(config_space)
    scheduler = BasicScheduler(searcher)
    
    # Create HPO tuner
    tuner = HPOTuner(scheduler, objective)
    
    # Run optimization (with small number of trials for demonstration)
    n_trials = 3  # For demonstration only; use ~50+ in practice
    print(f"Running {n_trials} trials...")
    tuner.run(n_trials)
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best configuration: {tuner.incumbent}")
    print(f"Best validation error: {tuner.incumbent_error}")
    
    # Plot optimization trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(tuner.incumbent_trajectory)
    plt.xlabel('Trial')
    plt.ylabel('Best Validation Error')
    plt.title('Optimization Trajectory')
    plt.grid(True)
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/optimization_trajectory.png')
    plt.close()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/hpo_results.json', 'w') as f:
        json.dump({
            'best_config': tuner.incumbent,
            'best_error': tuner.incumbent_error,
            'trajectory': tuner.incumbent_trajectory,
            'records': tuner.records
        }, f, indent=2)
    
    print("\nResults saved to results/hpo_results.json")
    print("Optimization trajectory plot saved to figures/optimization_trajectory.png")


if __name__ == '__main__':
    main()