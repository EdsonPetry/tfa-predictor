#!/usr/bin/env python
"""
Script to analyze results from a hyperparameter optimization campaign.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_results(results_dir):
    """Load results from an HPO campaign.
    
    Args:
        results_dir (str): Directory containing results
        
    Returns:
        dict: Campaign results
        list: Individual trial results
    """
    # Load campaign results
    with open(os.path.join(results_dir, 'final_results.json'), 'r') as f:
        campaign_results = json.load(f)
    
    # Load individual trial results
    trial_results = []
    results_path = os.path.join(results_dir, 'results')
    if os.path.exists(results_path):
        for file in os.listdir(results_path):
            if file.startswith('trial_') and file.endswith('.json'):
                with open(os.path.join(results_path, file), 'r') as f:
                    trial_results.append(json.load(f))
    
    return campaign_results, trial_results


def create_dataframe(trial_results):
    """Create a DataFrame from trial results.
    
    Args:
        trial_results (list): List of trial result dictionaries
        
    Returns:
        pandas.DataFrame: DataFrame with trial data
    """
    # Extract relevant information
    data = []
    for trial in trial_results:
        row = {
            'trial_id': trial.get('trial_id'),
            'error': trial.get('error'),
            'submit_time': trial.get('submit_time'),
            'complete_time': trial.get('complete_time'),
        }
        
        # Add config parameters
        config = trial.get('config', {})
        for key, value in config.items():
            row[f'config_{key}'] = value
        
        # Add additional info
        additional_info = trial.get('additional_info', {})
        if additional_info:
            for key, value in additional_info.items():
                row[f'info_{key}'] = value
        
        data.append(row)
    
    return pd.DataFrame(data)


def plot_error_distribution(df, best_error=None, save_path=None):
    """Plot the distribution of validation errors.
    
    Args:
        df (pandas.DataFrame): DataFrame with trial data
        best_error (float, optional): Best validation error
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['error'], kde=True)
    
    if best_error is not None:
        plt.axvline(best_error, color='r', linestyle='--', 
                   label=f'Best error: {best_error:.4f}')
        plt.legend()
    
    plt.title('Distribution of Validation Errors')
    plt.xlabel('Validation Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_parallel_coordinates(df, save_path=None):
    """Create a parallel coordinates plot for hyperparameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with trial data
        save_path (str, optional): Path to save the plot
    """
    # Select columns that start with 'config_' but exclude list/dict type columns
    config_cols = []
    for col in df.columns:
        if col.startswith('config_'):
            # Check if column contains lists or dicts
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                continue
            config_cols.append(col)
    
    if len(config_cols) > 0:
        # Create a new DataFrame with normalized values
        plot_df = df[config_cols + ['error']].copy()
        
        # Normalize columns for better visualization
        for col in config_cols:
            if plot_df[col].dtype in [np.float64, np.int64]:
                plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min() + 1e-8)
        
        # Normalize error (inverted, so lower is better)
        error_max = plot_df['error'].max()
        error_min = plot_df['error'].min()
        plot_df['error_normalized'] = 1 - (plot_df['error'] - error_min) / (error_max - error_min + 1e-8)
        
        # Create parallel coordinates plot
        plt.figure(figsize=(12, 6))
        pd.plotting.parallel_coordinates(
            plot_df, 'error_normalized', 
            colormap=plt.cm.viridis
        )
        plt.title('Parallel Coordinates Plot of Hyperparameters')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def plot_error_vs_params(df, save_path=None):
    """Plot validation error vs. hyperparameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with trial data
        save_path (str, optional): Path to save the plot
    """
    # Select numeric config columns, excluding lists/dicts
    config_cols = []
    for col in df.columns:
        if col.startswith('config_'):
            # Skip columns with list or dict values
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                continue
            # Only include numeric columns
            if df[col].dtype in [np.float64, np.int64]:
                config_cols.append(col)
    
    if len(config_cols) > 0:
        # Create subplots
        n_cols = min(2, len(config_cols))
        n_rows = (len(config_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for i, col in enumerate(config_cols):
            if i < len(axes):
                ax = axes[i]
                param_name = col.replace('config_', '')
                
                scatter = ax.scatter(df[col], df['error'], 
                                    c=df['error'], cmap='viridis', 
                                    alpha=0.7)
                
                ax.set_xlabel(param_name)
                ax.set_ylabel('Validation Error')
                ax.set_title(f'Error vs {param_name}')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                if i == 0:
                    plt.colorbar(scatter, ax=ax, label='Error')
                
                # Use log scale for some parameters
                if param_name in ['lr', 'weight_decay', 'learning_rate']:
                    ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def plot_optimization_trajectory(trials, save_path=None):
    """Plot the optimization trajectory.
    
    Args:
        trials (list): List of trial results
        save_path (str, optional): Path to save the plot
    """
    # Sort trials by completion time if available
    if all('complete_time' in trial for trial in trials):
        sorted_trials = sorted(trials, key=lambda x: x['complete_time'])
    else:
        # Otherwise, use trial_id if available
        sorted_trials = sorted(trials, key=lambda x: x.get('trial_id', 0))
    
    # Extract errors
    errors = [trial['error'] for trial in sorted_trials]
    
    # Calculate best seen so far
    best_so_far = np.minimum.accumulate(errors)
    
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'o-', alpha=0.5, label='Trial errors')
    plt.plot(best_so_far, 'r-', linewidth=2, label='Best so far')
    
    plt.xlabel('Trial')
    plt.ylabel('Validation Error')
    plt.title('Optimization Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    """Parse arguments and analyze results."""
    parser = argparse.ArgumentParser(description='Analyze HPO results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing HPO results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis plots')
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(args.results_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    campaign_results, trial_results = load_results(args.results_dir)
    df = create_dataframe(trial_results)
    
    print(f"Loaded {len(trial_results)} trials from {args.results_dir}")
    print(f"Best validation error: {campaign_results['best_error']}")
    print(f"Best configuration: {campaign_results['best_config']}")
    
    # Generate plots
    plot_error_distribution(
        df, 
        best_error=campaign_results['best_error'],
        save_path=os.path.join(output_dir, 'error_distribution.png')
    )
    
    plot_parallel_coordinates(
        df,
        save_path=os.path.join(output_dir, 'parallel_coordinates.png')
    )
    
    plot_error_vs_params(
        df,
        save_path=os.path.join(output_dir, 'error_vs_params.png')
    )
    
    plot_optimization_trajectory(
        trial_results,
        save_path=os.path.join(output_dir, 'optimization_trajectory.png')
    )
    
    print(f"Analysis complete. Plots saved to {output_dir}")


if __name__ == '__main__':
    main()