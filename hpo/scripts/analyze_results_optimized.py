#!/usr/bin/env python
"""
Script to analyze results from a hyperparameter optimization campaign.
Optimized version for handling large numbers of trials efficiently.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm
import time

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analyze_results')


def load_results(results_dir, limit=None, sort_by_error=True, verbose=False):
    """Load results from an HPO campaign with improved efficiency.
    
    Args:
        results_dir (str): Directory containing results
        limit (int, optional): Maximum number of trials to load
        sort_by_error (bool): Whether to sort trials by error and take the best ones
        verbose (bool): Whether to show detailed logging
        
    Returns:
        dict: Campaign results
        list: Individual trial results
    """
    # Load campaign results
    try:
        with open(os.path.join(results_dir, 'final_results.json'), 'r') as f:
            campaign_results = json.load(f)
    except FileNotFoundError:
        logger.error(f"final_results.json not found in {results_dir}")
        return {}, []
    except json.JSONDecodeError:
        logger.error(f"Error parsing final_results.json")
        return {}, []
    
    # Load individual trial files in batches
    results_path = os.path.join(results_dir, 'results')
    if not os.path.exists(results_path):
        logger.error(f"Results directory not found: {results_path}")
        return campaign_results, []
    
    # Get all trial files
    trial_files = [f for f in os.listdir(results_path) 
                  if f.startswith('trial_') and f.endswith('.json')]
    
    if verbose:
        logger.info(f"Found {len(trial_files)} trial files")
    
    # If sorting by error, we need to extract the error from each file first
    if sort_by_error and limit is not None:
        trial_errors = []
        for file in trial_files:
            try:
                with open(os.path.join(results_path, file), 'r') as f:
                    data = json.load(f)
                    error = data.get('error')
                    if error is not None:
                        trial_errors.append((file, error))
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        # Sort by error and take the top 'limit' trials
        trial_errors.sort(key=lambda x: x[1])
        selected_files = [f for f, _ in trial_errors[:limit]]
    else:
        # Otherwise just take the first 'limit' files
        selected_files = trial_files[:limit] if limit else trial_files
    
    if verbose:
        logger.info(f"Loading {len(selected_files)} trial results")
        pbar = tqdm(total=len(selected_files))
    
    # Use parallel processing for faster loading
    trial_results = []
    
    # Process files in batches to avoid memory issues
    batch_size = 50  # Adjust based on available memory
    for i in range(0, len(selected_files), batch_size):
        batch_files = selected_files[i:i+batch_size]
        batch_results = []
        
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            futures = {
                executor.submit(load_trial_file, os.path.join(results_path, file)): file 
                for file in batch_files
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    batch_results.append(result)
                if verbose:
                    pbar.update(1)
        
        trial_results.extend(batch_results)
    
    if verbose:
        pbar.close()
        logger.info(f"Successfully loaded {len(trial_results)} trials")
    
    return campaign_results, trial_results


def load_trial_file(filepath):
    """Load a single trial file with error handling.
    
    Args:
        filepath (str): Path to the trial file
        
    Returns:
        dict: Trial data or None if there was an error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError):
        return None


def create_dataframe(trial_results, skip_complex_fields=True):
    """Create a DataFrame from trial results more efficiently.
    
    Args:
        trial_results (list): List of trial result dictionaries
        skip_complex_fields (bool): Whether to skip list/dict fields
        
    Returns:
        pandas.DataFrame: DataFrame with trial data
    """
    # Early exit for empty results
    if not trial_results:
        return pd.DataFrame()
    
    # Extract relevant information
    data = []
    for trial in trial_results:
        if not trial:
            continue
            
        row = {
            'trial_id': trial.get('trial_id'),
            'error': trial.get('error'),
            'submit_time': trial.get('submit_time'),
            'complete_time': trial.get('complete_time'),
        }
        
        # Add config parameters, skipping complex types if requested
        config = trial.get('config', {})
        for key, value in config.items():
            if skip_complex_fields and isinstance(value, (list, dict)):
                continue
            row[f'config_{key}'] = value
        
        # Add additional info, skipping complex types if requested
        additional_info = trial.get('additional_info', {})
        if additional_info:
            for key, value in additional_info.items():
                if skip_complex_fields and isinstance(value, (list, dict)):
                    continue
                row[f'info_{key}'] = value
        
        data.append(row)
    
    # Use more efficient DataFrame creation
    df = pd.DataFrame(data)
    
    # Convert columns to appropriate types where possible for better performance
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric if possible
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    
    return df


def plot_error_distribution(df, best_error=None, save_path=None, show_plot=True):
    """Plot the distribution of validation errors.
    
    Args:
        df (pandas.DataFrame): DataFrame with trial data
        best_error (float, optional): Best validation error
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Use more efficient plotting options
    sns.histplot(df['error'], kde=True, bins=min(30, len(df) // 3 + 5))
    
    if best_error is not None:
        plt.axvline(best_error, color='r', linestyle='--', 
                   label=f'Best error: {best_error:.4f}')
        plt.legend()
    
    plt.title('Distribution of Validation Errors')
    plt.xlabel('Validation Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100)
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_parallel_coordinates(df, save_path=None, show_plot=True, max_params=8):
    """Create a parallel coordinates plot for hyperparameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with trial data
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
        max_params (int): Maximum number of parameters to include
    """
    # Select columns that start with 'config_' but exclude list/dict type columns
    config_cols = []
    for col in df.columns:
        if col.startswith('config_'):
            # Check if column contains lists or dicts
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                continue
            config_cols.append(col)
    
    # If too many parameters, select the ones most correlated with error
    if len(config_cols) > max_params:
        corr_with_error = []
        for col in config_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                corr = df[col].corr(df['error'])
                if not np.isnan(corr):
                    corr_with_error.append((col, abs(corr)))
        
        # Sort by correlation and take top max_params
        corr_with_error.sort(key=lambda x: x[1], reverse=True)
        config_cols = [col for col, _ in corr_with_error[:max_params]]
    
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
        
        # Limit the number of trials for better visualization
        if len(plot_df) > 100:
            # Take a sample of 100 trials, prioritizing those with better errors
            plot_df = plot_df.sort_values('error').head(100)
        
        # Create parallel coordinates plot
        plt.figure(figsize=(12, 6))
        pd.plotting.parallel_coordinates(
            plot_df, 'error_normalized', 
            colormap=plt.cm.viridis
        )
        plt.title('Parallel Coordinates Plot of Hyperparameters')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=100)
            
        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_error_vs_params(df, save_path=None, show_plot=True, max_plots=8):
    """Plot validation error vs. hyperparameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with trial data
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
        max_plots (int): Maximum number of parameters to plot
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
    
    # If too many parameters, select the ones most correlated with error
    if len(config_cols) > max_plots:
        corr_with_error = []
        for col in config_cols:
            corr = df[col].corr(df['error'])
            if not np.isnan(corr):
                corr_with_error.append((col, abs(corr)))
        
        # Sort by correlation and take top max_plots
        corr_with_error.sort(key=lambda x: x[1], reverse=True)
        config_cols = [col for col, _ in corr_with_error[:max_plots]]
    
    if len(config_cols) > 0:
        # Create subplots
        n_cols = min(2, len(config_cols))
        n_rows = (len(config_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(config_cols):
            if i < len(axes):
                ax = axes[i]
                param_name = col.replace('config_', '')
                
                # Use fewer points for large datasets
                if len(df) > 200:
                    # Sample points, ensuring we include best and worst trials
                    sorted_df = df.sort_values('error')
                    top_df = sorted_df.head(100)
                    bottom_df = sorted_df.tail(100)
                    sample_df = pd.concat([top_df, bottom_df])
                else:
                    sample_df = df
                
                scatter = ax.scatter(sample_df[col], sample_df['error'], 
                                    c=sample_df['error'], cmap='viridis', 
                                    alpha=0.7)
                
                ax.set_xlabel(param_name)
                ax.set_ylabel('Validation Error')
                ax.set_title(f'Error vs {param_name}')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                if i == 0:
                    plt.colorbar(scatter, ax=ax, label='Error')
                
                # Use log scale for some parameters
                if param_name.lower() in ['lr', 'weight_decay', 'learning_rate', 'lambda']:
                    ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100)
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_optimization_trajectory(trials, save_path=None, show_plot=True, smooth=True, window=5):
    """Plot the optimization trajectory.
    
    Args:
        trials (list): List of trial results
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
        smooth (bool): Whether to apply smoothing to the trajectory
        window (int): Window size for smoothing
    """
    # Sort trials by completion time if available
    if all('complete_time' in trial for trial in trials if trial):
        sorted_trials = sorted((t for t in trials if t and 'complete_time' in t), 
                              key=lambda x: x['complete_time'])
    else:
        # Otherwise, use trial_id if available
        sorted_trials = sorted((t for t in trials if t and 'trial_id' in t), 
                              key=lambda x: x.get('trial_id', 0))
    
    # Extract errors
    errors = [trial['error'] for trial in sorted_trials if 'error' in trial]
    
    if not errors:
        logger.error("No valid errors found in trials")
        return
    
    # Calculate best seen so far
    best_so_far = np.minimum.accumulate(errors)
    
    plt.figure(figsize=(10, 6))
    
    # Apply smoothing if requested and if there are enough points
    if smooth and len(errors) > window:
        smoothed_errors = np.convolve(errors, np.ones(window)/window, mode='valid')
        smoothed_x = np.arange(len(smoothed_errors))
        plt.plot(errors, 'o', alpha=0.3, color='blue', label='Trial errors')
        plt.plot(smoothed_x, smoothed_errors, '-', alpha=0.7, color='blue', label='Smoothed errors')
    else:
        plt.plot(errors, 'o-', alpha=0.5, color='blue', label='Trial errors')
    
    plt.plot(best_so_far, 'r-', linewidth=2, label='Best so far')
    
    plt.xlabel('Trial')
    plt.ylabel('Validation Error')
    plt.title('Optimization Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100)
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def compute_statistics(df):
    """Compute statistics about the trials.
    
    Args:
        df (pandas.DataFrame): DataFrame with trial data
        
    Returns:
        dict: Statistics about the trials
    """
    stats = {
        'num_trials': len(df),
        'min_error': df['error'].min(),
        'max_error': df['error'].max(),
        'mean_error': df['error'].mean(),
        'median_error': df['error'].median(),
        'std_error': df['error'].std(),
    }
    
    # Add runtime statistics if available
    if 'submit_time' in df.columns and 'complete_time' in df.columns:
        df['runtime'] = pd.to_datetime(df['complete_time']) - pd.to_datetime(df['submit_time'])
        df['runtime_seconds'] = df['runtime'].dt.total_seconds()
        
        stats.update({
            'min_runtime': df['runtime_seconds'].min(),
            'max_runtime': df['runtime_seconds'].max(),
            'mean_runtime': df['runtime_seconds'].mean(),
            'median_runtime': df['runtime_seconds'].median(),
        })
    
    return stats


def main():
    """Parse arguments and analyze results."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Analyze HPO results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing HPO results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis plots')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of trials to analyze')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Only analyze top K performing trials')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plots (only save them)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed progress information')
    parser.add_argument('--max_params', type=int, default=8,
                        help='Maximum number of parameters to include in plots')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply smoothing to trajectory plot')
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine which trials to load
    limit = args.top_k or args.limit
    sort_by_error = args.top_k is not None
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(args.results_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Analyzing results from {args.results_dir}")
    logger.info(f"Output will be saved to {output_dir}")
    
    # Load results
    campaign_results, trial_results = load_results(
        args.results_dir, 
        limit=limit,
        sort_by_error=sort_by_error,
        verbose=args.verbose
    )
    
    if not trial_results:
        logger.error("No trial results found or could be loaded")
        sys.exit(1)
    
    # Create dataframe
    logger.info("Creating dataframe from trial results")
    df = create_dataframe(trial_results)
    
    if df.empty:
        logger.error("Failed to create dataframe from trial results")
        sys.exit(1)
    
    # Compute statistics
    stats = compute_statistics(df)
    
    # Display information
    logger.info(f"Loaded {len(trial_results)} trials from {args.results_dir}")
    
    if campaign_results:
        logger.info(f"Best validation error: {campaign_results.get('best_error', 'N/A')}")
        logger.info(f"Best configuration: {campaign_results.get('best_config', 'N/A')}")
    
    logger.info(f"Error statistics: min={stats['min_error']:.4f}, max={stats['max_error']:.4f}, "
               f"mean={stats['mean_error']:.4f}, median={stats['median_error']:.4f}")
    
    if 'mean_runtime' in stats:
        logger.info(f"Average runtime per trial: {stats['mean_runtime']:.2f} seconds")
    
    # Generate plots
    logger.info("Generating plots")
    
    logger.debug("Generating error distribution plot")
    plot_error_distribution(
        df, 
        best_error=campaign_results.get('best_error'),
        save_path=os.path.join(output_dir, 'error_distribution.png'),
        show_plot=not args.no_show
    )
    
    logger.debug("Generating parallel coordinates plot")
    plot_parallel_coordinates(
        df,
        save_path=os.path.join(output_dir, 'parallel_coordinates.png'),
        show_plot=not args.no_show,
        max_params=args.max_params
    )
    
    logger.debug("Generating error vs parameters plot")
    plot_error_vs_params(
        df,
        save_path=os.path.join(output_dir, 'error_vs_params.png'),
        show_plot=not args.no_show,
        max_plots=args.max_params
    )
    
    logger.debug("Generating optimization trajectory plot")
    plot_optimization_trajectory(
        trial_results,
        save_path=os.path.join(output_dir, 'optimization_trajectory.png'),
        show_plot=not args.no_show,
        smooth=args.smooth
    )
    
    # Save statistics to file
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis complete in {elapsed_time:.2f} seconds. Plots saved to {output_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during analysis: {e}")
        sys.exit(1)