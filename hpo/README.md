# Asynchronous Random Search for Hyperparameter Optimization on HPC Clusters

This module implements asynchronous random search for hyperparameter optimization (HPO) on HPC clusters, with specific support for Rutgers' Amarel HPC cluster. It handles job submission and monitoring through SLURM, provides fault tolerance through checkpointing, and optimizes communication between workers to minimize overhead.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running on Amarel](#running-on-amarel)
- [Monitoring and Analysis](#monitoring-and-analysis)
- [Example](#example)

## Overview

The asynchronous random search algorithm distributes hyperparameter trials across multiple compute nodes on the Amarel cluster. It efficiently manages the submission and monitoring of SLURM jobs, ensuring maximum utilization of the available resources.

Key features:
- Asynchronous evaluation of hyperparameter configurations
- Integration with SLURM workload manager
- Automatic checkpointing and fault tolerance
- Real-time monitoring of optimization progress
- Seamless integration with existing models and data loaders

## Components

The implementation consists of several components:

- `AsyncRandomSearch`: The core searcher that implements random sampling and manages the search state
- `Scheduler`: Scheduler that handles job submission and result collection
- `run_trial.py`: Script for running individual hyperparameter trials on compute nodes
- `main_hpo.py`: Main script for running HPO campaigns
- `config/`: Directory containing search space and cluster configurations
- `scripts/`: Directory containing job submission scripts

## Quick Start

1. Create a configuration file for your model in `hpo/config/`
2. Customize the SLURM settings in the configuration file
3. Submit the HPO campaign using the provided submission script:

```bash
sbatch hpo/scripts/submit_hpo_campaign.sh
```

## Configuration

### Search Space Definition

Define your hyperparameter search space in a configuration file:

```python
# hpo/config/my_model_config.py
def get_config_space():
    return {
        'input_size': 3883,  # Fixed value
        'output_size': 214,  # Fixed value
        'hidden_sizes': [
            [1024, 512, 256], 
            [2048, 1024, 512]
        ],  # Categorical
        'lr': loguniform(1e-4, 1e-2),  # Continuous (log scale)
        'weight_decay': loguniform(1e-5, 1e-3),  # Continuous (log scale)
        'batch_size': [32, 64, 128],  # Categorical
        'max_epochs': [50, 100, 200],  # Categorical
    }
```

### SLURM Configuration

Configure SLURM settings for your jobs:

```python
def get_slurm_config():
    return {
        'partition': 'main',  # Amarel partition
        'time': '4:00:00',    # Time limit per job
        'mem': '16G',         # Memory per job
        'cpus_per_task': 4,   # CPUs per job
        'gpus': 1,            # GPUs per job (set to 0 if not needed)
        'env_setup': 'source activate tfa-predictor',  # Environment setup
    }
```

### HPO Campaign Configuration

Configure the overall HPO campaign:

```python
def get_hpo_config():
    return {
        'n_trials': 100,               # Total number of trials
        'max_concurrent_jobs': 8,      # Maximum concurrent jobs
        'output_dir': 'hpo_results/my_model',  # Results directory
        'checkpoint_interval': 5,      # Checkpoint frequency
        'seed': 42,                    # Random seed
    }
```

## Running on Amarel

### Step 1: Prepare Your Configuration

Create a configuration file for your model in `hpo/config/`.

### Step 2: Customize the Submission Script

Update the submission script `hpo/scripts/submit_hpo_campaign.sh` with your email and other details.

### Step 3: Submit the HPO Campaign

```bash
sbatch hpo/scripts/submit_hpo_campaign.sh
```

### Step 4: Monitor Progress

Check the status of your campaign:

```bash
# View the manager job status
squeue -u your_netid

# Check the log file for progress
tail -f hpo_manager_*.out

# Check individual trial jobs
squeue -u your_netid | grep trial
```

## Monitoring and Analysis

### Real-time Monitoring

The HPO campaign logs information to both console and log files:

- Overall progress: `hpo_campaign_*.log`
- Manager job output: `hpo_manager_*.out`
- Individual trial outputs: `trial_*/job_*.out`

### Results Analysis

After the campaign completes:

1. Best configuration: `hpo_results/*/best_config.json`
2. All trial results: `hpo_results/*/results/`
3. Learning curves: `hpo_results/*/plots/` (if generated)

To analyze the results, you can use the provided utilities or custom scripts.

## Example

Here's a complete example of running the asynchronous random search for optimizing an MLP model:

### 1. Create Configuration

```python
# hpo/config/mlp_example.py
import numpy as np
from scipy.stats import uniform, loguniform

def get_config_space():
    return {
        'input_size': 3883,
        'output_size': 214,
        'hidden_sizes': [
            [1024, 512, 256],
            [2048, 1024, 512],
        ],
        'lr': loguniform(1e-4, 1e-2),
        'weight_decay': loguniform(1e-5, 1e-3),
        'batch_size': [32, 64, 128],
        'max_epochs': [50, 100],
    }

def get_slurm_config():
    return {
        'partition': 'main',
        'time': '2:00:00',
        'mem': '16G',
        'cpus_per_task': 4,
        'gpus': 1,
        'env_setup': 'source activate tfa-predictor',
    }

def get_hpo_config():
    return {
        'n_trials': 50,
        'max_concurrent_jobs': 4,
        'output_dir': 'hpo_results/mlp_example',
        'checkpoint_interval': 5,
        'seed': 42,
    }
```

### 2. Submit HPO Campaign

```bash
# Submit the HPO campaign
sbatch hpo/scripts/submit_hpo_campaign.sh --config mlp_example
```

### 3. Analyze Results

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

# Load results
with open('hpo_results/mlp_example/final_results.json', 'r') as f:
    results = json.load(f)

print(f"Best configuration: {results['best_config']}")
print(f"Best error: {results['best_error']}")

# Load all trials
trials = []
for i in range(results['n_trials']):
    try:
        with open(f"hpo_results/mlp_example/results/trial_{i}.json", 'r') as f:
            trial = json.load(f)
            trials.append(trial)
    except FileNotFoundError:
        continue

# Create DataFrame
df = pd.DataFrame([
    {
        'trial_id': t['trial_id'],
        'error': t['error'],
        'runtime': t['additional_info']['training_time'],
        'lr': t['config']['lr'],
        'weight_decay': t['config']['weight_decay'],
        'hidden_sizes': str(t['config']['hidden_sizes']),
        'batch_size': t['config']['batch_size'],
        'max_epochs': t['config']['max_epochs'],
    }
    for t in trials
])

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(df['error'], bins=20)
plt.axvline(results['best_error'], color='r', linestyle='--')
plt.title('Distribution of Validation Errors')
plt.xlabel('Validation Error')
plt.ylabel('Frequency')
plt.show()

# Plot error vs hyperparameters
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].scatter(df['lr'], df['error'])
axs[0, 0].set_xlabel('Learning Rate')
axs[0, 0].set_ylabel('Validation Error')
axs[0, 0].set_xscale('log')

axs[0, 1].scatter(df['weight_decay'], df['error'])
axs[0, 1].set_xlabel('Weight Decay')
axs[0, 1].set_ylabel('Validation Error')
axs[0, 1].set_xscale('log')

axs[1, 0].scatter(df['batch_size'], df['error'])
axs[1, 0].set_xlabel('Batch Size')
axs[1, 0].set_ylabel('Validation Error')

axs[1, 1].scatter(df['max_epochs'], df['error'])
axs[1, 1].set_xlabel('Max Epochs')
axs[1, 1].set_ylabel('Validation Error')

plt.tight_layout()
plt.show()
```

By following these instructions, you can effectively use the asynchronous random search implementation to optimize hyperparameters for your models on the Amarel HPC cluster.