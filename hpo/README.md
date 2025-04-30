# Hyperparameter Optimization (HPO) Framework

This module provides a flexible and extensible framework for hyperparameter optimization of machine learning models. It supports both local execution and distributed execution on HPC clusters like SLURM.

## Directory Structure

```
hpo/
├── base/                     # Abstract base classes
│   ├── HPOSearcher.py        # Base searcher class 
│   ├── HPOScheduler.py       # Base scheduler class
│   └── HPOTuner.py           # Base tuner class
├── searchers/                # Concrete searcher implementations
│   ├── RandomSearcher.py     # Random search implementation
│   └── AsyncRandomSearch.py  # Async implementation for cluster jobs
├── schedulers/               # Concrete scheduler implementations
│   ├── BasicScheduler.py     # Basic scheduler for local execution
│   └── Scheduler.py          # SLURM scheduler for cluster jobs
├── utils/                    # HPO-specific utilities
│   └── serialization.py      # Utility functions (e.g., numpy_to_python)
├── config/                   # Configuration files
│   └── mlp_config.py         # MLP hyperparameter search space
├── examples/                 # Example scripts
│   └── optimize_mlp.py       # Example of MLP optimization
├── scripts/                  # Script utilities
│   ├── analyze_results.py           # Analyze HPO results
│   ├── analyze_results_optimized.py # Optimized analysis for large trials
│   ├── debug_hpo.sh                 # Debug script
│   ├── run_local_hpo.sh             # Local run script
│   └── submit_hpo_campaign.sh       # Cluster submission script
├── main_hpo.py               # Main entry point for HPO campaigns
└── run_trial.py              # Trial runner
```

## Core Components

### Base Classes

- **HPOSearcher**: Abstract base class for hyperparameter search algorithms
- **HPOScheduler**: Abstract base class for scheduling and managing HPO trials
- **HPOTuner**: Base class for tuning hyperparameters with a given scheduler and objective

### Searchers

- **RandomSearcher**: Simple random search implementation
- **AsyncRandomSearch**: Asynchronous random search for distributed execution on HPC clusters

### Schedulers

- **BasicScheduler**: Simple scheduler for local execution
- **Scheduler**: SLURM-based scheduler for running on clusters

## Usage

### Local Optimization

```python
from hpo.searchers.RandomSearcher import RandomSearcher
from hpo.schedulers.BasicScheduler import BasicScheduler
from hpo.base.HPOTuner import HPOTuner

# Define search space
config_space = {
    'lr': loguniform(1e-4, 1e-2),
    'hidden_sizes': [[64, 64], [128, 128], [256, 256]],
    'dropout_rate': uniform(0, 0.5)
}

# Define objective function
def objective(lr, hidden_sizes, dropout_rate):
    # Train and evaluate model
    # Return validation error
    return validation_error

# Create searcher and scheduler
searcher = RandomSearcher(config_space)
scheduler = BasicScheduler(searcher)

# Create tuner and run optimization
tuner = HPOTuner(scheduler, objective)
tuner.run(number_of_trials=50)
```

### Cluster Optimization

```bash
# Submit HPO campaign to cluster
python -m hpo.main_hpo --config mlp_config
```

## Analysis

After running an HPO campaign, analyze the results with:

```bash
# For small to medium campaigns (< 100 trials)
python hpo/scripts/analyze_results.py --results_dir hpo_results/experiment_name

# For large campaigns (100+ trials)
python hpo/scripts/analyze_results_optimized.py --results_dir hpo_results/experiment_name --limit 50
```

## Adding New Components

### New Search Algorithm

1. Create a new file in `searchers/` (e.g., `BayesianSearcher.py`)
2. Extend the `HPOSearcher` base class
3. Implement the required methods: `sample_configuration()` and `update()`

### New Scheduler

1. Create a new file in `schedulers/` (e.g., `KubernetesScheduler.py`) 
2. Extend the `HPOScheduler` base class
3. Implement the required methods: `suggest()` and `update()`

## Configuration

Create a configuration file in the `config/` directory (see `mlp_config.py` for an example).
Each configuration file should define:

1. `get_config_space()`: Returns the hyperparameter search space
2. `get_slurm_config()`: Returns SLURM configuration (for cluster execution)
3. `get_hpo_config()`: Returns general HPO configuration (number of trials, etc.)