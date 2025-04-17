# TFA-Predictor

A deep learning framework for predicting Transcription Factor Activities (TFAs) from gene expression data.

## Overview

TFA-Predictor is a PyTorch-based machine learning toolkit designed for predicting transcription factor activities from gene expression profiles. It provides a modular and extensible architecture for training, evaluating, and optimizing deep learning models for regulatory genomics research.

## Features

- **Modular Architecture**: Extensible base classes for models, data, and trainers
- **Hyperparameter Optimization**: Distributed HPO on Amarel HPC cluster
- **Visualization**: Tools for model learning and prediction analysis
- **TF Activity Prediction**: MLP-based models for predicting TF activities

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/tfa-predictor.git
cd tfa-predictor

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
tfa-predictor/
├── base/                 # Base class implementations
│   ├── BaseDataModule.py # Base data module
│   ├── BaseModule.py     # Base model
│   └── BaseTrainer.py    # Base trainer
├── data/                 # Data storage directory
├── data_loader/          # Data loading utilities
│   └── Data.py           # Data module implementation
├── hpo/                  # Hyperparameter optimization
│   ├── AsyncRandomSearch.py        # Async random search for HPC
│   ├── BasicScheduler.py           # Basic HPO scheduler
│   ├── config/                     # HPO configurations
│   ├── examples/                   # Example HPO scripts
│   ├── HPOScheduler.py             # Base HPO scheduler
│   ├── HPOSearcher.py              # Base HPO searcher
│   ├── HPOTuner.py                 # HPO tuner implementation
│   ├── main_hpo.py                 # Main HPO entry point
│   ├── RandomSearcher.py           # Random search implementation
│   ├── README.md                   # HPO documentation
│   ├── run_trial.py                # Single HPO trial runner
│   ├── Scheduler.py                # HPO scheduler implementation
│   └── scripts/                    # HPO utility scripts
├── models/               # Model implementations
│   └── MLP.py            # Multi-layer perceptron
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt      # Python dependencies
├── trainer/              # Training implementations
│   └── Trainer.py        # Trainer for model training
└── utils/                # Utility functions
    ├── HyperParameters.py # Hyperparameter management
    └── visualization.py   # Visualization tools
```

## Usage

### Training a Model

```python
from data_loader.Data import Data
from models.MLP import MLP
from trainer.Trainer import Trainer

# Load data
data = Data(batch_size=32)

# Create model
model = MLP(input_size=3883, output_size=214)

# Train model
trainer = Trainer(max_epochs=100, visualize=True)
trainer.fit(model, data)
```

### Running from Command Line

```bash
# Train a model
python trainer/Trainer.py

# Hyperparameter optimization (local)
python hpo/examples/optimize_mlp.py

# Hyperparameter optimization (Amarel)
sbatch hpo/scripts/submit_hpo_campaign.sh --config mlp_config
```

### Visualizing Results

The framework includes built-in visualization tools for analyzing model performance:

```python
# View learning curves and prediction quality
trainer = Trainer(max_epochs=100, visualize=True, save_plots=True)
trainer.fit(model, data)

# Analyze HPO results
python hpo/scripts/analyze_results.py --results_dir hpo_results/mlp
```

## Hyperparameter Optimization

TFA-Predictor includes a comprehensive hyperparameter optimization framework with support for distributed execution on the Rutgers Amarel HPC cluster. See the [HPO README](/hpo/README.md) for detailed instructions.

Key features:
- Asynchronous random search
- Integration with SLURM for job management
- Fault tolerance with checkpointing
- Visualization and analysis tools

## Examples

Check the notebooks directory for examples:

- `notebooks/data.ipynb`: Data exploration and preprocessing
- `notebooks/mlp.ipynb`: Basic MLP model training
- `notebooks/model_visualization.ipynb`: Visualizing model learning and performance

## Model Architecture

The default model is a Multi-Layer Perceptron (MLP) with configurable hidden layers, optimized for predicting TF activities from gene expression:

```
Input (Gene Expression: 3883 features)
  ↓
Hidden Layer 1 (1024 neurons, Tanh, BatchNorm, Dropout)
  ↓
Hidden Layer 2 (512 neurons, Tanh, BatchNorm, Dropout)
  ↓
Hidden Layer 3 (256 neurons, Tanh, BatchNorm, Dropout)
  ↓
Output (TF Activities: 214 values)
```

## Data

The framework expects gene expression data and TF activity labels in CSV format:
- Gene expression data: Samples × Genes matrix
- TF activity data: Samples × TFs matrix

Place your data in the `data/` directory, organized as:
```
data/
├── gene-xprs/
│   └── processed/
│       └── xprs-data.csv
└── tfa/
    └── processed/
        └── tfa-labels.csv
```

## Contributing

Contributions to TFA-Predictor are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use TFA-Predictor in your research, please cite:

```
@software{tfa_predictor,
  author = {Your Name},
  title = {TFA-Predictor: Deep Learning Framework for Transcription Factor Activity Prediction},
  year = {2023},
  url = {https://github.com/your-username/tfa-predictor}
}
```

## Acknowledgments

- Yang Lab at Rutgers University
- Rutgers Office of Advanced Research Computing (OARC) for providing access to the Amarel cluster