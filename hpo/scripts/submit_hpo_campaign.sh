#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --job-name=hpo_manager
#SBATCH --output=hpo_manager_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=edson.petry@rutgers.edu

# This script serves as the main entry point for running an HPO campaign
# on the Amarel cluster. It starts a manager process that submits
# and monitors individual trial jobs.

# Print job information
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM job ID: $SLURM_JOB_ID"

# Set up environment variables
export PROJECT_DIR="/home/elp95/tfa-predictor"
cd $PROJECT_DIR

# Add project directory to Python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
echo "PYTHONPATH set to: $PYTHONPATH"

# Load necessary modules
module purge
module load python/3.12.3

# Set up debugging information
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Contents of data directory:"
ls -la $PROJECT_DIR/data

# Make necessary directories
mkdir -p logs
mkdir -p hpo_results

# Run the HPO campaign
echo "Starting HPO campaign..."
# Use python module format to ensure proper imports
python -m hpo.main_hpo --config mlp_config

echo "HPO campaign completed at $(date)"
