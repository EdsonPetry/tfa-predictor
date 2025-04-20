#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=48:00:00
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

# Load necessary modules
module purge
module load python/3.10.4

# Activate virtual environment
source activate tfa-predictor

# Set up environment variables
export PROJECT_DIR="/home/elp95/tfa-predictor"
cd $PROJECT_DIR

# Make necessary directories
mkdir -p logs
mkdir -p hpo_results

# Run the HPO campaign
echo "Starting HPO campaign..."
python -u hpo/main_hpo.py --config mlp_config

echo "HPO campaign completed at $(date)"
