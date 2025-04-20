#!/bin/bash

# This script runs the HPO campaign locally without SLURM
# It's a simpler version of submit_hpo_campaign.sh for local testing

# Print information
echo "Local HPO job started at $(date)"
echo "Running on host: $(hostname)"

# Set up environment variables
export PROJECT_DIR="/home/edson/Desktop/yang_lab/tfa-predictor"
cd $PROJECT_DIR

# Make necessary directories
mkdir -p logs
mkdir -p hpo_results

# Modify the AsyncRandomSearch class to work locally
echo "Setting up local HPO environment..."

# Run the HPO campaign
echo "Starting local HPO campaign..."
# Add the project root to PYTHONPATH to ensure modules can be found
PYTHONPATH=$PROJECT_DIR python -u hpo/main_hpo.py --config mlp_config

echo "Local HPO campaign completed at $(date)"