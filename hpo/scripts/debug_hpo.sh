#!/bin/bash

# This script helps debug the HPO campaign locally

# Print information
echo "Debug script started at $(date)"
echo "Running on host: $(hostname)"

# Set up environment variables
export PROJECT_DIR="/home/edson/Desktop/yang_lab/tfa-predictor"
cd $PROJECT_DIR

# Make necessary directories
mkdir -p logs
mkdir -p hpo_results

# Run the HPO campaign with minimal config
echo "Starting HPO campaign in debug mode..."

# Run with verbose Python output
export PYTHONVERBOSE=1
python -u hpo/main_hpo.py --config mlp_config

echo "Debug completed at $(date)"