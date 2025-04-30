#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=hpo_trial
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=edson.petry@rutgers.edu
#SBATCH --mail-type=END,FAIL

# Print some information about the job
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo SLURM job ID is $SLURM_JOB_ID

# Load required modules
module purge
module load python/3.8.2

# Set up environment for trial
cd /home/elp95/tfa-predictor

# Run the trial
echo Starting trial...
python -m hpo.run_trial --config /home/elp95/tfa-predictor/hpo_results/mlp_extended3/jobs/trial_186/config.json --output /home/elp95/tfa-predictor/hpo_results/mlp_extended3/jobs/trial_186/results.json

echo Trial completed!
exit 0