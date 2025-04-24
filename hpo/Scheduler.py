import os
import json
import time
import logging
from datetime import datetime

from hpo.HPOScheduler import HPOScheduler
from hpo.AsyncRandomSearch import AsyncRandomSearch


class Scheduler(HPOScheduler):
    """Scheduler for running hyperparameter optimization on HPC clusters.
    
    This scheduler works with the AsyncRandomSearch to manage
    trial evaluation on HPC clusters. It handles job submission,
    monitoring, and results collection.
    """
    
    def __init__(self, searcher, slurm_config=None):
        """Initialize the HPC scheduler.
        
        Args:
            searcher: The HPO searcher to use (typically AsyncRandomSearch)
            slurm_config (dict, optional): SLURM configuration options
        """
        super().__init__()
        self.save_hyperparameters(ignore=['slurm_config'])
        
        self.searcher = searcher
        self.slurm_config = slurm_config or {}
        
        # Setup logging
        self.logger = logging.getLogger('amarel_scheduler')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if not already added
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def suggest(self):
        """Suggest a new configuration to evaluate.
        
        Returns:
            dict: A configuration to evaluate
        """
        return self.searcher.sample_configuration()
    
    def update(self, config, error, info=None):
        """Update the scheduler with results from a trial.
        
        Args:
            config (dict): The configuration that was evaluated
            error (float): The validation error/score
            info (dict, optional): Additional information from the trial
        """
        self.searcher.update(config, error, info)
    
    def create_job_script(self, config_path, output_path, script_path=None):
        """Create a SLURM job script for the given configuration.
        
        Args:
            config_path (str): Path to the configuration file
            output_path (str): Path to write results
            script_path (str, optional): Path to save the job script
        
        Returns:
            str: The job script content
        """
        # Default SLURM settings
        slurm_settings = {
            'partition': self.slurm_config.get('partition', 'main'),
            'time': self.slurm_config.get('time', '4:00:00'),
            'mem': self.slurm_config.get('mem', '8G'),
            'cpus_per_task': self.slurm_config.get('cpus_per_task', 4),
            'gpus': self.slurm_config.get('gpus', 0)
        }
        
        # Job script header
        header = [
            "#!/bin/bash",
            f"#SBATCH --partition={slurm_settings['partition']}",
            f"#SBATCH --time={slurm_settings['time']}",
            f"#SBATCH --mem={slurm_settings['mem']}",
            f"#SBATCH --cpus-per-task={slurm_settings['cpus_per_task']}",
            f"#SBATCH --job-name=hpo_trial",
            f"#SBATCH --output=%x-%j.out"
        ]
        
        # Add GPU resources if needed
        if 'gpus' in slurm_settings and slurm_settings['gpus'] > 0:
            header.append(f"#SBATCH --gres=gpu:{slurm_settings['gpus']}")
        
        # Add any additional SLURM settings
        for key, value in self.slurm_config.items():
            if key not in ['partition', 'time', 'mem', 'cpus_per_task', 'gpus']:
                header.append(f"#SBATCH --{key}={value}")
        
        # Make script body
        script_body = [
            "",
            "# Print some information about the job",
            "echo Running on host `hostname`",
            "echo Time is `date`",
            "echo Directory is `pwd`",
            "echo SLURM job ID is $SLURM_JOB_ID",
            "",
            "# Load required modules",
            "module purge",
            "module load python/3.8.2",
            "",
            "# Set up environment for trial",
            "cd /home/elp95/tfa-predictor",
            "",
            "# Run the trial",
            "echo Starting trial...",
            f"python -m hpo.run_trial --config {config_path} --output {output_path}",
            "",
            "echo Trial completed!",
            "exit 0"
        ]
        
        # Combine script parts
        script = "\n".join(header + script_body)
        
        # Save script if path is provided
        if script_path:
            with open(script_path, 'w') as f:
                f.write(script)
            os.chmod(script_path, 0o755)  # Make executable
        
        return script
    
    def run_hpo_campaign(self, n_trials, output_dir, max_concurrent_jobs=4, resume_from=None):
        """Run a hyperparameter optimization campaign on an HPC cluster.
        
        This function should be called from a submit script that manages
        the overall HPO process.
        
        Args:
            n_trials (int): Number of trials to run
            output_dir (str): Directory to store results
            max_concurrent_jobs (int): Maximum number of concurrent SLURM jobs
            resume_from (str, optional): Path to checkpoint to resume from
            
        Returns:
            dict: Results of the HPO campaign
        """
        if not isinstance(self.searcher, AsyncRandomSearch):
            self.logger.warning("Scheduler is designed to work with AsyncRandomSearch")
            self.logger.warning(f"Current searcher is: {type(self.searcher).__name__}")
        
        # Create job script template
        job_script_template = self.create_job_script(
            config_path="{config_path}",
            output_path="{output_dir}/jobs/trial_{trial_id}/results.json"
        )
        
        # Run the search
        results = self.searcher.run(
            n_trials=n_trials,
            job_script_template=job_script_template,
            resume_from=resume_from
        )
        
        # Save final results
        output_file = os.path.join(output_dir, 'final_results.json')
        with open(output_file, 'w') as f:
            json.dump({
                'best_config': results['best_config'],
                'best_error': results['best_error'],
                'n_trials': results['n_trials'],
                'elapsed_time': results['elapsed_time'],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        self.logger.info(f"HPO campaign completed. Results saved to {output_file}")
        
        return results