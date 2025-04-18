import os
import time
import json
import uuid
import signal
import logging
import pickle
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np

from hpo.HPOSearcher import HPOSearcher
from utils.HyperParameters import HyperParameters


class AsyncRandomSearch(HPOSearcher):
    """Asynchronous Random Search implementation for HPC clusters.
    
    This searcher manages a pool of asynchronous workers for hyperparameter
    optimization using SLURM workload manager. It uses random sampling
    to explore the hyperparameter space, while efficiently distributing
    trials across multiple compute nodes.
    """
    
    def __init__(self, config_space, 
                 output_dir='./hpo_results',
                 max_concurrent_jobs=4,
                 checkpoint_interval=10,
                 initial_config=None,
                 seed=None):
        """Initialize the Amarel asynchronous random search.
        
        Args:
            config_space (dict): Dictionary mapping parameter names to their search domains
            output_dir (str): Directory to store results, logs, and checkpoints
            max_concurrent_jobs (int): Maximum number of jobs to run concurrently
            checkpoint_interval (int): Save checkpoint every N trials
            initial_config (dict, optional): Initial configuration to evaluate
            seed (int, optional): Random seed for reproducibility
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Set random seed provided
        if seed is not None:
            np.random.seed(seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'jobs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize state
        self.active_jobs = {}
        self.completed_trials = []
        self.best_config = None
        self.best_error = float('inf')
        self.trial_counter = 0
        self.start_time = time.time()
        
        self.logger.info(f"Initialized AmarelAsyncRandomSearch with output_dir={output_dir}")
        self.logger.info(f"Configuration space: {config_space}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logger = logging.getLogger('amarel_async_search')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.output_dir, 'search.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def sample_configuration(self):
        """Sample a new configuration from the search space.
        
        Returns:
            dict: A sampled configuration
        """
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
            self.logger.info(f"Using initial configuration: {result}")
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
            self.logger.debug(f"Sampled new configuration: {result}")
        
        return result
    
    def update(self, config, error, additional_info=None):
        """Update the searcher with results from a completed trial.
        
        Args:
            config (dict): The configuration that was evaluated
            error (float): The validation error/score
            additional_info (dict, optional): Additional information from the trial
        """
        self.completed_trials.append({
            'config': config,
            'error': error,
            'additional_info': additional_info,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update best configuration if this result is better
        if error < self.best_error:
            self.best_config = config
            self.best_error = error
            self.logger.info(f"New best configuration found! Error: {error}")
            self.logger.info(f"Best config: {config}")
            
            # Save best config to file
            with open(os.path.join(self.output_dir, 'best_config.json'), 'w') as f:
                json.dump({
                    'config': config,
                    'error': error,
                    'trial': len(self.completed_trials),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        
        # Save checkpoint if needed
        if len(self.completed_trials) % self.checkpoint_interval == 0:
            self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save a checkpoint of the current state."""
        checkpoint_path = os.path.join(
            self.output_dir, 'checkpoints', 
            f'checkpoint_{len(self.completed_trials)}.pkl'
        )
        
        state = {
            'completed_trials': self.completed_trials,
            'active_jobs': self.active_jobs,
            'best_config': self.best_config,
            'best_error': self.best_error,
            'trial_counter': self.trial_counter,
            'start_time': self.start_time,
            'current_time': time.time()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load state from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        self.completed_trials = state['completed_trials']
        self.active_jobs = state['active_jobs']
        self.best_config = state['best_config']
        self.best_error = state['best_error']
        self.trial_counter = state['trial_counter']
        self.start_time = state['start_time']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Restored {len(self.completed_trials)} completed trials")
        self.logger.info(f"Best error so far: {self.best_error}")
    
    
    
    def _submit_job(self, config, job_script_template):
        """Submit a job locally (SLURM bypass for local testing).
        
        Args:
            config (dict): Configuration to evaluate
            job_script_template (str): Template for SLURM job script
        
        Returns:
            str: Job ID
        """
        trial_id = self.trial_counter
        self.trial_counter += 1
        
        # Create job directory
        job_dir = os.path.join(self.output_dir, 'jobs', f'trial_{trial_id}')
        os.makedirs(job_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(job_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run the job directly instead of submitting to SLURM
        import subprocess
        import uuid
        
        job_id = str(uuid.uuid4())
        
        # Record job info
        self.active_jobs[job_id] = {
            'trial_id': trial_id,
            'config': config,
            'submit_time': datetime.now().isoformat(),
            'job_dir': job_dir
        }
        
        # Run the job in a separate process
        cmd = f"python -m hpo.run_trial --config {config_path} --output {self.output_dir}/jobs/trial_{trial_id}/results.json"
        
        # Run in background (this is just for demonstration - in practice, you'd want a more robust solution)
        subprocess.Popen(cmd, shell=True)
        
        self.logger.info(f"Started local job {job_id} for trial {trial_id}")
        
        return job_id


    
    
    def _check_job_status(self, job_id):
        """Check if job results exist (local version).
        
        Args:
            job_id (str): Job ID
        
        Returns:
            str: Job status
        """
        job_info = self.active_jobs[job_id]
        results_path = os.path.join(job_info['job_dir'], 'results.json')
        
        if os.path.exists(results_path):
            return "COMPLETED"
        else:
            # Wait a bit before checking again
            import time
            time.sleep(0.5)
            return "RUNNING"

    
    def _process_completed_job(self, job_id):
        """Process results from a completed job.
        
        Args:
            job_id (str): SLURM job ID
        
        Returns:
            bool: True if results were successfully processed, False otherwise
        """
        job_info = self.active_jobs[job_id]
        trial_id = job_info['trial_id']
        
        # Check for results file
        results_path = os.path.join(job_info['job_dir'], 'results.json')
        if not os.path.exists(results_path):
            self.logger.error(f"Results file not found for job {job_id} (trial {trial_id})")
            return False
        
        # Load results
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            config = job_info['config']
            error = results['error']
            additional_info = results.get('additional_info', None)
            
            # Update the searcher with the results
            self.update(config, error, additional_info)
            
            # Copy results to the results directory
            output_path = os.path.join(self.output_dir, 'results', f'trial_{trial_id}.json')
            with open(output_path, 'w') as f:
                json.dump({
                    'trial_id': trial_id,
                    'config': config,
                    'error': error,
                    'additional_info': additional_info,
                    'job_id': job_id,
                    'submit_time': job_info['submit_time'],
                    'complete_time': datetime.now().isoformat()
                }, f, indent=2)
            
            self.logger.info(f"Processed results for job {job_id} (trial {trial_id}): error={error}")
            
            # Remove job from active jobs
            del self.active_jobs[job_id]
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error processing results for job {job_id}: {e}")
            return False
    
    def run(self, n_trials, job_script_template, resume_from=None):
        """Run the asynchronous random search.
        
        Args:
            n_trials (int): Total number of trials to run
            job_script_template (str): Template for SLURM job script
            resume_from (str, optional): Path to checkpoint file to resume from
        
        Returns:
            dict: Best configuration found
        """
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Register signal handlers for graceful shutdown
        def handle_signal(sig, frame):
            self.logger.info(f"Received signal {sig}, saving checkpoint and exiting")
            self._save_checkpoint()
            exit(0)
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        self.logger.info(f"Starting asynchronous random search for {n_trials} trials")
        self.logger.info(f"Maximum concurrent jobs: {self.max_concurrent_jobs}")
        
        while len(self.completed_trials) < n_trials:
            # Check status of active jobs
            jobs_to_remove = []
            for job_id in list(self.active_jobs.keys()):
                status = self._check_job_status(job_id)
                
                if status in ['COMPLETED', 'COMPLETING']:
                    if self._process_completed_job(job_id):
                        jobs_to_remove.append(job_id)
                
                elif status in ['FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL']:
                    self.logger.warning(f"Job {job_id} failed with status {status}")
                    jobs_to_remove.append(job_id)
            
            # Submit new jobs if slots are available
            while (len(self.active_jobs) < self.max_concurrent_jobs and 
                   len(self.completed_trials) + len(self.active_jobs) < n_trials):
                config = self.sample_configuration()
                self._submit_job(config, job_script_template)
            
            # Print status
            elapsed = time.time() - self.start_time
            self.logger.info(f"Status: {len(self.completed_trials)}/{n_trials} trials completed, "
                           f"{len(self.active_jobs)} jobs active, best error: {self.best_error}, "
                           f"elapsed: {elapsed:.1f}s")
            
            # Sleep before next check
            time.sleep(30)
        
        self.logger.info(f"Search completed, best error: {self.best_error}")
        self.logger.info(f"Best config: {self.best_config}")
        
        return {
            'best_config': self.best_config,
            'best_error': self.best_error,
            'n_trials': len(self.completed_trials),
            'elapsed_time': time.time() - self.start_time
        }