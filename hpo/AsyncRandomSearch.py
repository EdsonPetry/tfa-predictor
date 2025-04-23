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


def numpy_to_python(obj):
    """Convert NumPy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj


class AsyncRandomSearch(HPOSearcher):
    """Asynchronous Random Search implementation for HPC clusters.

    This searcher manages a pool of asynchronous workers for hyperparameter
    optimization using SLURM workload manager. It uses random sampling
    to explore the hyperparameter space, while efficiently distributing
    trials across multiple compute nodes.
    """

    def __init__(
        self,
        config_space,
        output_dir="./hpo_results",
        max_concurrent_jobs=4,
        checkpoint_interval=10,
        initial_config=None,
        seed=None,
    ):
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
        os.makedirs(os.path.join(output_dir, "jobs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize state
        self.active_jobs = {}
        self.completed_trials = []
        self.best_config = None
        self.best_error = float("inf")
        self.trial_counter = 0
        self.start_time = time.time()

        self.logger.info(
            f"Initialized AmarelAsyncRandomSearch with output_dir={output_dir}"
        )
        self.logger.info(f"Configuration space: {config_space}")

    def _setup_logging(self):
        """Set up logging configuration."""
        logger = logging.getLogger("amarel_async_search")
        logger.setLevel(logging.INFO)

        # Create file handler
        log_file = os.path.join(self.output_dir, "search.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add to handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
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
            result = {}
            for name, domain in self.config_space.items():
                # Handle different types of parameters
                if hasattr(domain, "rvs"):  # Distributions like uniform, loguniform
                    result[name] = domain.rvs()
                elif isinstance(domain, list):  # Lists of options
                    if name == 'hidden_sizes' or any(isinstance(x, list) for x in domain):
                        # For nested lists like hidden_sizes, use Python's random
                        import random
                        result[name] = random.choice(domain)
                    else:
                        result[name] = np.random.choice(domain)
                else:  # Fixed values
                    result[name] = domain

            self.logger.debug(f"Sampled new configuration: {result}")

        return result

    def update(self, config, error, additional_info=None):
        """Update the searcher with results from a completed trial.

        Args:
            config (dict): The configuration that was evaluated
            error (float): The validation error/score
            additional_info (dict, optional): Additional information from the trial
        """
        self.completed_trials.append(
            {
                "config": config,
                "error": error,
                "additional_info": additional_info,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Update best configuration if this result is better
        if error < self.best_error:
            self.best_config = config
            self.best_error = error
            self.logger.info(f"New best configuration found! Error: {error}")
            self.logger.info(f"Best config: {config}")

            # Save best config to file - convert NumPy types to Python native types
            serializable_config = numpy_to_python(config)

            with open(os.path.join(self.output_dir, "best_config.json"), "w") as f:
                json.dump(
                    {
                        "config": serializable_config,
                        "error": (
                            float(error) if isinstance(error, np.number) else error
                        ),
                        "trial": len(self.completed_trials),
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

        # Save checkpoint if needed
        if len(self.completed_trials) % self.checkpoint_interval == 0:
            self._save_checkpoint()

    def _save_checkpoint(self):
        """Save a checkpoint of the current state."""
        checkpoint_path = os.path.join(
            self.output_dir,
            "checkpoints",
            f"checkpoint_{len(self.completed_trials)}.pkl",
        )

        state = {
            "completed_trials": self.completed_trials,
            "active_jobs": self.active_jobs,
            "best_config": self.best_config,
            "best_error": self.best_error,
            "trial_counter": self.trial_counter,
            "start_time": self.start_time,
            "current_time": time.time(),
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)

        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path):
        """Load state from a checkpoint file.

        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)

        self.completed_trials = state["completed_trials"]
        self.active_jobs = state["active_jobs"]
        self.best_config = state["best_config"]
        self.best_error = state["best_error"]
        self.trial_counter = state["trial_counter"]
        self.start_time = state["start_time"]

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Restored {len(self.completed_trials)} completed trials")
        self.logger.info(f"Best error so far: {self.best_error}")

    def _submit_job(self, config, job_script_template):
        """Submit a job to SLURM for Amarel HPC.

        Args:
            config (dict): Configuration to evaluate
            job_script_template (str): Template for SLURM job script

        Returns:
            str: SLURM Job ID
        """
        trial_id = self.trial_counter
        self.trial_counter += 1

        # Create job directory
        job_dir = os.path.join(self.output_dir, "jobs", f"trial_{trial_id}")
        os.makedirs(job_dir, exist_ok=True)

        # Save configuration - convert NumPy types to Python native types
        config_path = os.path.join(job_dir, "config.json")
        serializable_config = numpy_to_python(config)

        with open(config_path, "w") as f:
            json.dump(serializable_config, f, indent=2)

        # Create job script from template
        job_script_path = os.path.join(job_dir, "job_script.sh")
        
        # Replace placeholders in the template
        job_script = job_script_template.format(
            config_path=config_path,
            output_dir=self.output_dir,
            trial_id=trial_id
        )
        
        with open(job_script_path, "w") as f:
            f.write(job_script)
        
        # Make script executable
        os.chmod(job_script_path, 0o755)
        
        # Submit job to SLURM using sbatch
        import subprocess
        
        try:
            # Submit the job script to SLURM using sbatch
            cmd = ["sbatch", job_script_path]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Extract job ID from sbatch output (format: "Submitted batch job 12345")
            job_id = result.stdout.strip().split()[-1]
            
            # Record job info
            self.active_jobs[job_id] = {
                "trial_id": trial_id,
                "config": config,
                "submit_time": datetime.now().isoformat(),
                "job_dir": job_dir,
                "slurm_job_id": job_id
            }
            
            self.logger.info(f"Submitted SLURM job {job_id} for trial {trial_id}")
            
            return job_id
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to submit job for trial {trial_id}: {e}")
            self.logger.error(f"sbatch output: {e.stdout}")
            self.logger.error(f"sbatch error: {e.stderr}")
            
            # Return a dummy job ID to keep track of this trial
            dummy_job_id = f"failed-{uuid.uuid4()}"
            self.active_jobs[dummy_job_id] = {
                "trial_id": trial_id,
                "config": config,
                "submit_time": datetime.now().isoformat(),
                "job_dir": job_dir,
                "status": "FAILED_TO_SUBMIT"
            }
            
            return dummy_job_id

    def _check_job_status(self, job_id):
        """Check the status of a SLURM job.

        Args:
            job_id (str): SLURM job ID

        Returns:
            str: Job status (RUNNING, COMPLETED, FAILED, etc.)
        """
        # Handle dummy jobs for failed submissions
        if job_id.startswith("failed-"):
            return "FAILED"
            
        job_info = self.active_jobs[job_id]
        results_path = os.path.join(job_info["job_dir"], "results.json")

        # First check if results file exists (job completed successfully)
        if os.path.exists(results_path):
            return "COMPLETED"
            
        # Check SLURM job status using squeue
        import subprocess
        
        try:
            # Check if the job is still in the queue
            cmd = ["squeue", "-j", job_id, "-h", "-o", "%T"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # If output is empty, job is no longer in queue (could be done or failed)
            if not result.stdout.strip():
                # Check sacct to get final job state
                cmd = ["sacct", "-j", job_id, "-n", "-o", "State"]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                status = result.stdout.strip().split()[0]
                
                # Map sacct status to our status values
                if status in ["COMPLETED", "COMPLETING"]:
                    # No results file but job completed - might be an error
                    self.logger.warning(f"Job {job_id} completed according to SLURM, but no results file found")
                    return "COMPLETED"
                elif status in ["FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"]:
                    self.logger.warning(f"Job {job_id} failed with SLURM status: {status}")
                    return status
                else:
                    return status
            else:
                # Job is still in queue with status from squeue output
                return result.stdout.strip()
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error checking status of job {job_id}: {e}")
            self.logger.error(f"Command output: {e.stdout}")
            self.logger.error(f"Command error: {e.stderr}")
            
            # If we can't determine status, assume it's still running
            return "RUNNING"

    def _process_completed_job(self, job_id):
        """Process results from a completed job.

        Args:
            job_id (str): SLURM job ID

        Returns:
            bool: True if results were successfully processed, False otherwise
        """
        job_info = self.active_jobs[job_id]
        trial_id = job_info["trial_id"]

        # Check for results file
        results_path = os.path.join(job_info["job_dir"], "results.json")
        if not os.path.exists(results_path):
            self.logger.error(
                f"Results file not found for job {job_id} (trial {trial_id})"
            )
            return False

        # Load results
        try:
            with open(results_path, "r") as f:
                results = json.load(f)

            config = job_info["config"]
            error = results["error"]
            additional_info = results.get("additional_info", None)

            # Update the searcher with the results
            self.update(config, error, additional_info)

            # Copy results to the results directory
            output_path = os.path.join(
                self.output_dir, "results", f"trial_{trial_id}.json"
            )

            # Convert NumPy types to Python native types for JSON serialization
            serializable_config = numpy_to_python(config)
            serializable_additional_info = numpy_to_python(additional_info)

            with open(output_path, "w") as f:
                json.dump(
                    {
                        "trial_id": trial_id,
                        "config": serializable_config,
                        "error": (
                            float(error) if isinstance(error, np.number) else error
                        ),
                        "additional_info": serializable_additional_info,
                        "job_id": job_id,
                        "submit_time": job_info["submit_time"],
                        "complete_time": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            self.logger.info(
                f"Processed results for job {job_id} (trial {trial_id}): error={error}"
            )

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

                if status in ["COMPLETED", "COMPLETING"]:
                    if self._process_completed_job(job_id):
                        jobs_to_remove.append(job_id)

                elif status in ["FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"]:
                    self.logger.warning(f"Job {job_id} failed with status {status}")
                    jobs_to_remove.append(job_id)

            # Submit new jobs if slots are available
            while (
                len(self.active_jobs) < self.max_concurrent_jobs
                and len(self.completed_trials) + len(self.active_jobs) < n_trials
            ):
                config = self.sample_configuration()
                self._submit_job(config, job_script_template)

            # Print status
            elapsed = time.time() - self.start_time
            self.logger.info(
                f"Status: {len(self.completed_trials)}/{n_trials} trials completed, "
                f"{len(self.active_jobs)} jobs active, best error: {self.best_error}, "
                f"elapsed: {elapsed:.1f}s"
            )

            # Sleep before next check (using a shorter interval for local testing)
            time.sleep(5)

        self.logger.info(f"Search completed, best error: {self.best_error}")
        self.logger.info(f"Best config: {self.best_config}")

        return {
            "best_config": self.best_config,
            "best_error": self.best_error,
            "n_trials": len(self.completed_trials),
            "elapsed_time": time.time() - self.start_time,
        }
