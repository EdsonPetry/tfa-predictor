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

# Create a temporary patch for AsyncRandomSearch.py to run locally
cat > patch_async_local.py << EOF
import os
import sys

# Path to the AsyncRandomSearch.py file
file_path = 'hpo/AsyncRandomSearch.py'

# Read the content of the file
with open(file_path, 'r') as f:
    content = f.read()

# Replace the _submit_job method to skip SLURM submission
submit_job_override = '''
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
'''

# Replace the _check_job_status method to use local process checking
check_job_status_override = '''
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
'''

# Apply the patches
import re
content = re.sub(r'def _submit_job\(self, config, job_script_template\):.*?return job_id', 
                submit_job_override, content, flags=re.DOTALL)
                
content = re.sub(r'def _check_job_status\(self, job_id\):.*?return "UNKNOWN"', 
                check_job_status_override, content, flags=re.DOTALL)

# Write the modified content back to the file
with open(file_path, 'w') as f:
    f.write(content)

print("Successfully patched AsyncRandomSearch.py for local execution")
EOF

# Apply the patch
python patch_async_local.py

# Run the HPO campaign
echo "Starting local HPO campaign..."
python -u hpo/main_hpo.py --config mlp_config

echo "Local HPO campaign completed at $(date)"

# Clean up the patch script
rm patch_async_local.py