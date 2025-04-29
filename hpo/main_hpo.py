#!/usr/bin/env python
"""
Main entry point for hyperparameter optimization campaigns on Amarel.
This script runs on the head node and manages the HPO process.
"""

import os
import sys
import json
import argparse
import importlib
import logging
from datetime import datetime

from hpo.AsyncRandomSearch import AsyncRandomSearch, numpy_to_python
from hpo.Scheduler import Scheduler


def setup_logging(output_dir):
    """Set up logging for the HPO campaign.
    
    Args:
        output_dir (str): Directory to save logs
        
    Returns:
        logging.Logger: Configured logger
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    logger = logging.getLogger('hpo_campaign')
    logger.setLevel(logging.INFO)
    
    # Create file handler in logs directory
    log_file = os.path.join(logs_dir, f'hpo_campaign_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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


def run_hpo_campaign(config_module):
    """Run a hyperparameter optimization campaign on Amarel.
    
    Args:
        config_module (str): Name of the config module to use
        
    Returns:
        dict: Results of the HPO campaign
    """
    # Import configuration module
    try:
        config_module = importlib.import_module(f'hpo.config.{config_module}')
    except ImportError:
        raise ImportError(f"Could not import config module: hpo.config.{config_module}")
    
    # Get configurations
    config_space = config_module.get_config_space()
    slurm_config = config_module.get_slurm_config()
    hpo_config = config_module.get_hpo_config()
    
    # Set up output directory
    output_dir = hpo_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting HPO campaign with config: {config_module.__name__}")
    
    # Create searcher
    searcher = AsyncRandomSearch(
        config_space=config_space,
        output_dir=output_dir,
        max_concurrent_jobs=hpo_config['max_concurrent_jobs'],
        checkpoint_interval=hpo_config['checkpoint_interval'],
        seed=hpo_config.get('seed')
    )
    
    # Create scheduler
    scheduler = Scheduler(searcher=searcher, slurm_config=slurm_config)
    
    # Check for existing checkpoint to resume from
    resume_from = None
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    if os.path.exists(checkpoints_dir):
        checkpoints = sorted([
            os.path.join(checkpoints_dir, f)
            for f in os.listdir(checkpoints_dir)
            if f.startswith('checkpoint_') and f.endswith('.pkl')
        ])
        if checkpoints:
            resume_from = checkpoints[-1]
            logger.info(f"Found checkpoint to resume from: {resume_from}")
    
    # Run the campaign
    try:
        results = scheduler.run_hpo_campaign(
            n_trials=hpo_config['n_trials'],
            output_dir=output_dir,
            max_concurrent_jobs=hpo_config['max_concurrent_jobs'],
            resume_from=resume_from
        )
        
        # Save final configuration to a easy-to-read format
        with open(os.path.join(output_dir, 'best_config_formatted.json'), 'w') as f:
            json.dump(numpy_to_python(results['best_config']), f, indent=4)
        
        logger.info("HPO campaign completed successfully")
        logger.info(f"Best configuration: {results['best_config']}")
        logger.info(f"Best error: {results['best_error']}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error running HPO campaign: {e}", exc_info=True)
        raise


def main():
    """Parse arguments and run the HPO campaign."""
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization on Amarel')
    parser.add_argument('--config', type=str, required=True, 
                        help='Configuration module name (without .py extension)')
    args = parser.parse_args()
    
    run_hpo_campaign(args.config)


if __name__ == '__main__':
    main()