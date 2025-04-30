"""
Hyperparameter Optimization (HPO) package for TFA Predictor.

This package provides tools for hyperparameter optimization,
including searchers, schedulers, and utilities for running HPO
campaigns locally or on clusters.
"""

# Base classes
from hpo.base.HPOSearcher import HPOSearcher
from hpo.base.HPOScheduler import HPOScheduler
from hpo.base.HPOTuner import HPOTuner

# Searchers
from hpo.searchers.RandomSearcher import RandomSearcher
from hpo.searchers.AsyncRandomSearch import AsyncRandomSearch

# Schedulers
from hpo.schedulers.BasicScheduler import BasicScheduler
from hpo.schedulers.Scheduler import Scheduler

# Utilities
from hpo.utils.serialization import numpy_to_python

# For backward compatibility
# These aliases ensure that existing imports still work
HPORandomSearcher = RandomSearcher