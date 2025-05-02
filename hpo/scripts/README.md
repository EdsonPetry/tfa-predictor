# HPO Analysis Scripts

This directory contains scripts for analyzing hyperparameter optimization (HPO) results.

## analyze_results.py

The original analysis script for hyperparameter optimization campaigns. It loads trial results and generates visualizations.

Usage:
```bash
python analyze_results.py --results_dir hpo_results/experiment_name --output_dir hpo_results/experiment_name/analysis
```

## analyze_results_optimized.py

An optimized version of the analysis script that can handle large numbers of trials more efficiently. It includes features for:

- Parallel processing of trial files
- Batch processing to avoid memory issues
- Selective loading of trials
- Performance optimizations for plotting
- Additional statistics and error handling

Usage:
```bash
# Basic usage
python analyze_results_optimized.py --results_dir hpo_results/experiment_name --output_dir hpo_results/experiment_name/analysis

# Limit to processing 50 trials for faster analysis
python analyze_results_optimized.py --results_dir hpo_results/experiment_name --limit 50

# Only analyze the top 30 performing trials
python analyze_results_optimized.py --results_dir hpo_results/experiment_name --top_k 30

# Generate plots without displaying them (useful for headless environments)
python analyze_results_optimized.py --results_dir hpo_results/experiment_name --no_show

# Show detailed progress information
python analyze_results_optimized.py --results_dir hpo_results/experiment_name --verbose

# Apply smoothing to trajectory plot
python analyze_results_optimized.py --results_dir hpo_results/experiment_name --smooth
```

For large experiments (>100 trials), it's recommended to use the optimized script with the `--limit` or `--top_k` option to improve performance.