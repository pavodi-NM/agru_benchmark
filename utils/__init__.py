"""
Utilities package for RNN Benchmark.

Contains training, visualization, and configuration utilities.
"""

from .training import (
    EarlyStopping,
    MetricTracker,
    create_optimizer,
    create_scheduler,
    train_epoch,
    evaluate,
    train_model,
    final_evaluation
)

from .visualization import (
    plot_training_curves,
    plot_accuracy_curves,
    plot_test_comparison,
    plot_multi_run_results,
    create_comparison_table,
    generate_all_plots
)

from .config import (
    load_config,
    validate_config,
    print_config
)

__all__ = [
    # Training
    'EarlyStopping',
    'MetricTracker',
    'create_optimizer',
    'create_scheduler',
    'train_epoch',
    'evaluate',
    'train_model',
    'final_evaluation',
    # Visualization
    'plot_training_curves',
    'plot_accuracy_curves',
    'plot_test_comparison',
    'plot_multi_run_results',
    'create_comparison_table',
    'generate_all_plots',
    # Config
    'load_config',
    'validate_config',
    'print_config'
]
