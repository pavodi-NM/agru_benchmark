#!/usr/bin/env python3
"""
RNN Architecture Benchmark: LSTM vs GRU vs A-GRU

This script runs a comprehensive benchmark comparing three recurrent neural
network architectures on sequence classification tasks:
1. Standard LSTM (PyTorch implementation)
2. Standard GRU (PyTorch implementation)
3. Antisymmetric GRU (A-GRU, custom implementation)

The A-GRU implements the antisymmetric formulation that views RNN dynamics
as a PDE solver, providing theoretical stability guarantees.

Supported Tasks:
- row_by_row_mnist: Sequential MNIST with 28 timesteps (row-by-row)
- pixel_by_pixel_mnist: Sequential MNIST with 784 timesteps (pixel-by-pixel)

Usage:
    python main.py --task row_by_row_mnist [--runs N] [--quick]
    python main.py --task pixel_by_pixel_mnist
    python main.py --list-tasks

Author: Claude (Anthropic)
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
from typing import Dict, List
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import LSTMClassifier, GRUClassifier, AGRUClassifier, count_parameters, get_model_summary
from data import create_data_loaders, get_dataset_info, print_dataset_info
from utils import (
    load_config, load_task_config, print_config,
    get_available_tasks, get_task_config_path,
    train_model, final_evaluation,
    generate_all_plots
)


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: Dict) -> torch.device:
    """Get the device to use based on configuration."""
    device_config = config['hardware']['device']
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_config)
        print(f"Using device: {device}")
    
    return device


def create_model(model_type: str, config: Dict) -> torch.nn.Module:
    """
    Create a model based on type and configuration.
    
    Args:
        model_type: One of 'lstm', 'gru', 'agru'
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    dataset_config = config['dataset']['sequential_mnist']
    model_config = config['models'][model_type]
    
    input_size = dataset_config['input_size']
    num_classes = dataset_config['num_classes']
    hidden_size = model_config['hidden_size']
    num_layers = model_config['num_layers']
    dropout = model_config['dropout']
    
    if model_type == 'lstm':
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=model_config.get('bidirectional', False),
            batch_first=True
        )
    elif model_type == 'gru':
        model = GRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=model_config.get('bidirectional', False),
            batch_first=True
        )
    elif model_type == 'agru':
        model = AGRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            gamma=model_config.get('gamma', 0.01),
            epsilon=model_config.get('epsilon', 1.0),
            learnable_epsilon=model_config.get('learnable_epsilon', True),
            dropout=dropout,
            batch_first=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def run_single_experiment(
    config: Dict,
    device: torch.device,
    seed: int
) -> Dict:
    """
    Run a single experiment with all enabled models.
    
    Args:
        config: Configuration dictionary
        device: Device to use
        seed: Random seed for this run
        
    Returns:
        Dictionary with training and test results for all models
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config, seed)
    
    results = {
        'training': {},
        'test': {}
    }
    
    # Train and evaluate each enabled model
    model_types = ['lstm', 'gru', 'agru']
    model_names = {'lstm': 'LSTM', 'gru': 'GRU', 'agru': 'A-GRU'}
    
    for model_type in model_types:
        if not config['models'][model_type].get('enabled', False):
            continue
        
        model_name = model_names[model_type]
        print(f"\n{'#'*60}")
        print(f"# {model_name} Model")
        print(f"{'#'*60}")
        
        # Create model
        model = create_model(model_type, config)
        print(get_model_summary(model, model_name))
        
        # Train
        trained_model, tracker = train_model(
            model, train_loader, val_loader, config, device, model_name, model_type
        )
        
        # Final evaluation on test set
        test_metrics = final_evaluation(
            trained_model, test_loader, device, model_name
        )
        
        # Store results
        results['training'][model_name] = {
            'train_loss': tracker.history['train_loss'],
            'val_loss': tracker.history['val_loss'],
            'val_accuracy': tracker.history['val_accuracy'],
            'epoch_time': tracker.history['epoch_time']
        }
        results['test'][model_name] = test_metrics
    
    return results


def run_multiple_experiments(
    config: Dict,
    device: torch.device,
    num_runs: int
) -> Dict:
    """
    Run multiple experiments for statistical significance.
    
    Args:
        config: Configuration dictionary
        device: Device to use
        num_runs: Number of runs to perform
        
    Returns:
        Aggregated results with mean and std
    """
    all_results = {
        'training': {},
        'test': {}
    }
    
    base_seed = config['training']['seed']
    
    for run in range(num_runs):
        print(f"\n{'*'*60}")
        print(f"* RUN {run + 1}/{num_runs}")
        print(f"{'*'*60}")
        
        run_seed = base_seed + run
        results = run_single_experiment(config, device, run_seed)
        
        # Aggregate training results
        for model_name, data in results['training'].items():
            if model_name not in all_results['training']:
                all_results['training'][model_name] = []
            all_results['training'][model_name].append(data)
        
        # Aggregate test results
        for model_name, metrics in results['test'].items():
            if model_name not in all_results['test']:
                all_results['test'][model_name] = []
            all_results['test'][model_name].append(metrics)
    
    # Compute statistics
    aggregated_test = {}
    for model_name, runs in all_results['test'].items():
        aggregated_test[model_name] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [r[metric] for r in runs]
            aggregated_test[model_name][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    all_results['test_aggregated'] = aggregated_test
    
    return all_results


def print_final_summary(results: Dict, multi_run: bool = False):
    """Print a summary of all results."""
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    if multi_run:
        test_results = results['test_aggregated']
        print(f"\n{'Model':<10} {'Accuracy':<20} {'F1 Score':<20}")
        print("-"*50)
        for model_name, metrics in test_results.items():
            acc = f"{metrics['accuracy']['mean']*100:.2f} ± {metrics['accuracy']['std']*100:.2f}%"
            f1 = f"{metrics['f1']['mean']*100:.2f} ± {metrics['f1']['std']*100:.2f}%"
            print(f"{model_name:<10} {acc:<20} {f1:<20}")
    else:
        test_results = results['test']
        print(f"\n{'Model':<10} {'Accuracy':<15} {'F1 Score':<15}")
        print("-"*40)
        for model_name, metrics in test_results.items():
            print(f"{model_name:<10} {metrics['accuracy']*100:.2f}%          {metrics['f1']*100:.2f}%")
    
    print("="*70)


def list_tasks():
    """List all available tasks."""
    tasks = get_available_tasks()
    if not tasks:
        print("No tasks found in config/ directory.")
        print("Create config files (e.g., config/my_task.yaml) to add tasks.")
        return

    print("\nAvailable tasks:")
    print("-" * 40)
    for task in tasks:
        try:
            config_path = get_task_config_path(task)
            # Load config to get description
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            description = config.get('dataset', {}).get('description', 'No description')
            sequence_mode = config.get('dataset', {}).get('sequence_mode', 'N/A')
            print(f"  {task}")
            print(f"    Mode: {sequence_mode}")
            print(f"    Description: {description}")
            print()
        except Exception as e:
            print(f"  {task} (error loading: {e})")
    print("-" * 40)
    print(f"\nUsage: python main.py --task <task_name>")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RNN Architecture Benchmark: LSTM vs GRU vs A-GRU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task row_by_row_mnist
  python main.py --task pixel_by_pixel_mnist --quick
  python main.py --task row_by_row_mnist --runs 5
  python main.py --list-tasks
        """
    )
    parser.add_argument(
        '--task', type=str, default=None,
        help='Task to run (e.g., row_by_row_mnist, pixel_by_pixel_mnist)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to custom configuration file (overrides task default)'
    )
    parser.add_argument(
        '--list-tasks', action='store_true',
        help='List all available tasks'
    )
    parser.add_argument(
        '--runs', type=int, default=None,
        help='Number of runs (overrides config)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: reduce epochs for testing'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results (default: ./results/<task_name>)'
    )

    args = parser.parse_args()

    # Handle --list-tasks
    if args.list_tasks:
        list_tasks()
        return

    # Validate that either --task or --config is provided
    if args.task is None and args.config is None:
        available = get_available_tasks()
        parser.error(
            f"Either --task or --config is required.\n"
            f"Available tasks: {available}\n"
            f"Use --list-tasks to see task descriptions."
        )

    # Load configuration
    print("Loading configuration...")
    if args.config is not None:
        # Custom config file provided
        config = load_config(args.config)
        task_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        # Load from task
        config = load_task_config(args.task)
        task_name = args.task

    print(f"Task: {task_name}")
    
    # Quick mode adjustments
    if args.quick:
        config['training']['num_epochs'] = 5
        config['training']['early_stopping']['patience'] = 3
        print("Quick mode enabled: reduced epochs")
    
    # Print configuration
    print_config(config)
    
    # Print dataset info
    dataset_info = get_dataset_info(config)
    print_dataset_info(dataset_info)
    
    # Get device
    device = get_device(config)
    
    # Determine number of runs
    num_runs = args.runs if args.runs is not None else config['evaluation'].get('num_runs', 1)
    multi_run = num_runs > 1
    
    print(f"\nStarting benchmark with {num_runs} run(s)...")
    start_time = time.time()
    
    # Run experiments
    if multi_run:
        results = run_multiple_experiments(config, device, num_runs)
    else:
        single_results = run_single_experiment(config, device, config['training']['seed'])
        results = {
            'training': single_results['training'],
            'test': single_results['test']
        }
    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time/60:.2f} minutes")
    
    # Print summary
    print_final_summary(results, multi_run)
    
    # Generate plots and save results
    # Default output directory is results/<task_name>
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join('./results', task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if multi_run:
        generate_all_plots(
            results['training'],
            results['test_aggregated'],
            config,
            output_dir,
            multi_run=True
        )
    else:
        generate_all_plots(
            results['training'],
            results['test'],
            config,
            output_dir,
            multi_run=False
        )
    
    print(f"\nResults saved to: {output_dir}")
    print("Benchmark complete!")


if __name__ == '__main__':
    main()
