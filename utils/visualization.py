"""
Visualization Utilities Module

This module provides functions for creating publication-quality plots
comparing model performance.

Author: Claude (Anthropic)
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
from typing import Dict, List, Optional
import os


def setup_plot_style(config: Dict):
    """Setup matplotlib style based on configuration."""
    style_config = config['visualization']['style']
    
    plt.rcParams['figure.figsize'] = style_config['figure_size']
    plt.rcParams['font.size'] = style_config['font_size']
    plt.rcParams['axes.grid'] = style_config['use_grid']
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['figure.dpi'] = config['visualization']['dpi']
    plt.rcParams['savefig.dpi'] = config['visualization']['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['font.family'] = 'sans-serif'


def plot_training_curves(
    results: Dict[str, Dict],
    save_path: str,
    config: Dict
):
    """
    Plot training and validation loss curves for all models.
    
    Args:
        results: Dictionary mapping model names to their training results
        save_path: Path to save the plot
        config: Configuration dictionary
    """
    setup_plot_style(config)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    # Training loss
    ax1 = axes[0]
    for idx, (model_name, data) in enumerate(results.items()):
        epochs = range(1, len(data['train_loss']) + 1)
        ax1.plot(epochs, data['train_loss'], label=model_name, 
                color=colors[idx], linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_xlim(1, None)
    
    # Validation loss
    ax2 = axes[1]
    for idx, (model_name, data) in enumerate(results.items()):
        epochs = range(1, len(data['val_loss']) + 1)
        ax2.plot(epochs, data['val_loss'], label=model_name,
                color=colors[idx], linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_xlim(1, None)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved training curves to: {save_path}")


def plot_accuracy_curves(
    results: Dict[str, Dict],
    save_path: str,
    config: Dict
):
    """
    Plot validation accuracy curves for all models.
    
    Args:
        results: Dictionary mapping model names to their training results
        save_path: Path to save the plot
        config: Configuration dictionary
    """
    setup_plot_style(config)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for idx, (model_name, data) in enumerate(results.items()):
        epochs = range(1, len(data['val_accuracy']) + 1)
        accuracy_percent = [acc * 100 for acc in data['val_accuracy']]
        ax.plot(epochs, accuracy_percent, label=model_name,
               color=colors[idx], linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(1, None)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved accuracy curves to: {save_path}")


def plot_individual_model_curves(
    results: Dict[str, Dict],
    save_path_prefix: str,
    config: Dict
):
    """
    Plot training vs validation curves for each individual model.

    Creates separate plots for each model showing:
    - Training loss vs Validation loss
    - Training accuracy vs Validation accuracy (if available)

    This helps identify overfitting in individual models.

    Args:
        results: Dictionary mapping model names to their training results
        save_path_prefix: Path prefix to save the plots (without extension)
        config: Configuration dictionary
    """
    setup_plot_style(config)
    plot_format = config['visualization']['plot_format']

    for model_name, data in results.items():
        # Create a figure with 2 subplots (loss and accuracy)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(data['train_loss']) + 1)

        # Plot 1: Training vs Validation Loss
        ax1 = axes[0]
        ax1.plot(epochs, data['train_loss'], label='Training Loss',
                color='#2E86AB', linewidth=2, marker='o', markersize=4, alpha=0.8)
        ax1.plot(epochs, data['val_loss'], label='Validation Loss',
                color='#A23B72', linewidth=2, marker='s', markersize=4, alpha=0.8)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.set_xlim(1, None)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Validation Accuracy (and training if available)
        ax2 = axes[1]
        if 'train_accuracy' in data:
            train_acc_percent = [acc * 100 for acc in data['train_accuracy']]
            ax2.plot(epochs, train_acc_percent, label='Training Accuracy',
                    color='#2E86AB', linewidth=2, marker='o', markersize=4, alpha=0.8)

        val_acc_percent = [acc * 100 for acc in data['val_accuracy']]
        ax2.plot(epochs, val_acc_percent, label='Validation Accuracy',
                color='#A23B72', linewidth=2, marker='s', markersize=4, alpha=0.8)

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.set_xlim(1, None)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        model_filename = model_name.lower().replace('-', '_').replace(' ', '_')
        save_path = f"{save_path_prefix}_{model_filename}.{plot_format}"
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {model_name} individual curves to: {save_path}")


def plot_test_comparison(
    test_results: Dict[str, Dict],
    save_path: str,
    config: Dict,
    multi_run: bool = False
):
    """
    Plot bar chart comparing test performance of all models.
    
    Args:
        test_results: Dictionary mapping model names to test metrics
        save_path: Path to save the plot
        config: Configuration dictionary
        multi_run: If True, include error bars from multiple runs
    """
    setup_plot_style(config)
    
    model_names = list(test_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if multi_run:
            values = [test_results[m][metric]['mean'] * 100 for m in model_names]
            errors = [test_results[m][metric]['std'] * 100 for m in model_names]
            bars = ax.bar(x + i * width, values, width, label=label, 
                         color=colors[i], yerr=errors, capsize=3)
        else:
            values = [test_results[m][metric] * 100 for m in model_names]
            bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Test Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved test comparison to: {save_path}")


def plot_multi_run_results(
    all_results: Dict[str, List[Dict]],
    save_path: str,
    config: Dict
):
    """
    Plot results from multiple runs with confidence intervals.
    
    Args:
        all_results: Dictionary mapping model names to list of run results
        save_path: Path to save the plot
        config: Configuration dictionary
    """
    setup_plot_style(config)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))
    
    # Validation loss with confidence bands
    ax1 = axes[0]
    for idx, (model_name, runs) in enumerate(all_results.items()):
        # Get all validation losses (keep as list, don't convert to array yet)
        all_val_losses = [r['val_loss'] for r in runs]

        # Handle different lengths by padding with NaN
        max_len = max(len(vl) for vl in all_val_losses)
        padded = np.full((len(all_val_losses), max_len), np.nan)
        for i, vl in enumerate(all_val_losses):
            padded[i, :len(vl)] = vl
        
        mean_loss = np.nanmean(padded, axis=0)
        std_loss = np.nanstd(padded, axis=0)
        epochs = range(1, len(mean_loss) + 1)
        
        ax1.plot(epochs, mean_loss, label=model_name, color=colors[idx], linewidth=2)
        ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                        color=colors[idx], alpha=0.2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Validation Loss (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Validation accuracy with confidence bands
    ax2 = axes[1]
    for idx, (model_name, runs) in enumerate(all_results.items()):
        all_val_acc = [r['val_accuracy'] for r in runs]

        max_len = max(len(va) for va in all_val_acc)
        padded = np.full((len(all_val_acc), max_len), np.nan)
        for i, va in enumerate(all_val_acc):
            padded[i, :len(va)] = va
        
        mean_acc = np.nanmean(padded, axis=0) * 100
        std_acc = np.nanstd(padded, axis=0) * 100
        epochs = range(1, len(mean_acc) + 1)
        
        ax2.plot(epochs, mean_acc, label=model_name, color=colors[idx], linewidth=2)
        ax2.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                        color=colors[idx], alpha=0.2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy (Mean ± Std)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved multi-run results to: {save_path}")


def create_comparison_table(
    test_results: Dict[str, Dict],
    training_results: Dict[str, Dict],
    save_path: str,
    multi_run: bool = False
):
    """
    Create a markdown table comparing all models.
    
    Args:
        test_results: Dictionary mapping model names to test metrics
        training_results: Dictionary mapping model names to training results
        save_path: Path to save the table
        multi_run: If True, include mean ± std
    """
    with open(save_path, 'w') as f:
        f.write("# RNN Architecture Comparison Results\n\n")
        f.write("## Test Performance Summary\n\n")
        
        if multi_run:
            f.write("| Model | Accuracy | Precision | Recall | F1 Score | Avg Epoch Time |\n")
            f.write("|-------|----------|-----------|--------|----------|----------------|\n")
            
            for model_name, results in test_results.items():
                acc = f"{results['accuracy']['mean']*100:.2f} ± {results['accuracy']['std']*100:.2f}"
                prec = f"{results['precision']['mean']*100:.2f} ± {results['precision']['std']*100:.2f}"
                rec = f"{results['recall']['mean']*100:.2f} ± {results['recall']['std']*100:.2f}"
                f1 = f"{results['f1']['mean']*100:.2f} ± {results['f1']['std']*100:.2f}"
                
                if model_name in training_results:
                    avg_time = np.mean([np.mean(r['epoch_time']) for r in training_results[model_name]])
                    time_str = f"{avg_time:.2f}s"
                else:
                    time_str = "N/A"
                
                f.write(f"| {model_name} | {acc}% | {prec}% | {rec}% | {f1}% | {time_str} |\n")
        else:
            f.write("| Model | Accuracy | Precision | Recall | F1 Score | Epochs | Best Val Acc |\n")
            f.write("|-------|----------|-----------|--------|----------|--------|-------------|\n")
            
            for model_name, results in test_results.items():
                acc = f"{results['accuracy']*100:.2f}"
                prec = f"{results['precision']*100:.2f}"
                rec = f"{results['recall']*100:.2f}"
                f1 = f"{results['f1']*100:.2f}"
                
                if model_name in training_results:
                    epochs = len(training_results[model_name]['train_loss'])
                    best_val = max(training_results[model_name]['val_accuracy']) * 100
                else:
                    epochs = "N/A"
                    best_val = "N/A"
                
                f.write(f"| {model_name} | {acc}% | {prec}% | {rec}% | {f1}% | {epochs} | {best_val:.2f}% |\n")
        
        f.write("\n")
    
    print(f"Saved comparison table to: {save_path}")


def generate_all_plots(
    training_results: Dict[str, Dict],
    test_results: Dict[str, Dict],
    config: Dict,
    output_dir: str,
    multi_run: bool = False
):
    """
    Generate all visualization plots.
    
    Args:
        training_results: Training results for each model
        test_results: Test results for each model
        config: Configuration dictionary
        output_dir: Directory to save plots
        multi_run: If True, treat results as from multiple runs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_format = config['visualization']['plot_format']
    
    if multi_run:
        plot_multi_run_results(
            training_results,
            os.path.join(output_dir, f'training_curves.{plot_format}'),
            config
        )
    else:
        plot_training_curves(
            training_results,
            os.path.join(output_dir, f'training_curves.{plot_format}'),
            config
        )
    
    if not multi_run:
        plot_accuracy_curves(
            training_results,
            os.path.join(output_dir, f'accuracy_curves.{plot_format}'),
            config
        )

        # Plot individual model curves (training vs validation)
        plot_individual_model_curves(
            training_results,
            os.path.join(output_dir, 'individual_model'),
            config
        )

    plot_test_comparison(
        test_results,
        os.path.join(output_dir, f'test_comparison.{plot_format}'),
        config,
        multi_run=multi_run
    )
    
    create_comparison_table(
        test_results,
        training_results,
        os.path.join(output_dir, 'results_summary.md'),
        multi_run=multi_run
    )
