"""
Configuration Utilities Module

This module handles loading and validating the YAML configuration file.
Supports task-based configuration with configs in the config/ directory.

Author: Claude (Anthropic)
"""

import yaml
import os
from typing import Dict, Any, List, Optional

# Default config directory relative to project root
CONFIG_DIR = "config"


def get_project_root() -> str:
    """Get the project root directory (parent of config/)."""
    # Assuming this file is in utils/, go up one level
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_config_dir() -> str:
    """Get the config directory path."""
    return os.path.join(get_project_root(), CONFIG_DIR)


def get_available_tasks() -> List[str]:
    """
    Get list of available tasks from config directory.

    Returns:
        List of task names (config file names without .yaml extension)
    """
    config_dir = get_config_dir()
    if not os.path.exists(config_dir):
        return []

    tasks = []
    for filename in os.listdir(config_dir):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            # Remove extension to get task name
            task_name = os.path.splitext(filename)[0]
            tasks.append(task_name)

    return sorted(tasks)


def get_task_config_path(task_name: str) -> str:
    """
    Get the config file path for a given task.

    Args:
        task_name: Name of the task (e.g., "row_by_row_mnist")

    Returns:
        Full path to the task's config file

    Raises:
        FileNotFoundError: If task config doesn't exist
    """
    config_dir = get_config_dir()

    # Try .yaml first, then .yml
    for ext in ['.yaml', '.yml']:
        config_path = os.path.join(config_dir, f"{task_name}{ext}")
        if os.path.exists(config_path):
            return config_path

    available = get_available_tasks()
    raise FileNotFoundError(
        f"No config file found for task '{task_name}'. "
        f"Available tasks: {available}"
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config


def load_task_config(task_name: str) -> Dict[str, Any]:
    """
    Load configuration for a specific task.

    Args:
        task_name: Name of the task (e.g., "row_by_row_mnist")

    Returns:
        Configuration dictionary
    """
    config_path = get_task_config_path(task_name)
    return load_config(config_path)


def validate_config(config: Dict[str, Any]):
    """
    Validate the configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['dataset', 'training', 'optimization', 'models', 'evaluation']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate dataset config
    if 'name' not in config['dataset']:
        raise ValueError("Dataset name must be specified")

    # Validate sequence_mode if present
    sequence_mode = config['dataset'].get('sequence_mode')
    if sequence_mode is not None:
        valid_modes = ['row_by_row', 'pixel_by_pixel']
        if sequence_mode not in valid_modes:
            raise ValueError(
                f"Invalid sequence_mode: {sequence_mode}. "
                f"Must be one of {valid_modes}"
            )

    # Validate training config
    training = config['training']
    if training['num_epochs'] <= 0:
        raise ValueError("Number of epochs must be positive")
    if training['batch_size'] <= 0:
        raise ValueError("Batch size must be positive")

    # Validate model configs
    models = config['models']
    enabled_models = []
    for model_type in ['lstm', 'gru', 'agru']:
        if model_type in models and models[model_type].get('enabled', False):
            enabled_models.append(model_type)

    if len(enabled_models) == 0:
        raise ValueError("At least one model must be enabled")

    print(f"Configuration validated. Enabled models: {enabled_models}")


def print_config(config: Dict[str, Any]):
    """Print configuration in a formatted way."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)

    # Dataset
    dataset = config['dataset']
    print(f"\nDataset: {dataset['name']}")
    if 'sequence_mode' in dataset:
        print(f"  Sequence Mode: {dataset['sequence_mode']}")
    
    # Training
    training = config['training']
    print(f"\nTraining:")
    print(f"  Epochs: {training['num_epochs']}")
    print(f"  Batch Size: {training['batch_size']}")
    print(f"  Random Seed: {training['seed']}")
    
    # Optimization
    opt = config['optimization']
    print(f"\nOptimization:")
    print(f"  Optimizer: {opt['optimizer']}")
    print(f"  Learning Rate: {opt['learning_rate']}")
    print(f"  Weight Decay: {opt['weight_decay']}")
    
    # Models
    print(f"\nModels:")
    models = config['models']
    for model_type in ['lstm', 'gru', 'agru']:
        if model_type in models:
            m = models[model_type]
            status = "enabled" if m.get('enabled', False) else "disabled"
            print(f"  {model_type.upper()}: {status}")
            if m.get('enabled', False):
                print(f"    Hidden Size: {m.get('hidden_size', 'default')}")
                print(f"    Num Layers: {m.get('num_layers', 1)}")
    
    print("="*60 + "\n")
