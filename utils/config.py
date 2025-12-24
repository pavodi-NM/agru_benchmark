"""
Configuration Utilities Module

This module handles loading and validating the YAML configuration file.

Author: Claude (Anthropic)
"""

import yaml
import os
from typing import Dict, Any


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
    print(f"\nDataset: {config['dataset']['name']}")
    
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
