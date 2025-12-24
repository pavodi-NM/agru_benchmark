"""
Training Utilities Module

This module provides the training pipeline, including:
- Training loop with gradient clipping
- Validation evaluation
- Early stopping
- Learning rate scheduling
- Metric tracking

Author: Claude (Anthropic)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to save if best
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            score = -score
            
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            
        return self.early_stop
    
    def load_best_model(self, model: nn.Module):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class MetricTracker:
    """
    Track metrics during training.
    
    Stores loss and other metrics for each epoch, supporting multiple runs.
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'epoch_time': []
        }
        
    def update(
        self,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        epoch_time: float
    ):
        """Add metrics for one epoch."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)
        self.history['epoch_time'].append(epoch_time)
        
    def get_best_epoch(self) -> int:
        """Get the epoch with best validation accuracy."""
        return int(np.argmax(self.history['val_accuracy']))
    
    def get_best_accuracy(self) -> float:
        """Get the best validation accuracy."""
        return max(self.history['val_accuracy'])


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    opt_config = config['optimization']
    opt_type = opt_config['optimizer'].lower()
    lr = opt_config['learning_rate']
    weight_decay = opt_config['weight_decay']
    
    if opt_type == 'adam':
        betas = tuple(opt_config['adam']['betas'])
        eps = opt_config['adam']['eps']
        return optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif opt_type == 'adamw':
        betas = tuple(opt_config['adam']['betas'])
        eps = opt_config['adam']['eps']
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif opt_type == 'sgd':
        momentum = opt_config['sgd']['momentum']
        nesterov = opt_config['sgd']['nesterov']
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif opt_type == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def create_scheduler(
    optimizer: optim.Optimizer, 
    config: Dict
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        
    Returns:
        Scheduler instance or None if disabled
    """
    sched_config = config['training']['lr_scheduler']
    
    if not sched_config['enabled']:
        return None
    
    sched_type = sched_config['type'].lower()
    
    if sched_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config['factor'],
            patience=sched_config['patience'],
            min_lr=sched_config['min_lr']
        )
    elif sched_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 10),
            gamma=sched_config['factor']
        )
    elif sched_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=sched_config['min_lr']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: Dict,
    epoch: int
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        config: Configuration dictionary
        epoch: Current epoch number
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    log_interval = config['logging']['log_interval']
    clip_grad = config['training']['gradient_clipping']
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        if clip_grad['enabled']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                clip_grad['max_norm']
            )
        
        optimizer.step()
        total_loss += loss.item()
        
        # Logging
        if config['logging']['verbose'] and (batch_idx + 1) % log_interval == 0:
            print(f'  Batch [{batch_idx + 1}/{num_batches}] - Loss: {loss.item():.6f}')
    
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        average_loss: Average loss
        accuracy: Classification accuracy
        metrics: Dictionary of additional metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            predictions = output.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Additional metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision_score(all_targets, all_predictions, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_predictions, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    }
    
    return avg_loss, accuracy, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    model_name: str
) -> Tuple[nn.Module, MetricTracker]:
    """
    Full training procedure.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to use
        model_name: Name of the model (for logging)
        
    Returns:
        model: Trained model (best validation accuracy)
        tracker: Metric tracker with training history
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Early stopping
    early_stop_config = config['training']['early_stopping']
    early_stopping = None
    if early_stop_config['enabled']:
        early_stopping = EarlyStopping(
            patience=early_stop_config['patience'],
            min_delta=early_stop_config['min_delta'],
            mode='min' if early_stop_config['monitor'] == 'val_loss' else 'max'
        )
    
    # Training loop
    tracker = MetricTracker()
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )
        
        # Validate
        val_loss, val_accuracy, val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Track metrics
        tracker.update(train_loss, val_loss, val_accuracy, epoch_time)
        
        # Print progress
        print(f'Epoch {epoch:3d}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_accuracy*100:.2f}% | '
              f'Time: {epoch_time:.1f}s')
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_loss, model):
                print(f'\nEarly stopping triggered at epoch {epoch}')
                early_stopping.load_best_model(model)
                break
    
    # Load best model if early stopping was used
    if early_stopping is not None and early_stopping.best_model_state is not None:
        early_stopping.load_best_model(model)
        print(f'Loaded best model from epoch {tracker.get_best_epoch() + 1}')
    
    return model, tracker


def final_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str
) -> Dict:
    """
    Final evaluation on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        model_name: Name of the model
        
    Returns:
        Dictionary with test metrics
    """
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, metrics = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'-'*60}")
    print(f"Test Results for {model_name}:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall: {metrics['recall']*100:.2f}%")
    print(f"  F1 Score: {metrics['f1']*100:.2f}%")
    print(f"{'-'*60}")
    
    return {
        'loss': test_loss,
        **metrics
    }
