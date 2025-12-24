"""
Standard LSTM and GRU Model Wrappers

This module provides wrapper classes for PyTorch's built-in LSTM and GRU
implementations, ensuring a consistent interface with the custom A-GRU.

Author: Pavodi NDOYI Maniamfu
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMClassifier(nn.Module):
    """
    LSTM-based sequence classifier using PyTorch's built-in LSTM.
    
    This wrapper provides a consistent interface with the A-GRU classifier.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        num_classes: Number of output classes
        num_layers: Number of LSTM layers
        dropout: Dropout probability (applied between layers if num_layers > 1)
        bidirectional: If True, becomes a bidirectional LSTM
        batch_first: If True, input is (batch, seq, feature)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True
    ):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # PyTorch's built-in LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Classification head
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # Initialize classification layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_0: Optional[torch.Tensor] = None,
        c_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            h_0: Initial hidden state (optional)
            c_0: Initial cell state (optional)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Initialize hidden states if not provided
        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype
            )
            c_0 = torch.zeros_like(h_0)
        
        # LSTM forward pass
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # Use the last hidden state for classification
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2]  # Last forward layer
            h_backward = h_n[-1]  # Last backward layer
            last_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Classification
        logits = self.fc(last_hidden)
        
        return logits


class GRUClassifier(nn.Module):
    """
    GRU-based sequence classifier using PyTorch's built-in GRU.
    
    This wrapper provides a consistent interface with the A-GRU classifier.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        num_classes: Number of output classes
        num_layers: Number of GRU layers
        dropout: Dropout probability (applied between layers if num_layers > 1)
        bidirectional: If True, becomes a bidirectional GRU
        batch_first: If True, input is (batch, seq, feature)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True
    ):
        super(GRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # PyTorch's built-in GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Classification head
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize classification layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            h_0: Initial hidden state (optional)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype
            )
        
        # GRU forward pass
        output, h_n = self.gru(x, h_0)
        
        # Use the last hidden state for classification
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2]  # Last forward layer
            h_backward = h_n[-1]  # Last backward layer
            last_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Classification
        logits = self.fc(last_hidden)
        
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, model_name: str) -> str:
    """Get a summary string of the model architecture."""
    total_params = count_parameters(model)
    summary = f"\n{'='*60}\n"
    summary += f"Model: {model_name}\n"
    summary += f"{'='*60}\n"
    summary += f"Total trainable parameters: {total_params:,}\n"
    summary += f"{'-'*60}\n"
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        summary += f"  {name}: {params:,} parameters\n"
    
    summary += f"{'='*60}\n"
    return summary
