"""
Antisymmetric Gated Recurrent Unit (A-GRU) Implementation

This module implements the A-GRU architecture as described in the reference paper.
The key innovation is replacing the standard GRU's candidate hidden state computation
with an antisymmetric formulation that provides stability guarantees.

Mathematical Formulation:
========================

Standard GRU update (rewritten as residual):
    h_s = h_{s-1} + z_s ⊙ (h̃_s - h_{s-1})

A-GRU modification:
    Δh̃_s = f(V_h·x_s + (W_h - W_h^T - γI)(r_s ⊙ h_{s-1}) + b_h)
    h_s = h_{s-1} + z_s ⊙ Δh̃_s
    h_s = h_{s-1} + ε(z_s ⊙ Δh̃_s)

Where:
    - W_h - W_h^T creates an antisymmetric (skew-symmetric) matrix
    - γI is a diffusion term for numerical stability
    - ε is a global step size controlling the dynamics scale
    - z_s is the update gate (local, input-dependent)
    - r_s is the reset gate

Key Properties:
    - Antisymmetric matrices have purely imaginary eigenvalues
    - This prevents exponential growth/decay of hidden states
    - The system behaves like a PDE solver with stable dynamics

Author: Claude (Anthropic)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class AGRUCell(nn.Module):
    """
    Single Antisymmetric GRU Cell.
    
    This implements one layer of the A-GRU, following the formulation:
        Δh̃_s = f(V_h·x_s + (W_h - W_h^T - γI)(r_s ⊙ h_{s-1}) + b_h)
        h_s = h_{s-1} + ε(z_s ⊙ Δh̃_s)
    
    Args:
        input_size: Number of expected features in the input
        hidden_size: Number of features in the hidden state
        gamma: Diffusion coefficient for stability (default: 0.01)
        epsilon: Global step size (default: 1.0)
        learnable_epsilon: Whether epsilon is a learnable parameter
        bias: If False, the layer does not use bias weights
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        gamma: float = 0.01,
        epsilon: float = 1.0,
        learnable_epsilon: bool = True,
        bias: bool = True
    ):
        super(AGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.bias = bias
        
        # =====================================================================
        # Gate Parameters (same structure as standard GRU)
        # =====================================================================
        
        # Update gate (z): controls how much of the new state to use
        # z_s = σ(W_z·x_s + U_z·h_{s-1} + b_z)
        self.W_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Reset gate (r): controls how much of previous state to forget
        # r_s = σ(W_r·x_s + U_r·h_{s-1} + b_r)
        self.W_r = nn.Linear(input_size, hidden_size, bias=bias)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # =====================================================================
        # Candidate State Parameters (MODIFIED for antisymmetry)
        # =====================================================================
        
        # Input-to-hidden for candidate: V_h in the formulation
        self.V_h = nn.Linear(input_size, hidden_size, bias=bias)
        
        # Hidden-to-hidden weight matrix W_h
        # The antisymmetric transformation (W_h - W_h^T) is computed dynamically
        # We store W_h as a parameter and compute the antisymmetric version
        self.W_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        # =====================================================================
        # Epsilon: Global step size
        # =====================================================================
        
        if learnable_epsilon:
            # Make epsilon a learnable parameter, initialized to the given value
            self.epsilon = nn.Parameter(torch.tensor(epsilon))
        else:
            # Register as buffer (saved with model but not updated by optimizer)
            self.register_buffer('epsilon', torch.tensor(epsilon))
        
        self.learnable_epsilon = learnable_epsilon
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.
        
        For the antisymmetric weight matrix, we initialize it such that
        the resulting antisymmetric matrix (W - W^T) has appropriate scale.
        """
        # Standard Xavier initialization for gate weights
        for module in [self.W_z, self.U_z, self.W_r, self.U_r, self.V_h]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Initialize W_h for antisymmetric computation
        # Scale by 1/sqrt(hidden_size) to keep activations stable
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W_h, -stdv, stdv)
    
    def _get_antisymmetric_matrix(self) -> torch.Tensor:
        """
        Compute the antisymmetric (skew-symmetric) recurrent weight matrix.
        
        Returns:
            A = W_h - W_h^T - γI
            
        This matrix has the property that A^T = -A (ignoring the diagonal),
        which ensures purely imaginary eigenvalues and thus stable dynamics.
        
        The -γI term adds a small diffusion for numerical stability,
        helping to dampen any residual instabilities.
        """
        # Compute W_h - W_h^T (antisymmetric part)
        antisym = self.W_h - self.W_h.t()
        
        # Subtract γI for stability (diffusion term)
        # This shifts eigenvalues slightly into the left half-plane
        identity = torch.eye(
            self.hidden_size, 
            device=self.W_h.device, 
            dtype=self.W_h.dtype
        )
        
        return antisym - self.gamma * identity
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for one time step.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h_prev: Previous hidden state of shape (batch_size, hidden_size)
                    If None, initialized to zeros.
        
        Returns:
            h_new: New hidden state of shape (batch_size, hidden_size)
        
        Computation:
            1. Update gate: z_s = σ(W_z·x + U_z·h_{s-1})
            2. Reset gate: r_s = σ(W_r·x + U_r·h_{s-1})
            3. Antisymmetric candidate: Δh̃_s = tanh(V_h·x + A·(r_s ⊙ h_{s-1}))
               where A = W_h - W_h^T - γI
            4. New hidden state: h_s = h_{s-1} + ε·(z_s ⊙ Δh̃_s)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h_prev is None:
            h_prev = torch.zeros(
                batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype
            )
        
        # ---------------------------------------------------------------------
        # Compute gates (same as standard GRU)
        # ---------------------------------------------------------------------
        
        # Update gate: determines how much of new candidate to incorporate
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))
        
        # Reset gate: determines how much of previous state to use in candidate
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))
        
        # ---------------------------------------------------------------------
        # Compute antisymmetric candidate update (DIFFERENT from standard GRU)
        # ---------------------------------------------------------------------
        
        # Get the antisymmetric recurrent matrix
        A = self._get_antisymmetric_matrix()
        
        # Apply reset gate to previous hidden state
        h_reset = r * h_prev  # Element-wise multiplication: (r_s ⊙ h_{s-1})
        
        # Compute the recurrent contribution using antisymmetric matrix
        # This is: A·(r_s ⊙ h_{s-1}) = (W_h - W_h^T - γI)·(r_s ⊙ h_{s-1})
        recurrent = F.linear(h_reset, A)
        
        # Compute candidate update direction
        # Δh̃_s = tanh(V_h·x_s + A·(r_s ⊙ h_{s-1}))
        delta_h_tilde = torch.tanh(self.V_h(x) + recurrent)
        
        # ---------------------------------------------------------------------
        # Compute new hidden state using residual update with epsilon scaling
        # ---------------------------------------------------------------------
        
        # h_s = h_{s-1} + ε·(z_s ⊙ Δh̃_s)
        # The epsilon allows the network to learn the characteristic time scale
        h_new = h_prev + self.epsilon * (z * delta_h_tilde)
        
        return h_new
    
    def extra_repr(self) -> str:
        """String representation of the cell configuration."""
        return (
            f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
            f'gamma={self.gamma}, epsilon={self.epsilon.item():.4f}, '
            f'learnable_epsilon={self.learnable_epsilon}, bias={self.bias}'
        )


class AGRU(nn.Module):
    """
    Multi-layer Antisymmetric GRU.
    
    This module stacks multiple AGRUCell layers to create a deep A-GRU network.
    
    Args:
        input_size: Number of expected features in the input
        hidden_size: Number of features in the hidden state
        num_layers: Number of recurrent layers (default: 1)
        gamma: Diffusion coefficient for stability
        epsilon: Global step size
        learnable_epsilon: Whether epsilon is learnable
        bias: If False, the layer does not use bias weights
        dropout: Dropout probability between layers (if num_layers > 1)
        batch_first: If True, input/output tensors are (batch, seq, feature)
    
    Inputs:
        input: Tensor of shape (seq_len, batch, input_size) or
               (batch, seq_len, input_size) if batch_first=True
        h_0: Initial hidden state of shape (num_layers, batch, hidden_size)
    
    Outputs:
        output: Tensor of shape (seq_len, batch, hidden_size) containing
                the output features from the last layer
        h_n: Final hidden state of shape (num_layers, batch, hidden_size)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        gamma: float = 0.01,
        epsilon: float = 1.0,
        learnable_epsilon: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = False
    ):
        super(AGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Create layers
        self.cells = nn.ModuleList()
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = AGRUCell(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                gamma=gamma,
                epsilon=epsilon,
                learnable_epsilon=learnable_epsilon,
                bias=bias
            )
            self.cells.append(cell)
        
        # Dropout layer (applied between layers, not after last layer)
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(
        self, 
        input: torch.Tensor, 
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all time steps and layers.
        
        Args:
            input: Input tensor
            h_0: Initial hidden states for all layers
        
        Returns:
            output: Output tensor containing hidden states from last layer
            h_n: Final hidden states from all layers
        """
        # Handle batch_first
        if self.batch_first:
            # (batch, seq, feature) -> (seq, batch, feature)
            input = input.transpose(0, 1)
        
        seq_len, batch_size, _ = input.size()
        
        # Initialize hidden states
        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=input.device, dtype=input.dtype
            )
        
        # Current hidden states for each layer
        h_t = [h_0[layer] for layer in range(self.num_layers)]
        
        # Output collection
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            x = input[t]
            
            # Process through each layer
            for layer, cell in enumerate(self.cells):
                h_t[layer] = cell(x, h_t[layer])
                x = h_t[layer]

                # Apply dropout between layers (not after last layer)
                if self.dropout_layer is not None and layer < self.num_layers - 1:
                    x = self.dropout_layer(x)
            
            outputs.append(h_t[-1])
        
        # Stack outputs: (seq_len, batch, hidden_size)
        output = torch.stack(outputs, dim=0)
        
        # Stack final hidden states: (num_layers, batch, hidden_size)
        h_n = torch.stack(h_t, dim=0)
        
        # Handle batch_first for output
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, h_n
    
    def extra_repr(self) -> str:
        """String representation of the module configuration."""
        return (
            f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
            f'num_layers={self.num_layers}, bias={self.bias}, '
            f'batch_first={self.batch_first}, dropout={self.dropout}'
        )


class AGRUClassifier(nn.Module):
    """
    A-GRU based sequence classifier.
    
    This wraps the AGRU with a classification head for sequence classification tasks.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units in A-GRU
        num_classes: Number of output classes
        num_layers: Number of A-GRU layers
        gamma: Diffusion coefficient
        epsilon: Global step size
        learnable_epsilon: Whether epsilon is learnable
        dropout: Dropout probability
        batch_first: If True, input is (batch, seq, feature)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        gamma: float = 0.01,
        epsilon: float = 1.0,
        learnable_epsilon: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True
    ):
        super(AGRUClassifier, self).__init__()
        
        self.agru = AGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            gamma=gamma,
            epsilon=epsilon,
            learnable_epsilon=learnable_epsilon,
            dropout=dropout,
            batch_first=batch_first
        )
        
        # Classification head
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Initialize classification layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Get A-GRU outputs
        output, h_n = self.agru(x)
        
        # Use the last hidden state from the final layer for classification
        # h_n shape: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Classification
        logits = self.fc(last_hidden)
        
        return logits
