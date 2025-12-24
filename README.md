# RNN Architecture Benchmark: LSTM vs GRU vs A-GRU

A comprehensive benchmarking study comparing three recurrent neural network architectures on the Sequential MNIST dataset.

## Overview

This project implements and compares:
1. **Standard LSTM** - PyTorch's built-in implementation
2. **Standard GRU** - PyTorch's built-in implementation  
3. **Antisymmetric GRU (A-GRU)** - Custom implementation based on the PDE solver formulation

### A-GRU Mathematical Formulation

The A-GRU modifies the standard GRU by viewing it as a PDE solver with antisymmetric dynamics:

```
Δh̃_s = f(V_h·x_s + (W_h - W_h^T - γI)(r_s ⊙ h_{s-1}) + b_h)
h_s = h_{s-1} + ε(z_s ⊙ Δh̃_s)
```

Where:
- `W_h - W_h^T` creates an antisymmetric (skew-symmetric) matrix with purely imaginary eigenvalues
- `γI` is a diffusion term for numerical stability
- `ε` is a learnable global step size
- `z_s` is the update gate (local, input-dependent)
- `r_s` is the reset gate

## Project Structure

```
rnn_benchmark/
├── config.yaml              # Single YAML configuration file
├── main.py                  # Main benchmark script
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── models/
│   ├── __init__.py
│   ├── agru.py             # A-GRU implementation (from scratch)
│   └── standard_models.py   # LSTM and GRU wrappers
├── data/
│   ├── __init__.py
│   └── dataset.py          # Sequential MNIST dataset handling
├── utils/
│   ├── __init__.py
│   ├── config.py           # Configuration loading
│   ├── training.py         # Training pipeline
│   └── visualization.py    # Plotting utilities
└── results/                 # Output directory (created automatically)
```

## Dataset

**Sequential MNIST** was chosen as the benchmark dataset because:
- Well-established and widely used in RNN literature
- Challenging for modeling long-term dependencies (28 time steps)
- Clear classification task (10 digits)
- Readily available through PyTorch

Each 28×28 image is treated as a sequence of 28 rows, where each row (28 pixels) is one time step.

| Property | Value |
|----------|-------|
| Sequence Length | 28 |
| Input Size | 28 |
| Number of Classes | 10 |
| Training Samples | ~51,000 |
| Validation Samples | ~9,000 |
| Test Samples | 10,000 |

## Installation

```bash
# Clone or extract the project
cd rnn_benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- matplotlib
- scikit-learn
- PyYAML

## Usage

### Basic Run (Single Experiment)

```bash
python main.py --config config.yaml
```

### Multiple Runs for Statistical Significance

```bash
python main.py --config config.yaml --runs 5
```

### Quick Test (Reduced Epochs)

```bash
python main.py --config config.yaml --quick
```

### Custom Output Directory

```bash
python main.py --config config.yaml --output-dir ./my_results
```

## Configuration

All hyperparameters are controlled through `config.yaml`. Key settings:

```yaml
# Training
training:
  num_epochs: 50
  batch_size: 128
  seed: 42

# Optimization
optimization:
  optimizer: adam
  learning_rate: 0.001

# Models
models:
  lstm:
    enabled: true
    hidden_size: 128
  gru:
    enabled: true
    hidden_size: 128
  agru:
    enabled: true
    hidden_size: 128
    gamma: 0.01      # Diffusion coefficient
    epsilon: 1.0     # Global step size
```

## Output

The benchmark generates:

1. **Training Curves** (`training_curves.png`) - Loss over epochs for all models
2. **Accuracy Curves** (`accuracy_curves.png`) - Validation accuracy over epochs
3. **Test Comparison** (`test_comparison.png`) - Bar chart comparing final metrics
4. **Results Summary** (`results_summary.md`) - Markdown table with all metrics

## Metrics

- **Accuracy** - Classification accuracy
- **Precision** - Macro-averaged precision
- **Recall** - Macro-averaged recall
- **F1 Score** - Macro-averaged F1

## Key Implementation Details

### A-GRU (`models/agru.py`)

The A-GRU implementation:
1. Uses standard GRU gating (update and reset gates)
2. Replaces the candidate hidden state with antisymmetric formulation
3. Enforces antisymmetry via `W_h - W_h^T`
4. Includes learnable global step size `ε`
5. Adds diffusion term `γI` for stability

### Training Pipeline

- Unified training loop for all models
- Same data loaders, loss function, and optimizer
- Early stopping with patience
- Gradient clipping for stability
- Learning rate scheduling

## Expected Results

Typical results on Sequential MNIST (may vary with random seeds):

| Model | Test Accuracy | Parameters |
|-------|---------------|------------|
| LSTM  | ~98%          | ~132K      |
| GRU   | ~98%          | ~99K       |
| A-GRU | ~97-98%       | ~83K       |

## License

MIT License

## References

The A-GRU formulation is based on viewing GRU dynamics as a PDE solver with stability enforced by antisymmetric weight matrices. Work conducted by Pavodi Maniamfu as a Ph.D. student at the University of Tsukuba, Japan.
