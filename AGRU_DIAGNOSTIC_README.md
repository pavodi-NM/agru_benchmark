# A-GRU Diagnostic Experiments

## Overview

This script systematically investigates why the Antisymmetric GRU (A-GRU) exhibits a larger training-validation loss gap compared to standard LSTM and GRU on Sequential MNIST.

## Observed Problem

From your benchmark results:

| Model | Accuracy | Val Loss Behavior |
|-------|----------|-------------------|
| LSTM  | 98.75%   | Converges close to training loss |
| GRU   | 98.91%   | Converges close to training loss |
| A-GRU | 98.07%   | **Plateaus ~0.09, large gap from training** |

The A-GRU shows clear signs of overfitting despite achieving reasonable accuracy.

## Hypotheses Tested

### 1. Learnable Epsilon (ε) Hypothesis
**Theory**: The learnable global step size ε can grow during training to make larger updates that overfit the training data.

**Experiment**: `2_Frozen_Epsilon` - Set `learnable_epsilon=False`

### 2. Insufficient Damping (γ) Hypothesis
**Theory**: The diffusion term `-γI` with γ=0.01 provides minimal damping, allowing hidden states to oscillate freely and memorize training patterns.

**Experiments**: 
- `3a_Gamma_0.1` - Increase γ to 0.1
- `3b_Gamma_0.5` - Increase γ to 0.5

### 3. Explicit Regularization Hypothesis
**Theory**: A-GRU's antisymmetric structure provides stability but not regularization - explicit dropout/weight decay may be needed.

**Experiments**:
- `4a_Dropout_0.2` - Add 20% dropout
- `4b_Dropout_0.3` - Add 30% dropout
- `6a_WeightDecay_0.001` - Increase weight decay 10x
- `6b_WeightDecay_0.01` - Increase weight decay 100x

### 4. Model Capacity Hypothesis
**Theory**: The full-rank antisymmetric matrix `(W - W^T)` creates higher effective capacity than standard GRU.

**Experiment**: `5_Hidden_64` - Reduce hidden size to 64

### 5. Epsilon Scale Hypothesis
**Theory**: Initial ε=1.0 may be too large, causing aggressive updates early in training.

**Experiments**:
- `8_Small_Epsilon_0.1` - Start with ε=0.1
- `9_Frozen_SmallEps_HighGamma` - Frozen ε=0.5 with γ=0.1

### 6. Combined Regularization
**Experiment**: `7_Combined_Regularization` - Apply multiple fixes together:
- γ=0.1
- learnable_epsilon=False  
- dropout=0.2
- weight_decay=0.001

## Usage

### Run All Experiments (Full)
```bash
python agru_diagnostic.py --output-dir ./diagnostic_results
```

### Quick Mode (10 epochs each, for testing)
```bash
python agru_diagnostic.py --quick --output-dir ./diagnostic_results
```

### Run Specific Experiments
```bash
python agru_diagnostic.py --experiments "1_Baseline" "2_Frozen_Epsilon" "7_Combined_Regularization"
```

## Output

The script generates:

1. **`training_curves_comparison.png`** - All experiments on same plot
2. **`gap_and_accuracy_comparison.png`** - Bar charts comparing gaps and accuracies
3. **`individual_*.png`** - Detailed plots for each experiment
4. **`diagnostic_report.md`** - Markdown report with:
   - Results table
   - Hypothesis evaluation (✅/❌)
   - Recommended configuration

## Experiment Configurations

| Name | γ | ε | Learnable ε | Dropout | Weight Decay | Hidden |
|------|---|---|-------------|---------|--------------|--------|
| 1_Baseline | 0.01 | 1.0 | Yes | 0.0 | 0.0001 | 128 |
| 2_Frozen_Epsilon | 0.01 | 1.0 | **No** | 0.0 | 0.0001 | 128 |
| 3a_Gamma_0.1 | **0.1** | 1.0 | Yes | 0.0 | 0.0001 | 128 |
| 3b_Gamma_0.5 | **0.5** | 1.0 | Yes | 0.0 | 0.0001 | 128 |
| 4a_Dropout_0.2 | 0.01 | 1.0 | Yes | **0.2** | 0.0001 | 128 |
| 4b_Dropout_0.3 | 0.01 | 1.0 | Yes | **0.3** | 0.0001 | 128 |
| 5_Hidden_64 | 0.01 | 1.0 | Yes | 0.0 | 0.0001 | **64** |
| 6a_WeightDecay_0.001 | 0.01 | 1.0 | Yes | 0.0 | **0.001** | 128 |
| 6b_WeightDecay_0.01 | 0.01 | 1.0 | Yes | 0.0 | **0.01** | 128 |
| 7_Combined | 0.1 | 1.0 | No | 0.2 | 0.001 | 128 |
| 8_Small_Epsilon | 0.01 | **0.1** | Yes | 0.0 | 0.0001 | 128 |
| 9_Frozen_SmallEps | 0.1 | 0.5 | No | 0.0 | 0.0001 | 128 |

## Expected Insights

After running, you should be able to identify:

1. **Primary cause** of the train-val gap
2. **Best configuration** for A-GRU that minimizes overfitting
3. **Trade-offs** between gap reduction and test accuracy
4. **Whether A-GRU can match** LSTM/GRU performance with proper tuning

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
```

## Code Structure

The script uses:
- `ExperimentConfig` dataclass for configuration
- **Optimized A-GRU** from `models/agru.py` with:
  - Phase 1 optimizations: Pre-computed identity matrix and antisymmetric caching
  - Phase 2 optimizations: TorchScript compilation support (disabled by default for diagnostics)
  - ~24s per epoch performance (vs ~35s baseline)
- Sequential MNIST data loading
- Training loop with early stopping
- Comprehensive visualization
- Automated report generation

**Note**: TorchScript compilation is disabled in diagnostic experiments for clearer profiling and debugging. The focus is on identifying overfitting causes, not maximizing speed.

## Interpreting Results

**Good sign**: Gap reduces significantly while maintaining accuracy
**Warning sign**: Gap reduces but accuracy drops substantially
**Key metric**: `train_val_gap` - lower is better (indicates less overfitting)

The best configuration should:
1. Have the smallest train-val gap
2. Maintain test accuracy ≥97%
3. Show stable, smooth convergence
