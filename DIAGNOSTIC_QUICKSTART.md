# A-GRU Diagnostic Experiments - Quick Start

## ✓ Updated for Optimized A-GRU

The diagnostic script now uses the **optimized A-GRU implementation** with:
- ✅ Phase 1: Pre-computed identity matrix and antisymmetric caching
- ✅ Phase 2: TorchScript compilation support
- ✅ ~24s per epoch performance (vs ~35s baseline)

**Note**: TorchScript is disabled by default in diagnostics for clearer profiling. The focus is on identifying overfitting causes.

## Verification

Run the verification test:
```bash
python test_diagnostic_import.py
```

Expected output:
```
✓ Successfully imported AGRUClassifier from agru_diagnostic.py
✓ Successfully created AGRUClassifier model
  Model has 61,579 parameters
✓ Forward pass successful, output shape: torch.Size([8, 10])
✓ Epsilon access successful: ε = 1.0000
✓ ALL TESTS PASSED
```

## Running Experiments

### Quick Test (Single Experiment, 10 epochs)
```bash
python agru_diagnostic.py --quick --experiments "1_Baseline"
```

This will:
- Train A-GRU with baseline configuration for 10 epochs
- Take ~4-5 minutes on GPU
- Generate plots and report in `./diagnostic_results/`

### Full Diagnostic Suite (All Experiments, 50 epochs each)
```bash
python agru_diagnostic.py --output-dir ./diagnostic_results
```

This runs all 10 experiments:
1. Baseline (γ=0.01, learnable ε)
2. Frozen Epsilon (learnable_epsilon=False)
3. Higher Gamma (γ=0.1, 0.5)
4. Dropout (0.2, 0.3)
5. Reduced Hidden Size (64)
6. Increased Weight Decay (0.001, 0.01)
7. Combined Regularization
8. Small Initial Epsilon (0.1)
9. Frozen Small ε + High γ

**Time estimate**: ~2-3 hours for all experiments on GPU

### Recommended: Run Selected Experiments
```bash
python agru_diagnostic.py --quick --experiments \
  "1_Baseline" \
  "2_Frozen_Epsilon" \
  "3a_Gamma_0.1" \
  "4a_Dropout_0.2" \
  "7_Combined_Regularization"
```

This runs the most important experiments in ~20-25 minutes.

## Output Files

In `./diagnostic_results/`:
- `training_curves_comparison.png` - All experiments on same plot
- `gap_and_accuracy_comparison.png` - Bar charts
- `individual_*.png` - Detailed plots per experiment
- `diagnostic_report.md` - **Analysis with recommendations**

## Key Metrics to Watch

In the diagnostic report, look for:

1. **Train-Val Gap**: Lower is better (indicates less overfitting)
   - Baseline A-GRU typically shows gap ~0.05-0.10
   - LSTM/GRU typically show gap ~0.01-0.03

2. **Test Accuracy**: Should maintain ≥97%
   - Baseline A-GRU: ~98%
   - LSTM/GRU: ~98.8%

3. **Epsilon Evolution**: For learnable ε experiments
   - Check if ε grows excessively during training
   - Large ε values (>2.0) may indicate instability

## Understanding Results

The diagnostic report will evaluate each hypothesis:

✅ **SUPPORTED** - This modification reduces the gap
❌ **NOT SUPPORTED** - No significant improvement

Best configuration will be recommended based on:
- Smallest train-val gap
- Maintained high test accuracy (≥97%)
- Stable convergence

## Performance Notes

With the optimized model:
- Each epoch: ~24s (vs ~35s unoptimized)
- 10 epochs per experiment: ~4 minutes
- Full suite (10 experiments × 50 epochs): ~2-3 hours

The optimization makes it practical to run comprehensive diagnostic experiments!

## Example: Interpreting Results

If you see:
```
| Experiment                  | Test Acc | Gap    |
|-----------------------------|----------|--------|
| 1_Baseline                  | 98.07%   | 0.0823 |
| 2_Frozen_Epsilon            | 98.12%   | 0.0654 | ← Better!
| 7_Combined_Regularization   | 97.89%   | 0.0412 | ← Best gap, slight acc drop
```

**Interpretation**:
- Freezing epsilon helps reduce overfitting (gap: 0.082 → 0.065)
- Combined regularization achieves best gap but trades some accuracy
- **Recommendation**: Use frozen epsilon (good balance)

## Next Steps

1. ✅ Verify setup: `python test_diagnostic_import.py`
2. 🧪 Run quick test: `python agru_diagnostic.py --quick --experiments "1_Baseline"`
3. 📊 Check results in `./diagnostic_results/`
4. 🔬 Run full suite or selected experiments
5. 📝 Read `diagnostic_report.md` for recommendations
6. ⚙️ Update `config.yaml` with recommended A-GRU settings

## Troubleshooting

**Issue**: Import error
**Fix**: Make sure you're in `/home/manisan/rnn_benchmark/` directory

**Issue**: CUDA out of memory
**Fix**: Reduce batch size in `ExperimentConfig` or use `--quick` mode

**Issue**: TorchScript compilation messages during run
**Note**: These are informational. TorchScript is disabled by default for diagnostics.

## Questions?

- See `AGRU_DIAGNOSTIC_README.md` for detailed hypothesis explanations
- See `AGRU_PERFORMANCE_OPTIMIZATION.md` for optimization details
- Run verification test if you encounter issues
