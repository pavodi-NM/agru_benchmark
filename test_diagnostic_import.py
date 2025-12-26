"""Quick test to verify the diagnostic script can import and create models."""

import sys
import torch

print("Testing diagnostic script compatibility with optimized A-GRU...")

# Import the diagnostic AGRUClassifier
try:
    from agru_diagnostic import AGRUClassifier
    print("✓ Successfully imported AGRUClassifier from agru_diagnostic.py")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Create a model instance
try:
    model = AGRUClassifier(
        input_size=28,
        hidden_size=128,
        num_classes=10,
        num_layers=1,
        gamma=0.01,
        epsilon=1.0,
        learnable_epsilon=True,
        dropout=0.0,
        use_torchscript=False
    )
    print("✓ Successfully created AGRUClassifier model")
    print(f"  Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test forward pass
try:
    x = torch.randn(8, 28, 28)  # batch=8, seq=28, features=28
    output = model(x)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    assert output.shape == (8, 10), f"Expected shape (8, 10), got {output.shape}"
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test epsilon access
try:
    epsilon_val = model.get_epsilon()
    print(f"✓ Epsilon access successful: ε = {epsilon_val:.4f}")
except Exception as e:
    print(f"✗ Epsilon access error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with TorchScript enabled
try:
    model_ts = AGRUClassifier(
        input_size=28,
        hidden_size=128,
        num_classes=10,
        use_torchscript=True
    )
    print("✓ Successfully created AGRUClassifier with TorchScript enabled")

    output_ts = model_ts(x)
    print(f"✓ TorchScript forward pass successful, output shape: {output_ts.shape}")
except Exception as e:
    print(f"⚠ TorchScript mode warning: {e}")
    print("  (This is optional for diagnostics)")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED - Diagnostic script is ready to use!")
print("="*60)
print("\nYou can now run:")
print("  python agru_diagnostic.py --quick --experiments \"1_Baseline\"")
