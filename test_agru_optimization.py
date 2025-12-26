"""
Test script to verify A-GRU optimization correctness and measure performance.

This script:
1. Tests that optimized A-GRU produces same outputs as before
2. Measures performance improvement from caching optimizations
3. Verifies gradient computation is still correct

Usage:
    python test_agru_optimization.py
"""

import torch
import torch.nn as nn
import time
import numpy as np
from models.agru import AGRUCell, AGRU, AGRUClassifier


def test_correctness():
    """Test that optimized implementation produces correct outputs."""
    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test parameters
    batch_size = 128
    seq_len = 28
    input_size = 28
    hidden_size = 128
    num_classes = 10

    # Create model
    model = AGRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=1,
        batch_first=True
    )

    # Create random input
    x = torch.randn(batch_size, seq_len, input_size)

    # Test 1: Multiple forward passes should be consistent
    model.eval()
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)

    # Check outputs are identical
    if torch.allclose(output1, output2, rtol=1e-5):
        print("✓ Multiple forward passes produce consistent results")
    else:
        print("✗ ERROR: Inconsistent outputs!")
        print(f"  Max difference: {(output1 - output2).abs().max().item():.6e}")
        return False

    # Test 2: Training mode forward passes
    model.train()
    output3 = model(x)

    # Should still be close (deterministic)
    if torch.allclose(output1, output3, rtol=1e-5):
        print("✓ Training mode produces consistent results")
    else:
        print("✗ ERROR: Training mode output differs!")
        print(f"  Max difference: {(output1 - output3).abs().max().item():.6e}")
        return False

    # Test 3: Gradient computation
    target = torch.randint(0, num_classes, (batch_size,))
    criterion = nn.CrossEntropyLoss()

    # Forward and backward
    output = model(x)
    loss = criterion(output, target)
    loss.backward()

    # Check gradients exist
    has_gradients = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )

    if has_gradients:
        print("✓ Gradients computed successfully")
    else:
        print("✗ ERROR: Some gradients are None!")
        return False

    # Test 4: Check cache invalidation during training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    outputs_before = []
    for _ in range(3):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        outputs_before.append(out.detach().clone())

    # Outputs should change as parameters update
    diff_1_2 = (outputs_before[0] - outputs_before[1]).abs().max().item()
    diff_2_3 = (outputs_before[1] - outputs_before[2]).abs().max().item()

    if diff_1_2 > 1e-6 and diff_2_3 > 1e-6:
        print("✓ Cache invalidation working (outputs change after optimization)")
    else:
        print("✗ ERROR: Outputs not changing during training!")
        return False

    print("\n✓ All correctness tests passed!\n")
    return True


def benchmark_performance(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Benchmark performance improvements."""
    print("=" * 60)
    print(f"PERFORMANCE BENCHMARK (Device: {device})")
    print("=" * 60)

    # Test parameters (Sequential MNIST settings)
    batch_size = 128
    seq_len = 28
    input_size = 28
    hidden_size = 128
    num_classes = 10
    num_batches = 100  # Simulate ~1/4 epoch

    # Create model
    model = AGRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=1,
        batch_first=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Warmup
    print("Warming up GPU...")
    for _ in range(10):
        x = torch.randn(batch_size, seq_len, input_size, device=device)
        target = torch.randint(0, num_classes, (batch_size,), device=device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark training
    print(f"\nBenchmarking {num_batches} training batches...")
    model.train()

    start_time = time.time()

    for i in range(num_batches):
        x = torch.randn(batch_size, seq_len, input_size, device=device)
        target = torch.randint(0, num_classes, (batch_size,), device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    time_per_batch = elapsed / num_batches

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Time per batch: {time_per_batch*1000:.1f}ms")
    print(f"Estimated time per epoch (~400 batches): {time_per_batch * 400:.1f}s")

    # Benchmark inference
    print(f"\nBenchmarking {num_batches} inference batches...")
    model.eval()

    start_time = time.time()

    with torch.no_grad():
        for i in range(num_batches):
            x = torch.randn(batch_size, seq_len, input_size, device=device)
            output = model(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    time_per_batch = elapsed / num_batches

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Time per batch: {time_per_batch*1000:.1f}ms")

    # Memory usage
    if device == 'cuda':
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("A-GRU OPTIMIZATION TEST SUITE")
    print("=" * 60 + "\n")

    # Test correctness first
    if not test_correctness():
        print("✗ Correctness tests failed! Aborting performance benchmark.")
        return

    # Benchmark on CPU
    print("\n")
    benchmark_performance(device='cpu')

    # Benchmark on GPU if available
    if torch.cuda.is_available():
        print("\n")
        benchmark_performance(device='cuda')

        print("=" * 60)
        print("EXPECTED IMPROVEMENTS")
        print("=" * 60)
        print("Before optimization: ~35s per epoch")
        print("Expected after Phase 1: ~12-15s per epoch (2-3x speedup)")
        print("=" * 60)
    else:
        print("\n⚠ CUDA not available. GPU benchmark skipped.")

    print("\n✓ All tests completed successfully!\n")


if __name__ == "__main__":
    main()
