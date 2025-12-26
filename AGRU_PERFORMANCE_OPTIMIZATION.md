# A-GRU Performance Optimization Notes

## Performance Tracking

| Phase | Optimizations | Time per Epoch | Speedup vs Baseline | Cumulative Speedup | Status |
|-------|--------------|----------------|---------------------|-------------------|--------|
| Baseline | None | ~35s | 1.0x | 1.0x | ✓ |
| Phase 1 | Cached identity + antisymmetric matrix | ~30s | 1.17x | 1.17x | ✓ Completed |
| Phase 2 | + TorchScript compilation | ~24s | 1.46x | 1.46x | ✓ Completed |
| Phase 3 | Custom CUDA (planned) | ~3-5s (estimated) | ~7-10x | ~7-10x | Pending |

**Current Status**: Phases 1 & 2 completed with **46% total speedup** (35s → 24s per epoch)

**Breakdown**:
- Phase 1 (caching): 35s → 30s (14% improvement, 1.17x)
- Phase 2 (TorchScript): 30s → 24s (20% improvement, 1.25x)
- **Combined**: 35s → 24s (31% reduction in time, 1.46x faster)

## Current Performance Issue

**Baseline**: A-GRU takes ~35s per epoch vs ~3.2s for standard LSTM/GRU (~10x slower)

**Task**: Sequential MNIST (28 timesteps, 128 hidden size, batch size 128)

## Root Cause Analysis

### 1. Python Loop Over Sequence (Most Critical - ~60% of slowdown)

**Location**: `models/agru.py:346-359`

```python
# Current implementation
for t in range(seq_len):  # Sequential processing in Python
    x = input[t]
    for layer, cell in enumerate(self.cells):
        h_t[layer] = cell(x, h_t[layer])
```

**Problem**:
- Processes 28 timesteps sequentially in Python
- Each timestep launches separate CUDA kernels
- Cannot parallelize across sequence dimension
- Significant Python interpreter overhead

**Comparison**:
- Standard LSTM/GRU: Single optimized C++/CUDA kernel processes entire sequence
- A-GRU: 28 separate Python iterations with kernel launches

### 2. Redundant Matrix Operations (Critical - ~30% of slowdown)

**Location**: `models/agru.py:135-159` (called from agru.py:208)

```python
def _get_antisymmetric_matrix(self) -> torch.Tensor:
    # Computed EVERY forward pass (28 timesteps × ~400 batches = 11,200 times/epoch!)
    antisym = self.W_h - self.W_h.t()  # Matrix transpose + subtraction

    # Creates new identity matrix EVERY call
    identity = torch.eye(
        self.hidden_size,
        device=self.W_h.device,
        dtype=self.W_h.dtype
    )

    return antisym - self.gamma * identity
```

**Problem**:
- Identity matrix (128×128 = 16,384 elements) created 11,200 times per epoch
- Matrix transpose computed 11,200 times per epoch
- All operations are on the same static weights (W_h only changes during backprop)

### 3. Lack of Kernel Fusion (~10% of slowdown)

**Problem**:
- Multiple small GPU operations (linear, sigmoid, tanh, element-wise multiply)
- Each operation has memory read/write overhead
- Cannot leverage GPU's parallel execution efficiently

**Comparison**:
- PyTorch's built-in RNNs use fused kernels (especially with cuDNN)
- All operations for a timestep happen in one kernel launch

## Optimization Strategies

### Quick Wins (Easy to implement, significant impact)

#### 1. Cache the Antisymmetric Matrix
**Estimated speedup**: 2-3x

```python
class AGRUCell(nn.Module):
    def __init__(self, ...):
        # ... existing code ...
        self.register_buffer('_cached_antisym', None)
        self.register_buffer('_identity',
            torch.eye(hidden_size) * gamma)

    def _get_antisymmetric_matrix(self) -> torch.Tensor:
        # Only recompute if W_h has changed (during training)
        # For inference, compute once and cache
        if self._cached_antisym is None or self.training:
            antisym = self.W_h - self.W_h.t()
            self._cached_antisym = antisym - self._identity.to(self.W_h.device)
        return self._cached_antisym
```

**Benefits**:
- Identity matrix created once during initialization
- Antisymmetric computation reduced dramatically
- No change to model semantics

**Limitations**:
- Still have Python loop overhead
- Cache invalidation needed during training

#### 2. Pre-compute Identity Matrix
**Estimated speedup**: 1.2-1.5x (if combined with other optimizations)

```python
def __init__(self, ...):
    # Create identity matrix once as a buffer
    self.register_buffer(
        '_gamma_identity',
        torch.eye(hidden_size) * gamma
    )
```

### Medium Effort Optimizations

#### 3. TorchScript Compilation
**Estimated speedup**: 2-4x

```python
# In AGRU class
def __init__(self, ...):
    # ... existing code ...
    # JIT compile the cell for reduced Python overhead
    self.cells = nn.ModuleList([
        torch.jit.script(AGRUCell(...)) for _ in range(num_layers)
    ])
```

**Benefits**:
- Reduces Python interpreter overhead
- May enable some kernel fusion
- Easy to implement (just add torch.jit.script)

**Limitations**:
- Not all Python code is TorchScript compatible
- May need minor code adjustments
- Still sequential timestep processing

#### 4. Vectorize Over Sequence Dimension (Partial)
**Estimated speedup**: 3-5x

**Strategy**:
- Precompute all gate values for entire sequence
- Still iterate but with larger batched operations
- Trade memory for speed

**Challenges**:
- RNN dependencies make full parallelization difficult
- May increase memory usage significantly

### Advanced Optimizations

#### 5. Custom CUDA Kernel
**Estimated speedup**: 5-10x (potentially match LSTM/GRU)

**Approach**:
- Write fused CUDA kernel for A-GRU cell
- Process entire sequence in optimized C++/CUDA
- Similar to PyTorch's built-in LSTM/GRU implementation

**Benefits**:
- Maximum performance (similar to built-in RNNs)
- Full control over memory access patterns
- Kernel fusion for all operations

**Challenges**:
- Requires CUDA programming expertise
- Significant development time
- Need to handle gradients correctly
- Platform-specific code

#### 6. Use torch.fx or torch.compile (PyTorch 2.0+)
**Estimated speedup**: 2-5x

```python
# In AGRU class
def __init__(self, ...):
    # ... existing code ...
    if hasattr(torch, 'compile'):
        self.forward = torch.compile(self.forward)
```

**Benefits**:
- Automatic kernel fusion
- Graph-level optimizations
- Easy to try (one line change)

**Limitations**:
- Requires PyTorch 2.0+
- May not handle dynamic control flow well
- Still limited by sequential dependencies

## Phase 1 Results Analysis

**Actual Results**: 14% speedup (35s → 30s per epoch)

**Why lower than expected?**

The original analysis overestimated the impact of identity matrix creation because:

1. **Python loop overhead dominates**: The 28-iteration Python loop is the main bottleneck (~60-70% of time)
   - Each iteration has Python interpreter overhead, function call overhead, etc.
   - Matrix operations are only part of each iteration's cost

2. **Other operations are significant**: Each timestep also includes:
   - Gate computations (2x sigmoid, linear layers)
   - Element-wise operations (multiply, add)
   - Memory allocation and transfers
   - The antisymmetric computation is only one small part

3. **Cache invalidation**: During training, we invalidate cache every batch
   - This is necessary for correctness (W_h changes during optimization)
   - So we get 28x reuse within a batch, but recompute for each batch

**What we saved**:
- Before: Create 128×128 identity matrix ~11,200 times per epoch
- After: Create once at initialization
- Savings: ~5-7s per epoch (14% of total)

**Conclusion**: Phase 1 optimizations were correctly implemented but the Python loop is the real bottleneck. Need Phase 2 (TorchScript) or Phase 3 (CUDA) for major gains.

## Implementation Priority

### Phase 1: Low-Hanging Fruit ✓ COMPLETED
1. Pre-compute and cache identity matrix (1 hour)
2. Cache antisymmetric matrix computation (2 hours)
3. Profile to verify improvements (1 hour)

**Expected result**: 2-3x speedup (35s → 12-15s per epoch)
**Actual result**: 1.17x speedup (35s → 30s per epoch)

### Phase 2: TorchScript ✓ COMPLETED
1. Make AGRUCell TorchScript compatible (2-4 hours)
2. Add torch.jit.script compilation (1 hour)
3. Test and benchmark (2 hours)

**Expected result**: Additional 1.5-2x speedup on top of Phase 1
**Actual result**: 1.25x additional speedup (30s → 24s per epoch)

**Real-world performance** (Sequential MNIST training):
- Epoch 1: 24.7s (includes data loading overhead)
- Epochs 2-5: ~23.4-24.3s per epoch
- Average: ~24s per epoch

**Analysis**: TorchScript compilation provided significant benefits:
- Reduced Python interpreter overhead for the sequential loop
- Better optimization of repeated cell operations across timesteps
- JIT compiler optimized the hot paths during warmup (first epoch)
- Combined with Phase 1 caching, achieved 1.46x total speedup over baseline

**Conclusion**: Combined Phases 1 & 2 provide **1.46x cumulative speedup** (35s → 24s).
This is a solid improvement for pure Python/PyTorch optimization. Further improvements
require fusing operations into custom CUDA kernels (Phase 3).

### Phase 3: Advanced (If needed)
1. Try torch.compile (PyTorch 2.0+) (1-2 hours)
2. OR implement custom CUDA kernel (40+ hours)

**Expected result**: Potentially match LSTM/GRU performance (~3s per epoch)

## Verification Plan

After each optimization:

1. **Correctness**: Verify outputs match original implementation
   ```python
   # Compare outputs with original implementation
   assert torch.allclose(optimized_output, original_output, rtol=1e-5)
   ```

2. **Performance**: Measure epoch time
   ```python
   import time
   start = time.time()
   # ... training loop ...
   epoch_time = time.time() - start
   ```

3. **Memory**: Check GPU memory usage
   ```python
   print(torch.cuda.max_memory_allocated() / 1e9, "GB")
   ```

## Notes

- Current implementation is **correct** - just not optimized
- PyTorch's built-in LSTM/GRU have years of engineering optimization
- For research/prototyping, current speed is acceptable
- Optimize only if needed for production or large-scale experiments

## References

- PyTorch LSTM source: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/RNN.cpp
- TorchScript docs: https://pytorch.org/docs/stable/jit.html
- Custom CUDA extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html
- torch.compile: https://pytorch.org/docs/stable/generated/torch.compile.html
