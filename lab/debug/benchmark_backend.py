"""
Benchmark test for fit_uni backend performance: numba vs pytorch (with CUDA if available)
"""
import numpy as np
import time
import torch
from unilasso.uni_lasso import fit_uni

def benchmark(dataset_name, X, y, n_runs=3):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}, X shape: {X.shape}, y shape: {y.shape}")
    print(f"{'='*60}")

    # Test numba backend
    numba_times = []
    for i in range(n_runs):
        start = time.time()
        _ = fit_uni(X, y, family="gaussian", backend="numba", n_lmdas=100, verbose=False)
        end = time.time()
        numba_times.append(end - start)
        print(f"Numba run {i+1}: {numba_times[-1]:.2f}s")

    numba_avg = np.mean(numba_times)
    numba_std = np.std(numba_times)
    print(f"\nNumba average: {numba_avg:.2f}s ± {numba_std:.2f}s")

    # Test pytorch backend
    pytorch_times = []
    for i in range(n_runs):
        start = time.time()
        _ = fit_uni(X, y, family="gaussian", backend="pytorch", n_lmdas=100, verbose=False)
        end = time.time()
        pytorch_times.append(end - start)
        print(f"PyTorch run {i+1}: {pytorch_times[-1]:.2f}s")

    pytorch_avg = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)
    print(f"\nPyTorch average: {pytorch_avg:.2f}s ± {pytorch_std:.2f}s")

    # Compare
    speedup = numba_avg / pytorch_avg if pytorch_avg > 0 else float('inf')
    if speedup > 1:
        print(f"\n✅ PyTorch is {speedup:.2f}x faster than numba")
    else:
        print(f"\n✅ Numba is {1/speedup:.2f}x faster than PyTorch")

    return numba_avg, pytorch_avg

if __name__ == "__main__":
    # Check CUDA availability
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Test datasets
    datasets = [
        ("Small (n=1000, p=100)", 1000, 100),
        ("Medium (n=2000, p=500)", 2000, 500),
        ("Large (n=5000, p=1000)", 5000, 1000),
    ]

    results = []
    for name, n, p in datasets:
        # Generate random data
        np.random.seed(42)
        X = np.random.randn(n, p)
        y = X @ np.random.randn(p) + np.random.randn(n) * 0.1

        numba_t, pytorch_t = benchmark(name, X, y, n_runs=3)
        results.append((name, numba_t, pytorch_t))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<30} {'Numba (s)':<12} {'PyTorch (s)':<12} {'Speedup':<8}")
    print("-"*60)
    for name, numba_t, pytorch_t in results:
        speedup = numba_t / pytorch_t if pytorch_t > 0 else 0
        print(f"{name:<30} {numba_t:<12.2f} {pytorch_t:<12.2f} {speedup:<8.2f}x")
