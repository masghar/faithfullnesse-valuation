import torch
import time

# Matrix size (Adjust 5000 to 10000 for even more "pressure")
size = 5000
iterations = 28

def benchmark_device(device_name):
    device = torch.device(device_name)
    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm-up (Important for GPU JIT compilation)
    for _ in range(3):
        _ = torch.matmul(a, b)
    if device_name == "mps": torch.mps.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(a, b)
        if device_name == "mps": torch.mps.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations

print(f"Benchmarking Matrix Multiplication ({size}x{size})...\n")

# Run CPU
cpu_time = benchmark_device("cpu")
print(f"CPU average time: {cpu_time:.4f} seconds")

# Run MPS (GPU)
mps_time = benchmark_device("mps")
print(f"MPS average time: {mps_time:.4f} seconds")

# Calculate Speedup
speedup = cpu_time / mps_time
print(f"\nResult: The M5 GPU is {speedup:.2f}x faster than the CPU for this task.")