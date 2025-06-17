import torch
import time

# Example dimensions: A (4x3), B (3x4)
m, k, n = 1000, 1000, 1000

# Initialize matrices with sample data
A = torch.arange(1, m * k + 1, dtype=torch.float64).reshape(m, k)
B = torch.arange(1, k * n + 1, dtype=torch.float64).reshape(k, n)

# Measure execution time
start_time = time.perf_counter()
C = torch.matmul(A, B)
end_time = time.perf_counter()
duration_ms = (end_time - start_time) * 1000

# Print result
# print("Result matrix C:")
# print(C)
print(f"PyTorch matrix multiplication took {duration_ms:.3f} ms")
