import torch
import time

#create sample tensors
a = torch.ones((1000, 1000), dtype=torch.float64, requires_grad=False)
b = torch.full((1000, 1000), 2.0, dtype=torch.float64, requires_grad=False)

# CPU Dot product
start_time = time.time()
ans = torch.matmul(a, b)
end_time = time.time()

time_taken = end_time - start_time

#print("Matrix A[5, 5]: ")
#print(a)

#print("Matrix B[5, 5]: ")
#print(b)

#print(f"Matrix Multiplication Result (a @ b : ")
#print(ans)
print(f"a @ b:  taken : {time_taken:.6f} sec")
