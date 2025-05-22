# \\[\\[ GPU Matrix-Multiplication Benchmark (Mac Studio M2) \\]\\]
import os
import time
import torch

# \\[\\[ Ensure MPS fallback for non-GPU ops \\]\\]
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# \\[\\[ Build large matrices on CPU \\]\\]
SIZE = 16192# Adjust up/down for your memory budget
a_cpu = torch.randn(SIZE, SIZE, device="cpu")
b_cpu = torch.randn(SIZE, SIZE, device="cpu")

# \\[\\[ CPU timing \\]\\]
start_cpu = time.perf_counter()
c_cpu = torch.mm(a_cpu, b_cpu)
cpu_time = time.perf_counter() - start_cpu

# \\[\\[ Transfer to GPU \\]\\]
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)
if device.type == "mps":
    torch.mps.synchronize()

# \\[\\[ GPU timing \\]\\]
start_gpu = time.perf_counter()
c_gpu = torch.mm(a_gpu, b_gpu)
if device.type == "mps":
    torch.mps.synchronize()
gpu_time = time.perf_counter() - start_gpu

# \\[\\[ GPU memory footprint \\]\\]
bytes_used = (a_gpu.element_size() * a_gpu.nelement() +
              b_gpu.element_size() * b_gpu.nelement() +
              c_gpu.element_size() * c_gpu.nelement())
print("CPU matmul time: {:.3f} s".format(cpu_time))
print("GPU matmul time: {:.3f} s".format(gpu_time))
print("GPU memory used: {:.2f} MiB".format(bytes_used / (1024**2)))