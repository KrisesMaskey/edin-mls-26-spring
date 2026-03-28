# test_cuda.py
import torch

x = torch.randn(10000, 10000, device='cuda')
y = torch.matmul(x, x)

torch.cuda.synchronize()
