import torch
import torch.nn as nn

import quant_cuda
import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from scipy.sparse import random
import numpy as np
import time
import pickle
import json

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking LLaMa-7B FC2 matvec (Ported GPTQ Kernels) ...')

DEV = torch.device('cuda:0')

B = 32 # num heads
M = 128 # head dim
N = 2048 # vcache seqlen

DTYPE = torch.float
matf = torch.randn((B, N, M), device=DEV, dtype=DTYPE)
vec = torch.randn((B, 1, M), device=DEV, dtype=DTYPE)
mul = torch.zeros((B, 1, N), device=DEV, dtype=DTYPE)

matf = matf.transpose(-2,-1).contiguous()

COUNT = 1000

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

cos = torch.randn((B, 1, M), device=DEV, dtype=DTYPE)
sin = torch.randn((B, 1, M), device=DEV, dtype=DTYPE)
vec1 = rotate_half(vec)

from torch.profiler import profile, record_function, ProfilerActivity
with torch.profiler.profile(
activities=[
   torch.profiler.ProfilerActivity.CPU,
   torch.profiler.ProfilerActivity.CUDA,
]
) as p:
    for _ in range(COUNT):
        v = (vec * cos) + (vec1 * sin)
        torch.matmul(v, matf, out=mul)
        torch.cuda.synchronize()

print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
