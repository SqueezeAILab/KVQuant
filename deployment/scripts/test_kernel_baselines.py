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

print('Benchmarking LLaMa-7B Baseline ...')

DEV = torch.device('cuda:0')

B = 32 # num heads
M = 128 # head dim
N = 2048 #4096 # kcache seqlen

benchmark_K = False # default is benchmark V

DTYPE = torch.float

from torch.profiler import profile, record_function, ProfilerActivity
with torch.profiler.profile(
activities=[
   torch.profiler.ProfilerActivity.CPU,
   torch.profiler.ProfilerActivity.CUDA,
]
) as p:

    if benchmark_K:
        random_tensor = torch.rand((B,N,M), dtype=DTYPE).to(DEV)
        vec = torch.rand((B,M,1)).to(DEV)

        COUNT = 1000
        for _ in range(COUNT):
            mul = torch.matmul(random_tensor, vec).to(DEV)
            torch.cuda.synchronize()
    else:
        random_tensor = torch.rand((B,M,N), dtype=DTYPE).to(DEV) #4096, 2048
        vec = torch.rand((B,N,1)).to(DEV) #32,2048

        COUNT = 1000
        for _ in range(COUNT):
            mul = torch.matmul(random_tensor, vec).to(DEV)
            torch.cuda.synchronize()

print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
