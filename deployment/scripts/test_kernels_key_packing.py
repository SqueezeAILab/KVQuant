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

from torch.distributions import Normal

#NF support
def get_nf4_signposts(bits=4):
    # for NF4 support
    dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    # get evenly spaced percentile values

    num_signposts_pos = (2 ** (bits - 1)) + 1 # for pos half
    num_signposts_neg = (2 ** (bits - 1)) # for neg half
    num_spaces_pos = (2 ** (bits - 1)) + 1 # for pos half
    num_spaces_neg = (2 ** (bits - 1)) # for neg half

    nf4_signposts_negative = []
    nf4_signposts_positive = []

    # from https://arxiv.org/pdf/2306.06965.pdf
    offsets = [0.5*(1/32 + 1/30), 1 - 0.5*(1/32 + 1/30)]
    list1 = [offsets[0]]
    spacing = (0.5 - offsets[0]) / (2 ** (bits - 1) - 1)

    add = offsets[0]
    for i in range(num_signposts_neg - 1):
        add += spacing
        list1.append(add)

    list2 = []
    spacing = (offsets[1] - 0.5) / (2 ** (bits - 1)) #1 extra space
    add = 0.5
    for i in range(num_signposts_pos - 1):
        list2.append(add)
        add += spacing
    list2.append(offsets[-1])

    # first do negative part [0->0.5]
    for i in range(num_signposts_neg):
        v1 = list1[i]
        val = dist.icdf(torch.tensor([v1])).data.numpy()
        nf4_signposts_negative.append(torch.tensor(val).item())

    # next do positive part [0.5->1]
    for i in range(num_signposts_pos):
        v1 = list2[i]
        val = dist.icdf(torch.tensor([v1])).data.numpy()
        nf4_signposts_positive.append(torch.tensor(val).item())

    signpost_neg_min = nf4_signposts_negative[0]
    signpost_neg_max = nf4_signposts_negative[-1]
    rangeval = abs(signpost_neg_min)-abs(signpost_neg_max)
    off = abs(signpost_neg_max)
    for s in range(len(nf4_signposts_negative)):
        nf4_signposts_negative[s] = (nf4_signposts_negative[s] + off) / rangeval

    signpost_pos_min = nf4_signposts_positive[0]
    signpost_pos_max = nf4_signposts_positive[-1]
    rangeval = abs(signpost_pos_max)-abs(signpost_pos_min)
    off = abs(signpost_pos_min)

    for s in range(len(nf4_signposts_positive)):
        nf4_signposts_positive[s] = (nf4_signposts_positive[s] - off) / rangeval

    del nf4_signposts_positive[0]

    #TODO delete last negative value and merge
    nf4_signposts = nf4_signposts_negative + nf4_signposts_positive

    assert (len(nf4_signposts) == (2 ** bits))
    return nf4_signposts

print('Benchmarking LLaMa-7B FC2 matvec (Ported GPTQ Kernels) ...')

DEV = torch.device('cuda:0')

B = 32 # num heads
M = 128 # head dim
num_heads = 32
head_dim = 128

N = 2048 #16384 #4096 # vcache seqlen

num_iters = 1000

with open(f'activations.pickle', 'rb') as handle:
    activations = pickle.load(handle)

with open(f'quantizers.pickle', 'rb') as handle:
    quantizers = pickle.load(handle)

from torch.profiler import profile, record_function, ProfilerActivity
with torch.profiler.profile(
activities=[
   torch.profiler.ProfilerActivity.CPU,
   torch.profiler.ProfilerActivity.CUDA,
]
) as p:

    for l in range(0,32):
        k_act = activations[f'self_attn.k_proj.{l}']
        v_act = activations[f'self_attn.v_proj.{l}']

        quantizer = quantizers[f'model.layers.{l}.self_attn.k_proj']

        ### Q*KT operation
        num_heads = 32
        head_dim = 128
        max_len = N
        q = torch.zeros((1,num_heads,head_dim), dtype=torch.float).cuda()

        maxval = torch.tensor(quantizer[0]).cuda().half().squeeze(0)
        minval = torch.tensor(quantizer[1]).cuda().half().squeeze(0)
        outlier_threshold_lower = minval.float()
        outlier_threshold_upper = maxval.float()
        offset = (maxval + minval) / 2
        rangeval = (maxval - minval) / 2

        kcache2 = torch.zeros((num_heads, head_dim // 8, max_len), dtype=torch.int).cuda()
        lookup_table = torch.zeros((num_heads, head_dim, 2 ** 4))

        # kernel is indep. of values used for signposts (note that activations are collected using the real values)
        nf4_signposts = get_nf4_signposts(4)

        for i in range(num_heads):
            for j in range(head_dim):
                idx = i * head_dim + j
                sf_tmp = rangeval[idx]
                offset_tmp = offset[idx]
                lookup_table[i,j] = torch.tensor(nf4_signposts) * sf_tmp.item() + offset_tmp.item()

        lookup_table = lookup_table.cuda()

        # initialize zeropoint
        zeropoint = (maxval + minval) / 2
        zeropoint = zeropoint.float().cuda()

        rows2 = torch.tensor([]).cuda()
        cols2 = torch.tensor([]).cuda()
        vals2 = torch.tensor([]).cuda()
        start_rows = torch.tensor([]).cuda()

        # code for fw pass
        for i in range(0,num_iters):
            newk = k_act[i]

            rows2, cols2, vals2, start_rows, num_threads, outlier_count = quant_cuda.vecquant4appendvecKsparse(
                kcache2,
                lookup_table,
                newk,
                zeropoint,
                rows2,
                cols2,
                vals2,
                start_rows,
                outlier_threshold_lower,
                outlier_threshold_upper,
                i
            )
            num_threads = num_threads[0]
            num_nonzeros = vals2.shape[0]

print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
