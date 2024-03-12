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

N = 2048 #4096 # vcache seqlen
max_len = N

num_iters = 1000

with open(f'activations.pickle', 'rb') as handle:
    activations = pickle.load(handle)

with open(f'quantizers.pickle', 'rb') as handle:
    quantizers = pickle.load(handle)

d = {}
for l in range(0,32):
    k_act = activations[f'self_attn.k_proj.{l}']
    v_act = activations[f'self_attn.v_proj.{l}']

    quantizer = quantizers[f'model.layers.{l}.self_attn.v_proj']

    ### Score * V ###
    hidden_size = 4096
    threshold_k = int(((1-0.99) / 2) * hidden_size) + 1

    lookup_table2 = torch.zeros((N, 2 ** 4)).cuda()
    rows2 = torch.tensor([]).cuda()
    cols2 = torch.tensor([]).cuda()
    vals2 = torch.tensor([]).cuda()
    start_cols = torch.tensor([]).cuda()
    vcache2 = torch.zeros((num_heads, head_dim // 8, max_len), dtype=torch.int).cuda()

    nf4_signposts = get_nf4_signposts(4)

    for i in range(0,N):
        newv = v_act[i].float()

        sorted_v,_ = newv.sort()
        minval = sorted_v[threshold_k]
        maxval = sorted_v[-threshold_k]
        offset = (maxval + minval) / 2
        sf = (maxval - minval) / 2

        lookup_table = torch.tensor(nf4_signposts).float() * sf.float().item() + offset.float().item()
        lookup_table2[i] = lookup_table
        zeropoint = lookup_table[7]

        # can just use upper / lower when using NF for benchmarking
        # otherwise we need to use minval / maxval obtained from sorting
        outlier_threshold_lower = lookup_table[0]
        outlier_threshold_upper = lookup_table[-1]

        rows2, cols2, vals2, start_cols, num_threads, outlier_count = quant_cuda.vecquant4appendvecVsparseorig(vcache2, lookup_table2, newv, zeropoint, rows2, cols2, vals2, start_cols, outlier_threshold_lower, outlier_threshold_upper, i)
        num_threads = num_threads[0]
        num_nonzeros = vals2.shape[0]

    #D+S matvec operation
    score = torch.zeros((1, num_heads, N), dtype=torch.float).cuda()
    mul = torch.zeros((1, num_heads, head_dim), dtype=torch.float).cuda()

    d[f'layer{l} score'] = score
    d[f'layer{l} vcache2'] = vcache2
    d[f'layer{l} mul'] = mul
    d[f'layer{l} lookup_table2'] = lookup_table2
    d[f'layer{l} N'] = N
    d[f'layer{l} rows2'] = rows2
    d[f'layer{l} cols2'] = cols2
    d[f'layer{l} start_cols'] = start_cols
    d[f'layer{l} vals2'] = vals2
    d[f'layer{l} num_threads'] = num_threads
    d[f'layer{l} num_nonzeros'] = num_nonzeros

# warmup
j = 0
for i in range(0,num_iters):
    quant_cuda.vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig(
        d[f'layer{j} score'],
        d[f'layer{j} vcache2'],
        d[f'layer{j} mul'],
        d[f'layer{j} lookup_table2'],
        d[f'layer{j} N'],
        d[f'layer{j} rows2'],
        d[f'layer{j} cols2'],
        d[f'layer{j} start_cols'],
        d[f'layer{j} vals2'],
        d[f'layer{j} N'],
        d[f'layer{j} num_threads'],
        d[f'layer{j} num_nonzeros']
    )


from torch.profiler import profile, record_function, ProfilerActivity
with torch.profiler.profile(
activities=[
   torch.profiler.ProfilerActivity.CPU,
   torch.profiler.ProfilerActivity.CUDA,
]
) as p:
    for j in range(0,32):
        for i in range(0,num_iters):
            quant_cuda.vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig(
                d[f'layer{j} score'],
                d[f'layer{j} vcache2'],
                d[f'layer{j} mul'],
                d[f'layer{j} lookup_table2'],
                d[f'layer{j} N'],
                d[f'layer{j} rows2'],
                d[f'layer{j} cols2'],
                d[f'layer{j} start_cols'],
                d[f'layer{j} vals2'],
                d[f'layer{j} N'],
                d[f'layer{j} num_threads'],
                d[f'layer{j} num_nonzeros']
            )
print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
