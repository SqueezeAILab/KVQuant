import numpy as np
import torch
import time

l1 = 4096
l2 = 4096
tensor11 = torch.randn(4096,4096).cuda()
tensor12 = torch.randn(4096,4096).cuda()
tensor13 = torch.randn(4096,4096).cuda()
tensor2 = torch.randn(4096).cuda()
tensor3 = torch.randn(l2).cuda()

COUNT = 1000

k = 21 # = CEIL(41 / 2), where 41 = CEIL(4096*0.01)

# warmup loop
for i in range(COUNT):
    torch.matmul(tensor11, tensor2)
    torch.cuda.synchronize()

tick = time.time()
for i in range(COUNT):
    a = torch.matmul(tensor11, tensor2)
    b = torch.matmul(tensor12, tensor2)
    c = torch.matmul(tensor13, tensor2)
    torch.cuda.synchronize()
print('baseline:', (time.time() - tick) / COUNT)

#adapting from https://stackoverflow.com/questions/52498690/how-to-use-cuda-stream-in-pytorch
#CUDA stream semantics: https://pytorch.org/docs/stable/notes/cuda.html#:~:text=You%20can%20also%20manually%20wait,work%20on%20the%20default%20stream.)
s1 = torch.cuda.Stream(device="cuda:0")
s2 = torch.cuda.Stream(device="cuda:0")

# frac = (4096*0.99) % 1

tick = time.time()
for i in range(COUNT):
    a = torch.matmul(tensor11, tensor2)
    s2.wait_stream(s1) # needed for correct execution
    with torch.cuda.stream(s1):
        b = torch.matmul(tensor12, tensor2)
        c = torch.matmul(tensor13, tensor2)
    with torch.cuda.stream(s2):
        pinned_memory = tensor3.cpu()
        t1 = torch.topk(pinned_memory,k)
        t2 = torch.topk(pinned_memory,k)

        # add these to check w/ linear interpolation (to match numpy)
        # upper = t1[-2] + (t1[-1] - t1[-2]) * frac
        # lower = t2[-2] + (t2[-1] - t2[-2]) * frac

    torch.cuda.synchronize()
print('offload:', (time.time() - tick) / COUNT)
