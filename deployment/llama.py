import time

import torch
import torch.nn as nn

from kvquant.modelutils import *

from kvquant.model_parse import (
    parse_model,
    get_layers,
    get_embedding,
    get_norm,
)

import transformers

import pickle
import json

def get_model(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model)
    model.seqlen = 2048
    return model

# function for benchmarking runtime
def benchmark(model, input_ids, check=False):
    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        past_key_values_length = 0
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i:i+1],
                # past_key_values=cache['past'],
                past_key_values_length_inp=past_key_values_length,
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory,torch.cuda.memory_allocated() / 1024 /1024)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            # cache['past'] = list(out.past_key_values)
            past_key_values_length += 1
            del out
        sync()
        import numpy as np
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):',max_memory)

if __name__ == '__main__':
    import argparse
    from kvquant.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Which dataset to use for benchmarking.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--torch_profile', action='store_true',
        help='Use CUDA profiling tool for timing runs.'
    )

    # arguments for quantization
    parser.add_argument(
        '--quantizer-path', type=str,
        help='Path to quantizers.'
    )
    parser.add_argument(
        '--include_sparse', action='store_true',
        help='Whether to use dense-and-sparse quantization.'
    )
    parser.add_argument(
        '--sparsity-threshold', type=float, default=1,
        help='Outlier percentile.'
    )

    DEV = torch.device('cuda:0')

    args = parser.parse_args()

    model = get_model(args.model)
    model.eval()
    model.model.set_devices()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    layers = model.model.layers
    print('Load quantizers.')
    with open(args.quantizer_path, 'rb') as handle:
        quantizers = pickle.load(handle)

    if args.benchmark:
        # load lookup table + outlier thresholds
        for k in quantizers.keys():
            if '.lut' in k:
                continue
            print('k: ', k)
            ln = int(k.split('.')[-3]) # layer number
            q = quantizers[k]

            if "k_proj" in k:
                layers[ln].self_attn.kcache.load_lookup_table(q, args.include_sparse, args.sparsity_threshold)
            elif "v_proj" in k:
                layers[ln].self_attn.vcache.load_lookup_table(q, args.include_sparse, args.sparsity_threshold)

        model = model.half()
        model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]

            if args.torch_profile:
                from torch.profiler import profile, record_function, ProfilerActivity
                with torch.profiler.profile(
                activities=[
                   torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA,
                ]
                ) as p:
                    benchmark(model, input_ids, check=args.check)
                print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
            else:
                benchmark(model, input_ids, check=args.check)
