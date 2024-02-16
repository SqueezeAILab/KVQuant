import time

import torch
import torch.nn as nn

from kvquant.modelutils import *
from kvquant.datautils import *
from kvquant.simquant_module_quantizer import *

from kvquant.model_parse import (
    parse_model,
    get_layers,
    get_embedding,
    get_norm,
)

import transformers

import pickle
import json

def get_model_longseqlen(model, seqlen, maxseqlen):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # Set RoPE scaling factor
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model)
    context_size = maxseqlen
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, config=config, trust_remote_code=True)

    model.seqlen = seqlen  #TODO
    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)

    return model

#function to find layers in the network (either for packing or for replacement)
def find_layers2(module, layers=[nn.Conv2d, nn.Linear, QuantLinearSim], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def get_activations(model, testenc, dev):
    print('Starting ...')

    model_type = parse_model(model)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers(model, model_type)
    embeddings = get_embedding(model, model_type)
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache.get('position_ids')

    print('Measuring runtime ...')
    activations = {}

    for i in range(len(layers)):
        print("Layer", i)
        layer = layers[i].to(dev)
        full = find_layers2(layer)

        perchannel_list = ['self_attn.k_proj']
        pertensor_list = ['self_attn.v_proj']
        full_list = ['self_attn.k_proj', 'self_attn.v_proj']

        sequential = list(full.keys())

        simquant = {}
        subset = {n: full[n] for n in sequential if n in full_list}
        sequential_subset = list(subset.keys())
        for name in sequential:
            if name in perchannel_list:
                print('perchannel name: ', name)
                simquant[name] = SimQuant(
                                        subset[name],
                                        4,
                                        perchannel=True,
                                        qchannel=0
                                     )
            elif name in pertensor_list:
                print('pertensor name: ', name)
                simquant[name] = SimQuant(
                                        subset[name],
                                        4,
                                        perchannel=True, # currently assuming per-token outliers
                                        qchannel=-1
                                     )
            else:
                continue

        def add_batch(name):
            def tmp(_, inp, out):
                simquant[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []

        for name in sequential_subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

        for h in handles:
            h.remove()

        data = simquant['self_attn.k_proj'].out.float().squeeze(0) # cached K act
        activations[f'self_attn.k_proj.{i}'] = data
        data = simquant['self_attn.v_proj'].out.float().squeeze(0) # cached V act
        activations[f'self_attn.v_proj.{i}'] = data

        for name in subset:
            simquant[name].free()

        del simquant
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model_type = parse_model(model)
    norm = get_norm(model, model_type)
    if norm is not None:
        norm = norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    model.config.use_cache = use_cache

    return activations


if __name__ == '__main__':
    import argparse
    from kvquant.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--torch_profile', action='store_true',
        help='Use CUDA profiling tool for timing runs.'
    )

    parser.add_argument(
        '--nsamples', type=int, default=1,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--seqlen', type=int, default=2048,
        help='Sequence length for calibration / eval.'
    )
    parser.add_argument(
        '--maxseqlen', type=int, default=2048,
        help='Sequence length for calibration / eval.'
    )

    #load quantizers for benchmarking
    parser.add_argument(
        '--quantizer-path', type=str, default='',
        help='Path to quantizers.'
    )

    parser.add_argument(
        '--output-path', type=str, default='',
        help='Path to output pickle.'
    )

    DEV = torch.device('cuda:0')

    args = parser.parse_args()

    model = get_model_longseqlen(args.model, args.seqlen, args.maxseqlen)
    model = model.half()
    model.eval()

    dataloader, testloader = get_loaders(
        "wikitext2", nsamples=args.nsamples, seed=0, model=args.model, seqlen=model.seqlen
    )

    with open(args.quantizer_path, 'rb') as handle:
        quantizers = pickle.load(handle)

    # replace layers
    perchannelquant_perchanneloutliers_q = {}
    pertokenquant_perchanneloutliers_q = {}

    for k in quantizers.keys():
        quantizers[k] = quantizers[k] + (-1, ) # empty for now (used to be LN params)
        if "k_proj" in k:
            perchannelquant_perchanneloutliers_q[k] = quantizers[k]
        if "v_proj" in k:
            pertokenquant_perchanneloutliers_q[k] = quantizers[k]

    #per-vector quant
    make_quant_sim(
        model,
        perchannelquant_perchanneloutliers_q,
        4,
        perchannel=True,
        perchanneloutliers=True,
        zeropoint=True,
        include_sparse=True,
        sparsity_threshold=0.99,
        dynamicquantization=False,
        dynamicoutliers=False,
        nuq=True
    )

    #per-vector quant
    make_quant_sim(
        model,
        pertokenquant_perchanneloutliers_q,
        4,
        perchannel=False,
        perchanneloutliers=True,
        zeropoint=True,
        include_sparse=True,
        sparsity_threshold=0.99,
        dynamicquantization=True,
        dynamicoutliers=True,
        nuq=True
    )

    act_to_store = get_activations(
        model,
        testloader,
        DEV
    )

    with open(args.output_path, 'wb') as handle:
        pickle.dump(act_to_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
