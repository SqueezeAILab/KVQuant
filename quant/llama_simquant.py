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

import pickle
import json

import math
import argparse

def get_model(model, seqlen, maxseqlen):
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
    model = AutoModelForCausalLM.from_pretrained(model, config=config, trust_remote_code=True, use_flash_attention_2=True, torch_dtype=torch.half)

    model.seqlen = seqlen  #TODO
    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)
    return model

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')
    model_type = parse_model(model)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
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
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print("Layer", i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            if model_type == 'opt':
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                )[0]
            else:
                assert model_type == 'llama'
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    norm = get_norm(model, model_type)
    if norm is not None:
        norm = norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def llama_calibration(model, dataloader, dev, perchannel_match, pertensor_match, bits, include_sparse=False, sparsity_threshold=0.999, nuq=False, fisher=None, norm=False):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Quantizing ...')

    quantizers = {}
    for i in range(len(layers)):
        print("Layer", i)
        layer = layers[i].to(dev)
        full = find_layers(layer)

        perchannel_list = []
        pertensor_list = []
        full_list = []

        for f in full:
            for p in perchannel_match:
                if p in f:
                    perchannel_list.append(f)
                    full_list.append(f)
            for p in pertensor_match:
                if p in f:
                    pertensor_list.append(f)
                    full_list.append(f)

        sequential = list(full.keys())

        simquant = {}
        subset = {n: full[n] for n in sequential if n in full_list}
        sequential_subset = list(subset.keys())
        for name in sequential:
            if name in perchannel_list:
                simquant[name] = SimQuant(
                                        subset[name],
                                        bits,
                                        perchannel=True,
                                        qchannel=0
                                     )
            elif name in pertensor_list:
                simquant[name] = SimQuant(
                                        subset[name],
                                        bits,
                                        perchannel=True,
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

        for name in subset:

            if fisher is not None:
                key = 'model.layers.%d.%s' % (i, name)
                key = key + '.weight'
                fisher_info = fisher[key].cpu()
            else:
                fisher_info = None

            quantizers['model.layers.%d.%s' % (i, name)] = simquant[name].quantize(
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                nuq=nuq,
                fisher=fisher_info,
                norm=norm
            )
            simquant[name].free()

        layers[i] = layer.cpu()
        del layer
        del simquant
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0,
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=16,
        help='Number of calibration data samples.'
    )

    #args for quantizers
    parser.add_argument(
        '--quantize', action='store_true',
        help='Whether to run calibration to quantize the KV cache.'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[2, 3, 4, 5, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--nuq', action='store_true',
        help='Whether to use non-uniform quantization.'
    )
    parser.add_argument(
        '--nf', action='store_true',
        help='Whether to use NormalFloat-based non-uniform quantization.'
    )

    #detailed quantization parameters
    parser.add_argument(
        '--perchannel', type=json.loads, default=["k_proj"],
        help='Tensors to use channel-wise quant.'
    )
    parser.add_argument(
        '--pertoken', type=json.loads, default=["v_proj"],
        help='Tensors to use token-wise quant.'
    )
    parser.add_argument(
        '--include_sparse', action='store_true',
        help='Whether to use dense-and-sparse quantization.'
    )
    parser.add_argument(
        '--sparsity-threshold', type=float, default=1,
        help='Outlier percentile.'
    )
    parser.add_argument(
        '--norm', action='store_true',
        help='Whether to use q-norm.'
    )
    parser.add_argument(
        '--quantizer_path', type=str, default=None,
        help='Path to load/store quantizer file'
    )

    # calibration parameters
    parser.add_argument(
        '--fisher', type=str, default=None,
        help='fisher information path'
    )
    parser.add_argument(
        '--seqlen', type=int, default=-1,
        help='Sequence length for calibration / eval.'
    )
    parser.add_argument(
        '--maxseqlen', type=int, default=2048,
        help='Maximum sequence length for the model.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'c4'], default='wikitext2',
        help='Which dataset to use for calibration / evaluation.'
    )

    DEV = torch.device('cuda:0')

    args = parser.parse_args()

    #load model
    print('Loading model ...')
    if args.load:
        model = get_model(args.model, args.seqlen, args.maxseqlen)
        model.load_state_dict(torch.load(args.load))
        model.eval()
    else:
        model = get_model(args.model, args.seqlen, args.maxseqlen)
        model.eval()

    if args.seqlen != -1:
        model.seqlen = args.seqlen

    model = model.half()
    print('Done.')

    # TODO: once multi-device evaluation framework is set up, at set_devices call here

    #load dataloaders
    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    if args.quantize:

        # run quantization here

        if args.fisher is not None:
            # support both safetensors and pt filetypes for fisher information
            from os import listdir
            from os.path import isfile, join
            onlyfiles = [join(args.fisher, f) for f in listdir(args.fisher) if (('pytorch_model' in f or 'safetensors' in f) and 'index' not in f)]

            mypath = onlyfiles[0]
            if 'safe' in mypath:
                from safetensors.torch import load_file
                fisher = load_file(mypath, device = 'cpu')
                for i in range(1,len(onlyfiles)):
                    d2 = load_file(onlyfiles[i], device = 'cpu')
                    fisher.update(d2)
            else:
                fisher = torch.load(mypath, map_location='cpu')
                for i in range(1,len(onlyfiles)):
                    d2 = torch.load(onlyfiles[i], map_location='cpu')
                    fisher.update(d2)
        else:
            fisher = None

        # run calibration
        quantizers = llama_calibration(
            model,
            dataloader,
            DEV,
            args.perchannel,
            args.pertoken,
            args.abits,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            nuq=args.nuq,
            fisher=fisher,
            norm=args.norm
        )

        with open(args.quantizer_path, 'wb') as handle:
            pickle.dump(quantizers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        # load quantizers and evaluate model

        with open(args.quantizer_path, 'rb') as handle:
            quantizers = pickle.load(handle)

        # replace layers
        perchannelquant = {}
        pertokenquant = {}

        perchannel_match = args.perchannel
        pertoken_match = args.pertoken

        for k in quantizers.keys():
            # quantizers[k] = quantizers[k] + (-1, ) # empty for now (used to be LN params)

            # filter out tensor list
            for p in perchannel_match:
                if p in k:
                    perchannelquant[k] = quantizers[k]

            for p in pertoken_match:
                if p in k:
                    pertokenquant[k] = quantizers[k]

        #per-vector quant
        make_quant_sim(
            model,
            perchannelquant,
            args.abits,
            perchannel=True,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=False,
            nuq=args.nuq,
            nf_nuq=args.nf,
            norm=args.norm
        )

        #per-vector quant
        make_quant_sim(
            model,
            pertokenquant,
            args.abits,
            perchannel=False,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=True,
            nuq=args.nuq,
            nf_nuq=args.nf,
            norm=args.norm
        )

        #run evaluation
        llama_eval(model, testloader, DEV)
