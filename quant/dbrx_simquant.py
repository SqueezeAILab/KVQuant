import time

import torch
import torch.nn as nn

from kvquant.modelutils import *
from kvquant.datautils import *
from kvquant.simquant_module_quantizer_dbrx import *

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

def get_model(model, seqlen):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.half)

    model.seqlen = seqlen
    return model

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')
    model_type = parse_model(model)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.blocks

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
            hidden_states = model.transformer.norm_f(hidden_states)
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
def llama_calibration(model, dataloader, dev, bits, include_sparse=False, sparsity_threshold=0.999, nuq=False, fisher=None, norm=False, cap_outliers=False, first_few_fp16=False):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.blocks

    model.transformer.wte = model.transformer.wte.to(dev)
    model.transformer.norm_f = model.transformer.norm_f.to(dev)
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
    model.transformer.wte = model.transformer.wte.cpu()
    model.transformer.norm_f = model.transformer.norm_f.cpu()
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
        full_list = []
        for f in full:
            if "qkv" in f:
                full_list.append(f)
        subset = {n: full[n] for n in ['norm_attn_norm.attn.Wqkv'] if n in full_list}

        simquant = {}
        name = 'norm_attn_norm.attn.Wqkv'
        simquant["k_proj"] = SimQuant(
                                        subset[name],
                                        bits,
                                        perchannel=True,
                                        qchannel=0
                                     )
        simquant["v_proj"] = SimQuant(
                                        subset[name],
                                        bits,
                                        perchannel=True,
                                        qchannel=-1
                                     )

        def add_batch(name):
            def tmp(_, inp, out):
                # only store k and v activations
                simquant["k_proj"].add_batch(inp[0].data, out.data[:,:,6144:7168])
                simquant["v_proj"].add_batch(inp[0].data, out.data[:,:,7168:])
            return tmp
        handles = []

        handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

        for h in handles:
            h.remove()

        if fisher is not None:
            key = 'transformer.blocks.%d.%s' % (i, name)
            key = key + '.weight'
            fisher_info = fisher[key].cpu()
            fisher_info_key = fisher_info[:,:,:1024]
            fisher_info_value = fisher_info[:,:,1024:]
        else:
            fisher_info = None
            fisher_info_key = None
            fisher_info_value = None

        #cap is always onlyK
        if "k_proj" in name:
            if cap_outliers == -1:
                cap = False
            else:
                cap = True
        else:
            cap = False

        quantizers['transformer.blocks.%d.%s' % (i, "k_proj")] = simquant["k_proj"].quantize(
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                nuq=nuq,
                fisher=fisher_info_key,
                norm=norm,
                cap_outliers=cap,
                first_few_fp16=first_few_fp16
        )
        simquant["k_proj"].free()

        quantizers['transformer.blocks.%d.%s' % (i, "v_proj")] = simquant["v_proj"].quantize(
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                nuq=nuq,
                fisher=fisher_info_value,
                norm=norm,
                cap_outliers=cap,
                first_few_fp16=first_few_fp16
        )
        simquant["v_proj"].free()

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
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'c4'], default='wikitext2',
        help='Which dataset to use for calibration / evaluation.'
    )

    # arguments for capping outliers and for attention sink
    parser.add_argument(
        '--cap_outliers', type=float, default=-1,
        help='Max % of outliers to retain per token.'
    )
    parser.add_argument(
        '--first_few_fp16', type=int, default=-1,
        help='Leave first few outlier tokens.'
    )
    parser.add_argument(
        '--clamp', action='store_true',
        help='Clamp w/ integer quantization'
    )

    DEV = torch.device('cuda:0')

    args = parser.parse_args()

    #load model
    print('Loading model ...')
    if args.load:
        model = get_model(args.model, args.seqlen)
        model.load_state_dict(torch.load(args.load))
        model.eval()
    else:
        model = get_model(args.model, args.seqlen)
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
            args.abits,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            nuq=args.nuq,
            fisher=fisher,
            norm=args.norm,
            cap_outliers=args.cap_outliers,
            first_few_fp16=args.first_few_fp16
        )

        with open(args.quantizer_path, 'wb') as handle:
            pickle.dump(quantizers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif args.quantizer_path is not None:

        model = model.half()

        # load quantizers and evaluate model

        with open(args.quantizer_path, 'rb') as handle:
            quantizers = pickle.load(handle)

        # need to merge K and V quantizer files during eval due to merged QKV
        quantizers_merged = {}
        for i in range(40):
            k_proj = quantizers[f"transformer.blocks.{i}.k_proj"]
            v_proj = quantizers[f"transformer.blocks.{i}.v_proj"]
            quantizers_merged[f"transformer.blocks.{i}.norm_attn_norm.attn.Wqkv"] = (k_proj,v_proj)

        # quantize wQKV
        make_quant_sim(
            model,
            quantizers_merged,
            args.abits,
            perchannel=True,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            nuq=args.nuq,
            nf_nuq=args.nf,
            norm=args.norm,
            cap_outliers=args.cap_outliers,
            first_few_fp16=args.first_few_fp16,
            clamp=args.clamp
        )

        #run evaluation
        res = llama_eval(model, testloader, DEV)
    else:
        # evaluate fp16 baseline model
        model = model.half()
        res = llama_eval(model, testloader, DEV)
