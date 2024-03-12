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

from transformers import AutoTokenizer

import pickle
import json

DEV = torch.device('cuda:0')

def get_model(model, maxseqlen, bits, include_sparse, first_few_fp16):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model)
    config.first_few_fp16 = first_few_fp16
    config.maxseqlen = maxseqlen
    config.abits = bits
    config.include_sparse = include_sparse
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, config=config, torch_dtype=torch.half, use_flash_attention_2=True, device_map="cpu")
    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--text', type=str,
        help='input text'
    )

    parser.add_argument(
        '--min_length', type=int, default=10,
        help='The minimum length of the sequence to be generated.'
    )

    parser.add_argument(
        '--max_length', type=int, default=256,
        help='The maximum length of the sequence to be generated.'
    )

    parser.add_argument(
        '--maxseqlen', type=int, default=-1,
        help='Used to set KV cache size'
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
    parser.add_argument(
        '--first_few_fp16', type=int, default=0,
        help='Store first few tokens separately in fp16'
    )
    parser.add_argument(
        '--norm', action='store_true',
        help='Whether to use q-norm.'
    )

    DEV = torch.device('cuda:0')

    args = parser.parse_args()

    model = get_model(args.model, args.maxseqlen, args.abits, args.include_sparse, args.first_few_fp16)
    model.eval()
    model.model.set_devices()

    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    input_ids = tokenizer.encode(args.text, return_tensors="pt").to(DEV)

    print('Load quantizers.')
    layers = model.model.layers
    if args.quantizer_path is not None:
        with open(args.quantizer_path, 'rb') as handle:
            quantizers = pickle.load(handle)
        for k in quantizers.keys():
            if '.lut' in k:
                continue
            ln = int(k.split('.')[-3]) # layer number
            q = quantizers[k]

            if "k_proj" in k:
                layers[ln].self_attn.kcache.reset()
                layers[ln].self_attn.kcache.load_lookup_table(q, args.include_sparse, args.sparsity_threshold, args.norm)
            elif "v_proj" in k:
                layers[ln].self_attn.vcache.reset()
                layers[ln].self_attn.vcache.load_lookup_table(q, args.include_sparse, args.sparsity_threshold, args.norm)

    model = model.half()
    model = model.to(DEV)

    with torch.no_grad():
        t1 = time.time()
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=args.min_length,
            max_length=args.max_length,
            use_cache=True
        )
        t2 = time.time()
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
    print('Time: ', t2-t1)
