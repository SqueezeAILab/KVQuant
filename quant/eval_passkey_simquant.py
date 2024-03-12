import random
import argparse
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from pathlib import Path
import jsonlines

import torch
import torch.distributed
import transformers
import deepspeed
import evaluate
import datasets
import numpy as np
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import LlamaTokenizer, pipeline
from datasets import load_dataset
from evaluate import logging
from tqdm import tqdm

from transformers import LlamaForCausalLM
from transformers import LlamaConfig

from kvquant.modelutils import *
from kvquant.datautils import *
from kvquant.simquant_module_quantizer import *

from kvquant.model_parse import (
    parse_model,
    get_layers,
    get_embedding,
    get_norm,
)

gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_prompt(max_tokens=16384):
    """Generates a text file and inserts an execute line at a random position."""
    # n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_total = (max_tokens - 32 - 26 - 11) // 25
    n_garbage_prefix = random.randint(0, n_garbage_total)
    n_garbage_suffix = n_garbage_total - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there." # 32 tokens
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." # 25 tokens
    garbage_prefix = garbage * n_garbage_prefix
    garbage_suffix = garbage * n_garbage_suffix
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key." # 26 tokens
    final_question = "What is the pass key? The pass key is" # 11 tokens
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def test_model(model, tokenizer, prompt_text, pass_key):

    model_input = tokenizer.encode(prompt_text, return_tensors="pt", max_length=100000, truncation=True).to(gpu_device)

    response = model.generate(model_input, num_return_sequences=1, max_new_tokens=10)
    response = tokenizer.batch_decode(response[:, model_input.shape[1]:], skip_special_tokens=True)[0]
    print(response)

    assert f"The pass key is {pass_key}" in prompt_text

    try:
        pass_key = int(re.search(r'\d+', response).group())
    except:
        pass_key = response[:20]

    return pass_key


def main():

    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxseqlen", type=int, default=32768)
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--path_to_ckp", type=str, default="/home/v-daweizhu/teamdrive/model/llama-7b")
    parser.add_argument("--path_to_output_dir", type=str, default="results/passkey")
    parser.add_argument("--simquant",  action='store_true')
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--quantizer-path", type=str, default=None, help='quantizer path')
    parser.add_argument("--norm",  action='store_true')
    args = parser.parse_args()

    model_name_or_path = args.path_to_ckp
    scaled_max_position_embeddings=args.maxseqlen

    config = LlamaConfig.from_pretrained(model_name_or_path)
    context_size = args.maxseqlen
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    config.use_cache=False
    config._flash_attn_2_enabled=True
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)

    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)

    print('load tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=True, batched=True)

    import pickle
    if args.simquant:
        with open(args.quantizer_path, 'rb') as handle:
            quantizers = pickle.load(handle)

        # replace layers
        perchannelquant = {}
        pertokenquant = {}

        for k in quantizers.keys():
            quantizers[k] = quantizers[k] + (0.1, ) #dummy value
            if "k_proj" in k:
                perchannelquant[k] = quantizers[k]
            if "v_proj" in k:
                pertokenquant[k] = quantizers[k]

        #per-vector quant
        make_quant_sim(
            model,
            perchannelquant,
            args.abits,
            perchannel=True,
            include_sparse=True,
            sparsity_threshold=0.99,
            nuq=True,
            nf_nuq=False,
            norm=args.norm
        )

        #per-vector quant
        make_quant_sim(
            model,
            pertokenquant,
            args.abits,
            perchannel=False,
            dynamicquantization=True,
            include_sparse=True,
            sparsity_threshold=0.99,
            nuq=True,
            nf_nuq=False,
            norm=args.norm
        )

    model.model.set_devices()

    result_list = list()
    length_list = [2048,4096,8192,16384,32768]
    for context_size in length_list:
        if context_size == scaled_max_position_embeddings:
            context_size -= 100
        print(f"context_size: {context_size}")
        correct_cnt = 0
        result_dict = {"scaled_length": scaled_max_position_embeddings, "context_size": context_size}
        iter_nums = 50
        for i in tqdm(range(iter_nums)):
            prompt_text, pass_key = generate_prompt(context_size)
            pred = test_model(model, tokenizer, prompt_text, pass_key)
            result = "Pass!" if pred == pass_key else "Fail!"
            correct_cnt += 1 if pred == pass_key else 0
            case_report = f"pred: {pred}, ans: {pass_key}, result: {result}"
            result_dict[f"case{i}"] = case_report
            #print(case_report)
        print(f"correct_rate: {correct_cnt/iter_nums}")
        result_dict["correct_rate"] = correct_cnt/iter_nums
        result_list.append(result_dict)

    root_dir = Path(__file__).parent.parent
    path_to_output_fn = (root_dir / args.path_to_output_dir / f"{args.model_name}.jsonl").as_posix()

    with jsonlines.open(path_to_output_fn, "w") as writer:
        writer.write_all(result_list)

if __name__ == "__main__":
    main()
