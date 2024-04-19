# KVQuant simulated quantization

This file contains the instructions to quantize a new model and to run simulated quantization.

---

## Installation

1. Create a conda environment
```
conda create --name kvquant python=3.10 -y
conda activate kvquant
```

2. Clone and install the dependencies
```
cd quant
pip install -e .
pip install flash-attn --no-build-isolation
```

3. (Optional) For DBRX evaluation, please install transformers from source in the dbrx directory.

---
## Scripts for evaluating LLaMA / Mistral Models

### llama_simquant.py - used for simulated quantization experiments

1. Quantize the model using nuq4-1% (note that <path-to-fisher-info> is the path to the directory after computing gradients, and that this step is mostly run on the CPU since we need to run K-means):
```
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py <path-to-llama-7b-hf> --abits 4 --nsamples 16 --seqlen 2048 --nuq --fisher <path-to-fisher-info> --quantize --include_sparse --sparsity-threshold 0.99 --quantizer-path quantizers.pickle ;
```

2. Evaluate with quantizer using nuq4-1%:
```
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py <path-to-llama-7b-hf> --abits 4 --nsamples 16 --seqlen 2048 --nuq --include_sparse --sparsity-threshold 0.99 --quantizer-path quantizers.pickle ;
```

3. Passkey evaluation with quantizer using nuq4-1% (note that the multi-GPU inference environment for running long sequence length passkey evaluation with larger models / longer context lengths is on the roadmap):
```
CUDA_VISIBLE_DEVICES=0 python eval_passkey_simquant.py --path_to_ckp <path-to-llama-7b-hf> --abits 4 --simquant --quantizer-path quantizers.pickle ;
```

---

## DBRX Results

Quantized checkpoints are provided for the DBRX-Base and DBRX-Instruct models. Perplexity evaluation on Wikitext-2 is included below for both DBRX-Base and DBRX-Instruct using KVQuant (input length 2k). 

These checkpoints leverage Attention-Sink Aware Quantization, so the `--first_few_fp16 1` argument must be used when running perplexity evaluation. Additionally, for the 2-bit checkpoints, Q-Norm is used, so the `--norm` argument should be passed in.

Example DBRX evaluation run:

```
CUDA_VISIBLE_DEVICES=0 python dbrx_simquant.py databricks/dbrx-base --abits 4 --nsamples 16 --seqlen 2048 --nuq --include_sparse --sparsity-threshold 0.99 --quantizer-path <path-to-quantizer-pickle-file> --first_few_fp16 1 ;
```

### DBRX-Base

| Model |  fp16 | nuq4-1% | nuq3-1% |  nuq2-1% |
| -------- | -------- | -------- | -------- | -------- |
| Perplexity    |  3.96 | 3.98 | 4.03 | 4.26 | 
| Checkpoint | | [dbrx-base-nuq4-s1](https://huggingface.co/squeeze-ai-lab/dbrx-base-a4-s1) | [dbrx-base-nuq3-s1](https://huggingface.co/squeeze-ai-lab/dbrx-base-a3-s1) | [dbrx-base-nuq2-s1](https://huggingface.co/squeeze-ai-lab/dbrx-base-a2-s1) |

### DBRX-Instruct

| Model |  fp16 | nuq4-1% | nuq3-1% |  nuq2-1% |
| -------- | -------- | -------- | -------- | -------- |
| Perplexity    |  4.30 | 4.31 | 4.36 | 4.61 | 
| Checkpoint | | [dbrx-instruct-nuq4-s1](https://huggingface.co/squeeze-ai-lab/dbrx-instruct-a4-s1) | [dbrx-instruct-nuq3-s1](https://huggingface.co/squeeze-ai-lab/dbrx-instruct-a3-s1) | [dbrx-instruct-nuq2-s1](https://huggingface.co/squeeze-ai-lab/dbrx-instruct-a2-s1) |

---

### Troubleshoot
If you are getting the following error, please open `[MODEL_PATH]/tokenizer_config.json` and fix `"tokenizer_class": "LlamaTokenizer"` to `"tokenizer_class": "LLaMATokenizer"`.
```
ValueError: Tokenizer class LlamaTokenizer does not exist or is not currently imported.
```

### Troubleshoot
`--nsamples` has to be the same as `--num_examples` in the gradient step. otherwise, you will face the error:
```
IndexError: The shape of the mask at index 0 does not match the shape of the indexed tensor at index
```
