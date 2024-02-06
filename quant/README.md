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

---
## Scripts (WIP)

### llama_simquant.py - used for simulated quantization experiments

1. Quantize the model using nuq4-1% (note that <path-to-fisher-info> is the path to the directory after computing gradients, and that this step is mostly run on the CPU since we need to run K-means):
```
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py <path-to-llama-7b-hf> --abits 4 --zeropoint --nsamples 16 --seqlen 2048 --nuq --fisher <path-to-fisher-info> --quantize --include_sparse --sparsity-threshold 0.99 --quantizer_path quantizers.pickle ;
```

2. Evaluate with quantizer using nuq4-1%:
```
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py <path-to-llama-7b-hf> --abits 4 --zeropoint --nsamples 16 --seqlen 2048 --nuq --include_sparse --sparsity-threshold 0.99 --quantizer_path quantizers.pickle ;
```

3. Passkey evaluation with quantizer using nuq4-1% (note that the multi-GPU inference environment for running long sequence length passkey evaluation with larger models / longer context lengths is on the roadmap):
```
CUDA_VISIBLE_DEVICES=0 python eval_passkey_simquant.py --path_to_ckp <path-to-llama-7b-hf> --abits 4 --simquant --quantizer_path quantizers.pickle ;
```

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
