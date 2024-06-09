# KVQuant Deployment Code

The code in this folder can be used for running end-to-end generation with the compressed KV cache.

## Installation

1. Create a conda environment
```
conda create --name deploy python=3.9 -y
conda activate deploy
```

2. Clone and install the dependencies (including the local transformers environment)
```
cd deployment/transformers
pip install -e .
pip install -r requirements.txt
pip install accelerate -U
cd ..
pip install -e .
cd kvquant
python setup_cuda.py install
cd ..
pip install flash-attn==2.5.5 --no-build-isolation
```

3. Run end-to-end generation

Note that the quantizer is obtained from steps in the quant directory.

```
cp ../quant/quantizers.pickle .
CUDA_VISIBLE_DEVICES=0 python llama.py <path-to-llama-7b-hf> wikitext2 --abits 4 --include_sparse --sparsity-threshold 0.99 --quantizer-path quantizers.pickle --benchmark 128 --check
```
