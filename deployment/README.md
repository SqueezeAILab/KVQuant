# KVQuant Deployment Code

The code in this folder can be used to run the inference experiments from the paper (for benchmarking kernel runtime) and to run end-to-end generation with the compressed KV cache.

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
```

3. Run kernel benchmarking

Note that the quantizer is obtained from steps in the quant directory.

```
cp ../quant/quantizers.pickle .
CUDA_VISIBLE_DEVICES=0 python cache-llama-activations.py <path-to-llama-7b-hf> --wbits 4 --nsamples 1 --seqlen 2048 --quantizer-path quantizers.pickle --output-path activations.pickle;
```

Assuming the activations and the quantizers are stored in "activations.pickle" and "quantizers.pickle", you can run the kernel benchmarks by running the python scripts in the "scripts" folder.

4. Run end-to-end generation

```
cp ../quant/quantizers.pickle .
CUDA_VISIBLE_DEVICES=0 python llama.py <path-to-llama-7b-hf> wikitext2 --abits 4 --include_sparse --sparsity-threshold 0.99 --quantizer-path quantizers.pickle --benchmark 128 --check
```
