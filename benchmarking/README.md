# KVQuant Deployment Code

The code in this folder can be used to run the inference experiments from the paper (for benchmarking kernel runtime).

## Installation

1. Create a conda environment
```
conda create --name benchmark python=3.9 -y
conda activate benchmark
```

2. Clone and install the dependencies (including the local transformers environment)
```
pip install transformers
pip install torch datasets sentencepiece scikit-learn protobuf
pip install accelerate -U
pip install -e .
cd kvquant
python setup_cuda.py install
cd ..
pip install flash-attn --no-build-isolation
```

3. Run kernel benchmarking

Note that the quantizer is obtained from steps in the quant directory.

```
cp ../quant/quantizers.pickle .
CUDA_VISIBLE_DEVICES=0 python cache-llama-activations.py <path-to-llama-7b-hf> --wbits 4 --nsamples 1 --seqlen 2048 --quantizer-path quantizers.pickle --output-path activations-seqlen2048.pickle;
```

Assuming the activations and the quantizers are stored in "activations.pickle" and "quantizers.pickle", you can run the kernel benchmarks by running the python scripts in the "scripts" folder.

