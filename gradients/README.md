## Gradient Computation Repository
We the Fisher Information matrix as a sensitivity metric. This repository, which builds on top of Huggingface's transformer library, is designed to calculate the Fisher sensitivity score (gradient square). This score can be employed in the quantization pipeline.

### Prerequisite
You will need to have your own Huggingface-compatible LLaMA checkpoint saved at `[MODEL_PATH]`.

Run the following command for setup:
```
conda create -n grad python=3.9 -y
conda activate grad
pip install -e .
pip install -r requirements.txt
pip install accelerate -U
```

### Command
Run the following command:
```
CUDA_VISIBLE_DEVICES=0 python run-fisher.py --model_name_or_path [MODEL_PATH] --output_dir [OUTPUT_PATH] --dataset wikitext2 --seqlen 2048 --maxseqlen 2048 --num_examples 16 
```

This command performs the following steps

1. Loads the model from `[MODEL_PATH]`.
2. Computes the gradient square using a subset of the wikitext2 training dataset as a calibration set. You can define and use your own calibration dataset.
3. Outputs the gradient square at `[OUTPUT_PATH]`. The output format will be identical to the loaded Huggingface model checkpoint, with the only difference being that the weight values are replaced by the gradient square.

If the model size exceeds the capacity of a single GPU, our framework provides an option to distribute the model across multiple GPUs.
This is automated by configuring multiple CUDA visible devices.
To be specific, the model is partitioned into multiple chunks of consecutive layers, and each segment is assigned to an individual GPU device.

You can also use the `--num_examples` argument to change the number of calibration examples. This defaults to 16.

### Troubleshoot
If you are getting the following error, please open `[MODEL_PATH]/tokenizer_config.json` and fix `"tokenizer_class": "LlamaTokenizer"` to `"tokenizer_class": "LLaMATokenizer"`.
```
ValueError: Tokenizer class LlamaTokenizer does not exist or is not currently imported.
```
