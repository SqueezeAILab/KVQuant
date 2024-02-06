import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.optim as optim
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

from tqdm import tqdm
import utils

# from memory_profiler import profile

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_DICT_NEW={
    "prompt":(
        "Below is a text imitation task. You will be given a text description and asked to rewrite it in a different style.\n\n"
        "### Input:\n{input}\n\n### Output:"
    )
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    load: Optional[str] = field(default="")
    #save_grad_path: str = field(
    #    metadata={"help": "Path to save the gradients"}
    #)


@dataclass
class DataArguments:
    dataset: str = field(default="c4")
    num_examples: int = field(default=16, metadata={"help": "Number of calibration examples"})
    seqlen: int = field(default=2048)
    maxseqlen: int = field(default=32768)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    print("[func] make_supervised_data_module")
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_modules(layer):
    # NOTE: This is llama-specific
    # For other models, replace this with proper names for all linear layers
    return[
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ]

def get_modules_kv(layer):
    # NOTE: This is llama-specific
    # For other models, replace this with proper names for all linear layers
    return[
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
    ]

# @profile
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.dataset == "c4":
        from datautils import get_loaders
        print("Calibration with C4 ")
        dataloader, testloader = get_loaders(data_args.dataset,  model=model_args.model_name_or_path, seqlen=data_args.seqlen, seed=0)
    elif data_args.dataset == "wikitext2":
        from datautils import get_loaders
        print("Calibration with Wikitext2 ")
        dataloader, testloader = get_loaders(data_args.dataset,  model=model_args.model_name_or_path, seqlen=data_args.seqlen, seed=0)
    else:
        raise NotImplementedError("Please define your own dataset here")


    # Set RoPE scaling factor
    import math
    from transformers import AutoConfig
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    context_size = data_args.maxseqlen
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    config._flash_attn_2_enabled = True

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )

#    model.seqlen = seqlen  #TODO
    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)

    model = model.bfloat16()
    try:
        model.lm_head.cuda()
    except:
        pass

    if model_args.load != "":
        model.load_state_dict(torch.load(model_args.load), strict=False)
        model.eval()

    # NOTE: this is llama-specific
    # For other models, replace this with proper variable names for model and layers
    _model = model.model
    _layers = _model.layers
    _model.set_devices()
    grads = {}

    # main loop
    for i, data in tqdm(enumerate(dataloader[:data_args.num_examples])):
        data = data[0]
        x = data.cuda()

        # act gradients
        for n, layer in enumerate(_layers):
            k_proj, v_proj = get_modules_kv(layer)
            k_proj.retain_grad = True
            v_proj.retain_grad = True

        # compute gradients
        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss
        loss.backward()

        # get grads
        for i, layer in enumerate(_layers):
            print(f'weight layer {i}')
            k_proj, v_proj = get_modules_kv(layer)
            kgrad = (k_proj.act.grad ** 2).float().cpu()
            vgrad = (v_proj.act.grad ** 2).float().cpu()

            if f'k_proj{i}' not in grads:
                grads[f'k_proj{i}'] = kgrad
            else:
                grads[f'k_proj{i}'] = torch.cat((grads[f'k_proj{i}'], kgrad), dim=1)
            if f'v_proj{i}' not in grads:
                grads[f'v_proj{i}'] = vgrad
            else:
                grads[f'v_proj{i}'] = torch.cat((grads[f'v_proj{i}'], vgrad), dim=1)

    ## This is a hacky solution to save the gradients
    # where we overwrite all the weights in the model as the gradients
    # and use HF save_pretrained`
    for i, layer in enumerate(_layers):
        k_proj, v_proj = get_modules_kv(layer)
        k_proj.weight.data = grads[f'k_proj{i}']
        v_proj.weight.data = grads[f'v_proj{i}']

    print(f"saving model gradient at {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
