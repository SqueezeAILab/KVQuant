    from dotenv import load_dotenv
import os
import tiktoken
import glob
import json
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import numpy as np
import asyncio
from asyncio import Semaphore
from datetime import datetime, timezone
import time

from transformers import LlamaForCausalLM
from transformers import LlamaConfig
from transformers import LlamaTokenizer

from kvquant.modelutils import *
from kvquant.datautils import *
from kvquant.simquant_module_quantizer import *

from kvquant.model_parse import (
    parse_model,
    get_layers,
    get_embedding,
    get_norm,
)

from tqdm import tqdm
import argparse
import random
import pickle
import tiktoken

class LLMNeedleHaystackTester:
    OURS_TEMPLATE = "You are a helpful assistant. USER: {context} {question} Don't give information outside the document or repeat your findings. Keep your response short and direct. ASSISTANT: "
    RANDOM_NEEDLE_CITIES  = [
        'Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
        'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
        'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
        'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
        'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
        'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
        'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
        'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
        'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
        'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
        'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
        'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta'
    ]

    def __init__(self,
                 needle="",
                 haystack_file="",
                 retrieval_question="What is the special magic {} number?",
                 results_version = 1,
                 rnd_number_digits = 7,
                 context_lengths_min = 1000,
                 context_lengths_max = 126000,
                 context_lengths_num_intervals = 10,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percent_interval_type = "linear",
                 save_results = False,
                 final_context_length_buffer = 200,
                 print_ongoing_status = True,
                 output_file='',
                 n_rounds=2,
                 model=None,
                 tokenizer=None):
        needle="\nThe special magic {city} number is: {rnd_number}\n"
        self.needle = needle
        if not needle or not haystack_file or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.rnd_number_digits = rnd_number_digits
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.haystack_file = haystack_file
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        self.output_file = output_file
        self.n_rounds = n_rounds

        self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        if document_depth_percent_interval_type == 'linear':
            self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
        elif document_depth_percent_interval_type == 'sigmoid':
            self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            raise ValueError(f"Unsupported document_depth_percent_interval_type: {document_depth_percent_interval_type}")

        self.model = model
        self.enc = tokenizer
        self.enc_tiktoken = tiktoken.encoding_for_model("gpt-4-1106-preview")

    def generate_random_number(self, num_digits):
        lower_bound = 10**(num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def read_context_files(self, n):
        max_context_length = max(self.context_lengths)
        contexts = []
        f = open(self.haystack_file, 'r')
        for _ in range(n):
            context = ""
            toks = 0
            while toks < max_context_length:
                text = json.loads(f.readline())['text']
                context += text
                toks += len(self.enc.encode(text))
            contexts.append(context)
        return contexts

    def encode_and_trim(self, context, context_length):
        tokens = self.enc.encode(context)
        if len(tokens) > context_length:
            context = self.enc.decode(tokens[:context_length])
        return context

    def create_contexts(self, needle_rnd_number, insert_needle, random_city, trim_context, context_length, depth_percent, seed):
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return
        needle = self.needle.format(city=random_city, rnd_number=needle_rnd_number)
        question = self.retrieval_question.format(random_city)
        if not insert_needle:
            needle = " " #replace needle with a space
        context = self.generate_context(needle, trim_context, context_length, depth_percent)
        results = {
            'context' : context,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'needle' : needle,
            'question' : question,
            'insert_needle' : insert_needle,
            'needle_rnd_number' : needle_rnd_number,
            'seed': seed,
         }
        return results

    def insert_needle(self, needle, context, depth_percent, context_length):
        tokens_needle = self.enc_tiktoken.encode(needle)
        tokens_context = self.enc_tiktoken.encode(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.enc_tiktoken.encode('.')

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.enc_tiktoken.decode(tokens_new_context)
        return new_context

    def generate_context(self, needle, trim_context, context_length, depth_percent):
        context = self.insert_needle(needle, trim_context, depth_percent, context_length)
        return context

    def run_test(self):
        contexts = []
        template = self.OURS_TEMPLATE

        def _key_from_result(result):
            return (result['context_length'], result['depth_percent'], result['seed'])

        results = []
        results2 = []
        completed = set()
        def exists(fname):
            return os.path.exists(fname)
        if exists(self.output_file):
            with open(self.output_file, 'r') as f:
                results = json.load(f)
                completed = set([_key_from_result(result) for result in results])
        print('completed', len(completed))

        full_contexts = self.read_context_files(self.n_rounds)
        full_tokens = []
        for full_context in full_contexts:
            full_tokens.append(self.enc.encode(full_context))


        start = time.time()
        for context_length in self.context_lengths:
            trim_contexts = []
            for full_token in full_tokens:
                trim_contexts.append(self.enc.decode(full_token[:context_length]))
            max_output_length = context_length + 1024 # buffer for response

            contexts = []
            for depth_percent in self.document_depth_percents:
                for i in range(self.n_rounds):
                    if (int(context_length), float(depth_percent), i) in completed:
                        continue
                    random_city = random.choice(LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES)
                    insert_needle = True
                    needle_rnd_number = str(self.generate_random_number(self.rnd_number_digits))
                    print("context length: " + str(context_length))
                    print("depth_percent : " + str(depth_percent))
                    context = self.create_contexts(needle_rnd_number, insert_needle, random_city, trim_contexts[i], context_length, depth_percent, i)
                    contexts.append(context)

            if len(contexts) == 0:
                continue

            # batch size 1 is required when using inference code
            B = 1
            n_pad = 0

            pbar = tqdm(total=len(contexts))
            for i in range(0, len(contexts), B):
                contexts_i = contexts[i:i+B]
                contexts_i_tmp = contexts_i[0]

                prompt = template.format(context=contexts_i_tmp['context'], question=contexts_i_tmp['question'])
                prompt_ind_tensor = self.enc(prompt, return_tensors="pt").input_ids

                # Generate
                t1 = time.time()
                outs = model.generate(
                        prompt_ind_tensor.cuda(),
                        max_length=max_output_length,
                        do_sample=False,
                        use_cache=True
                )
                t2 = time.time()
                print('time: ', t2-t1)

                # reset KV cache after getting response
                layers = model.model.layers
                if args.quantizer_path is not None:
                    with open(args.quantizer_path, 'rb') as handle:
                        quantizers = pickle.load(handle)
                    for k in quantizers.keys():
                        ln = int(k.split('.')[-3]) # layer number
                        q = quantizers[k]
                        if "k_proj" in k:
                            layers[ln].self_attn.kcache.reset()
                        elif "v_proj" in k:
                            layers[ln].self_attn.vcache.reset()

                # get response as a string
                outs = [self.enc.decode(outs[0])]

                # append results to text file
                for j, (context, out) in enumerate(zip(contexts_i, outs)):
                    if i + j < n_pad:
                        continue
                    results.append({
                        'context_length': context['context_length'],
                        'depth_percent': context['depth_percent'],
                        'response': out,
                        'answer': context['needle_rnd_number'],
                        'correct': context['needle_rnd_number'] in out,
                        'seed': context['seed'],
                    })
                    results2.append({
                        'context_length': context['context_length'],
                        'depth_percent': context['depth_percent'],
                        'correct': context['needle_rnd_number'] in out
                    })
                with open(self.output_file, 'w') as f:
                    json.dump(results, f)
                with open(self.output_file+'.tmp', 'w') as f:
                    json.dump(results2, f)
                pbar.update(len(contexts_i))
            pbar.close()
        print('elapsed', time.time() - start)
        print('done')


    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()

if __name__ == "__main__":

    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxseqlen", type=int, default=131072)
    parser.add_argument("--path_to_ckp", type=str, default="/home/v-daweizhu/teamdrive/model/llama-7b")
    parser.add_argument("--simquant",  action='store_true')
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--quantizer-path", type=str, default=None, help='quantizer path')
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
        help='Number of initial tokens to store in fp16.'
    )

    args = parser.parse_args()

    haystack_file="/home/chooper/LWM/pg19.json"
    max_tokens_per_batch=args.maxseqlen
    output_file="results.json"
    context_lengths_min=1024
    context_lengths_max=args.maxseqlen-1024
    n_context_length_intervals=10
    n_document_depth_intervals=10
    n_rounds=3

    # load local model
    model_name_or_path = args.path_to_ckp
    config = LlamaConfig.from_pretrained(model_name_or_path)
    config.first_few_fp16 = args.first_few_fp16
    config.maxseqlen = args.maxseqlen
    config.abits = args.abits
    config.include_sparse = args.include_sparse
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=config, torch_dtype=torch.float16, trust_remote_code=True, use_flash_attention_2=True )
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=True, batched=True)

    # code for using quantized models
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
                layers[ln].self_attn.kcache.load_lookup_table(q, args.include_sparse, args.sparsity_threshold)
            elif "v_proj" in k:
                layers[ln].self_attn.vcache.reset()
                layers[ln].self_attn.vcache.load_lookup_table(q, args.include_sparse, args.sparsity_threshold)

    model = model.cuda()
    _model = model.model
    _model.set_devices()

    ht = LLMNeedleHaystackTester(
        haystack_file=haystack_file,
        context_lengths_min=context_lengths_min,
        context_lengths_max=context_lengths_max,
        context_lengths_num_intervals=n_context_length_intervals,
        document_depth_percent_intervals=n_document_depth_intervals,
        output_file=output_file,
        n_rounds=n_rounds,
        model=model,
        tokenizer=tokenizer
    )
    ht.start_test()
