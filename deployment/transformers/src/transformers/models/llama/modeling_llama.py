# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import is_torch_fx_available
from .configuration_llama import LlamaConfig

import quant_cuda

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class LlamaRotaryEmbeddingDynamic(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__() # don't use baseline LlamaRotaryEmbedding as base class to avoid instantiating cached sin/cos

        self.dim = dim
        self.base = base
        self.device = device
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, start_pos, end_pos):
        t = torch.arange(start_pos, end_pos, device=self.device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos_ret = emb.cos().to(x.dtype)
        sin_ret = emb.sin().to(x.dtype)

        return (cos_ret,sin_ret)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    position_ids = position_ids.cpu()
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# apply RoPE to only Q here
def apply_rotary_pos_emb_query(q, cos, sin, position_ids, unsqueeze_dim=1):
    position_ids = position_ids.cpu()
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

# apply RoPE to only Q here using dynamic RoPE cos/sin computation
def apply_rotary_pos_emb_query_dynamic(q, cos, sin, position_ids, unsqueeze_dim=1):
    position_ids = position_ids.cpu()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def compute_lut(
    inp,
    bits=4,
    qchannel=-1,
    zeropoint=True
):
    """
    inp: act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    zeropoint: whether to use asymmetric quantization

    Computes per-channel scaling factor and offset dynamically (using minmax quantization)
    """

    maxval = torch.max(inp, dim=qchannel).values
    minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    if zeropoint: # use zeropoint quantization
        offset = (maxval + minval) / 2
        rangeval = (maxval - minval) / 2
        if len(inp.shape) > 1: #single-vec quant
            offset = offset.unsqueeze(qchannel)
            rangeval = rangeval.unsqueeze(qchannel)
    else: # use absmax quantization
        rangeval = torch.max(torch.abs(maxval), torch.abs(minval))
        if len(inp.shape) > 1: #single-vec quant
            rangeval = rangeval.unsqueeze(qchannel)
        offset = 0

    return rangeval, offset

# class to manage the key cache
class QuantK(nn.Module):
    def __init__(self, bits=2, hidden_size=4096, num_heads=32, max_position_embeddings=-1, include_sparse=False, sparsity_threshold=0.99, rope_theta=10000, use_orig_sparse=False, first_few_fp16=0):

        """
        bits: number of bits for quantization
        hidden_size: hidden size for the model
        num_heads: number of heads for the model
        max_position_embeddings: max sequence length
        include_sparse: whether to isolate outliers
        sparsity_threshold: what percentage of outliers to remove
        rope_theta: theta for RoPE embeddings
        use_orig_sparse: whether to use original sparse kernels for keys (without capping outlier percentage)
        first_few_fp16: how many initial tokens to store in fp16

        This class manages the compressed key cache during generation
        """

        super().__init__()

        # model parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # quantization parameters
        self.bits = bits
        self.lut = None # global LUT
        self.lookup_table = None # stores per-channel LUTs (in current implementation)
        self.zeropoint = None

        # dense-and-sparse quantization parameters
        self.sparsity_threshold = sparsity_threshold
        self.include_sparse = include_sparse
        self.outlier_threshold_upper = None
        self.outlier_threshold_lower = None
        self.num_threads = -1

        # KV cache parameters
        self.max_len = max_position_embeddings
        self.klen = 0
        self.kcache = torch.zeros((self.num_heads, (self.head_dim // 32) * self.bits, self.max_len), dtype=torch.int).cuda()

        # outlier params
        if self.include_sparse:
            self.outliers = torch.zeros((self.max_len,42), dtype=torch.float).cuda()
            self.outlier_indices = torch.zeros((self.max_len,42), dtype=torch.int).cuda()

        # For LWM - rope theta
        self.rope_theta = rope_theta

        # dense-and-sparse quantization parameters for original code
        self.rows = torch.tensor([]).cuda()
        self.cols = torch.tensor([]).cuda()
        self.vals = torch.tensor([]).cuda()
        self.start_rows = torch.tensor([]).cuda()
        self.num_threads = -1
        self.use_orig_sparse = use_orig_sparse

        # first few in fp16
        self.first_few_fp16 = first_few_fp16

        # Q-Norm
        self.norm = False

    def reset(self):
        """
        This class resets the compressed key cache
        """
        self.klen = 0
        self.kcache[self.kcache!=0] = 0

        if self.include_sparse:
            if self.use_orig_sparse:
                # dense-and-sparse quantization parameters for original code
                self.rows = torch.tensor([]).cuda()
                self.cols = torch.tensor([]).cuda()
                self.vals = torch.tensor([]).cuda()
                self.start_rows = torch.tensor([]).cuda()
                self.num_threads = -1

            else:
                self.outliers[self.outliers!=0] = 0
                self.outlier_indices[self.outlier_indices!=0] = 0

    # Initial function to load lookup tables
    def load_lookup_table(self, quantizer, include_sparse = True, sparsity_threshold = 0.99, norm=False): # need to load using calibration data
        """
        quantizer: calibrated outlier thresholds / LUT for the current layer
        include_sparse: whether to isolate outliers
        sparsity_threshold: what percentage of outliers to remove
        norm: whether to use Q-Norm

        Loads the LUT and outlier thresholds for the key cache
        """

        self.outlier_threshold_upper = torch.tensor(quantizer[0]).cuda().half().flatten()
        self.outlier_threshold_lower = torch.tensor(quantizer[1]).cuda().half().flatten()
        self.lut = torch.tensor(quantizer[2][0]).squeeze(-1).to(self.kcache.device)
        self.lut,_ = self.lut.sort()
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold

        # initialize lookup table
        self.lookup_table = torch.zeros((self.num_heads, self.head_dim, 2 ** self.bits))
        maxval = self.outlier_threshold_upper
        minval = self.outlier_threshold_lower

        # compute offset and scaling factor here:
        offset = (maxval + minval) / 2
        rangeval = (maxval - minval) / 2

        # Q-Norm
        self.norm = norm
        if self.norm:
            self.normscale = quantizer[3].cuda()
            self.normoffset = quantizer[4].cuda()
            self.lookup_table2 = torch.zeros((self.num_heads, self.head_dim, 2 ** self.bits))
        else:
            self.normscale = None
            self.normoffset = None
            self.lookup_table2 = None

        # initialize per-channel LUT
        for i in range(self.num_heads):
            for j in range(self.head_dim):

                idx = i * self.head_dim + j
                sf_tmp = rangeval[idx]
                offset_tmp = offset[idx]

                lut_tmp = torch.tensor(self.lut)
                lut_tmp,_ = lut_tmp.sort()

                # Q-Norm
                if norm:
                    lut_tmp2 = (lut_tmp * self.normscale + self.normoffset) * sf_tmp.item() + offset_tmp.item()
                    self.lookup_table2[i,j] = lut_tmp2

                lut_tmp = lut_tmp * sf_tmp.item() + offset_tmp.item()
                self.lookup_table[i,j] = lut_tmp

        self.lookup_table = self.lookup_table.cuda()
        if self.norm:
            self.lookup_table2 = self.lookup_table2.cuda()

        # initialize zeropoint
        self.zeropoint = (self.outlier_threshold_upper + self.outlier_threshold_lower) / 2
        self.zeropoint = self.zeropoint.float().cuda()
        self.outlier_threshold_upper = self.outlier_threshold_upper.float().cuda()
        self.outlier_threshold_lower = self.outlier_threshold_lower.float().cuda()

    # forward pass to compute Q*K^T with the compressed KV cache
    def forward_fused_sparse_orig(self, q, k):
        """
        q: current query vector
        k: key vector to append

        Performs the forward pass using the compressed key cache (and also appends a new key vector)
        This function contains the original implementation without capping the outlier percentage
        """

        assert(self.include_sparse)
        assert(self.bits == 4) # 3-bit / 2-bit not implemented yet

        k = k.flatten()
        q = q.float()
        q = q.transpose(0,1).contiguous()
        k = k.float()

        # sparse packing kernel
        self.rows, self.cols, self.vals, self.start_rows, self.num_threads, outlier_count = quant_cuda.vecquant4appendvecKsparseorig(
            self.kcache,
            self.lookup_table,
            k,
            self.zeropoint,
            self.rows,
            self.cols,
            self.vals,
            self.start_rows,
            self.outlier_threshold_lower,
            self.outlier_threshold_upper,
            self.klen - self.first_few_fp16
        )
        self.num_threads = self.num_threads[0]
        self.num_nonzeros = self.vals.shape[0]

        self.klen += 1
        mul = torch.zeros((q.shape[0], q.shape[1], self.klen - self.first_few_fp16), dtype=torch.float, device=q.device) #TODO: support longer q seqlens

        quant_cuda.vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig(
            q,
            self.kcache,
            mul,
            self.lookup_table,
            self.klen - self.first_few_fp16,
            self.rows,
            self.cols,
            self.start_rows,
            self.vals,
            self.klen - self.first_few_fp16,
            self.num_threads,
            self.num_nonzeros,
            self.rope_theta,
            self.first_few_fp16
        )

        mul = mul.transpose(0,1).contiguous()
        mul = mul.half()
        return mul

    # forward pass to compute Q*K^T with the compressed KV cache
    def parallel_pack_orig(self, k):
        """
        k: key vectors to append

        Pack several key vectors in parallel
        This function contains the original implementation without capping the outlier percentage
        """

        assert(self.include_sparse)
        assert(self.bits == 4) # 3-bit / 2-bit not implemented yet

        k = k.float().contiguous()

        # update klen
        self.klen += k.shape[-1]

        # sparse packing kernel
        if self.include_sparse:
            outliers_rescaled = k.clone()

            if self.bits == 4:
                quant_cuda.vecquant4appendvecKsparseParallel(
                    self.kcache,
                    self.lookup_table,
                    k,
                    outliers_rescaled,
                    self.outlier_threshold_lower,
                    self.outlier_threshold_upper,
                )

            else:
                # 4-bit / 3-bit aren't implemented yet
                assert (False)

            # detect outliers
            k = k.reshape(-1,k.shape[-1])
            k_outliers_lower = (k < self.outlier_threshold_lower.unsqueeze(-1))
            k_outliers_above = (k > self.outlier_threshold_upper.unsqueeze(-1))

            # subtract high / low LUT values
            numvals = 2 ** self.bits
            lut = self.lookup_table.reshape(-1,self.lookup_table.shape[-1])
            lut_lower = lut[:,0].unsqueeze(-1)
            lut_upper = lut[:,numvals-1].unsqueeze(-1)

            # subtract nearest values
            k1 = k - lut_lower
            k2 = k - lut_upper
            k[k_outliers_lower] = k1[k_outliers_lower]
            k[k_outliers_above] = k2[k_outliers_above]

            # zero out non-outliers
            k_outliers = torch.logical_or(k_outliers_above, k_outliers_lower)
            k_not_outliers = ~k_outliers
            k[k_not_outliers] = 0

            # pack as CSR
            csr_mat = k.t().contiguous().to_sparse_csr()

            # extract rows, cols, vals
            self.rows = csr_mat.crow_indices().int()
            self.cols = csr_mat.col_indices().int()
            self.vals = csr_mat.values().float()

            # initialize start_rows / num_threads
            if len(self.vals) > 0:

                self.num_threads = int((self.vals.shape[0]+9) / 10)
                nnz_per_thread = 10
                self.num_nonzeros = self.vals.shape[0]

                # create start_rows (all -1)
                num_threads_tmp = int((self.num_threads + 127) / 128) * 128 # 128 is blocksize
                self.start_rows = -torch.ones((num_threads_tmp,)).int().cuda()

                # need to initialize multiple starting rows
                j = 0
                for i in range(len(self.rows) - 1):
                    start = self.rows[i]
                    end = self.rows[i+1]

                    while (j*10 < end):
                        self.start_rows[j] = i
                        j += 1

        else:
            assert (False)


    # forward pass to compute Q*K^T with the compressed KV cache
    def forward_fused_sparse(self, q, k):
        """
        q: current query vector
        k: key vector to append

        Performs the forward pass using the compressed key cache (and also appends a new key vector)
        """

        k = k.flatten()
        q = q.float()
        q = q.transpose(0,1).contiguous()
        k = k.float()

        # sparse packing kernel
        if self.include_sparse:
            outliers_rescaled = k.clone()

            if self.bits == 4:
                quant_cuda.vecquant4appendvecKsparse(
                    self.kcache,
                    self.lookup_table,
                    k,
                    outliers_rescaled,
                    self.outlier_threshold_lower,
                    self.outlier_threshold_upper,
                    self.klen - self.first_few_fp16
                )

            elif self.bits == 3:
                quant_cuda.vecquant3appendvecKsparse(
                    self.kcache,
                    self.lookup_table,
                    k,
                    outliers_rescaled,
                    self.outlier_threshold_lower,
                    self.outlier_threshold_upper,
                    self.klen - self.first_few_fp16
                )

            elif self.bits == 2:
                quant_cuda.vecquant2appendvecKsparse(
                    self.kcache,
                    self.lookup_table,
                    k,
                    outliers_rescaled,
                    self.outlier_threshold_lower,
                    self.outlier_threshold_upper,
                    self.klen - self.first_few_fp16
                )

            else:
                assert (False)

            # need to sort outliers -> which to keep (in order to have fixed size memory allocation)
            threshold_k = int(((1-self.sparsity_threshold) / 2) * self.hidden_size) + 1 # should be 21
            outliers_rescaled = outliers_rescaled.cpu()
            upper_outliers_tmp,upper_outlier_indices = torch.topk(outliers_rescaled,threshold_k)
            lower_outliers_tmp,lower_outlier_indices = torch.topk(outliers_rescaled,threshold_k,largest=False)
            upper_outliers_tmp = upper_outliers_tmp.cuda()
            lower_outliers_tmp = lower_outliers_tmp.cuda()
            upper_outlier_indices = upper_outlier_indices.cuda()
            lower_outlier_indices = lower_outlier_indices.cuda()

            # get actual outlier values (from k vector)
            upper_outliers = k[upper_outlier_indices]
            lower_outliers = k[lower_outlier_indices]

            # subtract offset -> need to subtract closest LUT element
            numvals = 2 ** self.bits

            # Q-Norm
            if self.norm:
                lut = self.lookup_table2.reshape(-1,numvals)
            else:
                lut = self.lookup_table.reshape(-1,numvals)

            # subtract outliers
            upper_outliers = upper_outliers - lut[upper_outlier_indices,numvals-1]
            lower_outliers = lower_outliers - lut[lower_outlier_indices,0]

            # make sure not storing more than needed
            # if normalized is between -1 and 1, then it isn't an outlier
            upper_outliers_zeros = upper_outliers_tmp <= 1
            lower_outliers_zeros = lower_outliers_tmp >= -1
            outlier_zeros_cat = torch.cat((upper_outliers_zeros, lower_outliers_zeros), dim=-1)

            # concatenate lower / upper outliers (to be appended to arrays)
            outlier_values_cat = torch.cat((upper_outliers, lower_outliers), dim=-1)
            outlier_indices_cat = torch.cat((upper_outlier_indices, lower_outlier_indices), dim=-1)
            outlier_indices,idx = outlier_indices_cat.sort()
            outlier_vals = outlier_values_cat[idx]

            # zero out ones that aren't actually outliers
            outlier_zeros_cat = outlier_zeros_cat[idx]
            outlier_vals[outlier_zeros_cat] = 0

            # need to append to outlier cache
            self.outliers[self.klen - self.first_few_fp16] = outlier_vals
            self.outlier_indices[self.klen - self.first_few_fp16] = outlier_indices

        else:
            if self.bits == 4:
                quant_cuda.vecquant4appendvecK(
                    self.kcache,
                    self.lookup_table,
                    k,
                    self.klen - self.first_few_fp16
                )

            elif self.bits == 3:
                quant_cuda.vecquant3appendvecK(
                    self.kcache,
                    self.lookup_table,
                    k,
                    self.klen - self.first_few_fp16
                )

            elif self.bits == 2:
                quant_cuda.vecquant2appendvecK(
                    self.kcache,
                    self.lookup_table,
                    k,
                    self.klen - self.first_few_fp16
                )

            else:
                assert (False)

        self.klen += 1
        mul = torch.zeros((q.shape[0], q.shape[1], self.klen - self.first_few_fp16), dtype=torch.float, device=q.device)

        if self.include_sparse:
            if self.bits == 4:
                quant_cuda.vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
                    q,
                    self.kcache,
                    mul,
                    self.lookup_table,
                    self.klen - self.first_few_fp16,
                    self.outliers,
                    self.outlier_indices,
                    self.rope_theta,
                    self.first_few_fp16
                )

            elif self.bits == 3:
                quant_cuda.vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
                    q,
                    self.kcache,
                    mul,
                    self.lookup_table,
                    self.klen - self.first_few_fp16,
                    self.outliers,
                    self.outlier_indices,
                    self.rope_theta,
                    self.first_few_fp16
                )

            elif self.bits == 2:
                if self.norm:
                    lookup_table = self.lookup_table2
                else:
                    lookup_table = self.lookup_table

                quant_cuda.vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
                    q,
                    self.kcache,
                    mul,
                    lookup_table, # self.lookup_table,
                    self.klen - self.first_few_fp16,
                    self.outliers,
                    self.outlier_indices,
                    self.rope_theta,
                    self.first_few_fp16
                )

            else:
                assert (False)

        else:
            if self.bits == 4:
                quant_cuda.vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
                    q,
                    self.kcache,
                    mul,
                    self.lookup_table,
                    self.klen - self.first_few_fp16,
                    self.rope_theta,
                    self.first_few_fp16
                )
            elif self.bits == 3:
                quant_cuda.vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
                    q,
                    self.kcache,
                    mul,
                    self.lookup_table,
                    self.klen - self.first_few_fp16,
                    self.rope_theta,
                    self.first_few_fp16
                )

            elif self.bits == 2:
                if self.norm:
                    lookup_table = self.lookup_table2
                else:
                    lookup_table = self.lookup_table

                quant_cuda.vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
                    q,
                    self.kcache,
                    mul,
                    lookup_table, # self.lookup_table,
                    self.klen - self.first_few_fp16,
                    self.rope_theta,
                    self.first_few_fp16
                )

            else:
                assert (False)

        mul = mul.transpose(0,1).contiguous()
        mul = mul.half()

        return mul

    # parallel packing function
    def parallel_pack(self, k):
        """
        k: key vectors to append

        Pack several key vectors in parallel
        """

        k = k.float().contiguous()

        # update klen
        self.klen += k.shape[-1]

        # sparse packing kernel
        if self.include_sparse:
            outliers_rescaled = k.clone()

            if self.bits == 4:
                quant_cuda.vecquant4appendvecKsparseParallel(
                    self.kcache,
                    self.lookup_table,
                    k,
                    outliers_rescaled,
                    self.outlier_threshold_lower,
                    self.outlier_threshold_upper,
                )

            elif self.bits == 3:
                quant_cuda.vecquant3appendvecKsparseParallel(
                    self.kcache,
                    self.lookup_table,
                    k,
                    outliers_rescaled,
                    self.outlier_threshold_lower,
                    self.outlier_threshold_upper,
                )

            elif self.bits == 2:
                quant_cuda.vecquant2appendvecKsparseParallel(
                    self.kcache,
                    self.lookup_table,
                    k,
                    outliers_rescaled,
                    self.outlier_threshold_lower,
                    self.outlier_threshold_upper,
                )

            else:
                assert (False)

            # reshape for topK
            outliers_rescaled = outliers_rescaled.reshape(-1,k.shape[-1]).transpose(0,1).contiguous()
            k = k.reshape(-1,k.shape[-1]).transpose(0,1).contiguous()

            # need to sort outliers -> which to keep (in order to have fixed size memory allocation)
            threshold_k = int(((1-self.sparsity_threshold) / 2) * self.hidden_size) + 1
            upper_outliers_tmp,upper_outlier_indices = torch.topk(outliers_rescaled,threshold_k,dim=-1)
            lower_outliers_tmp,lower_outlier_indices = torch.topk(outliers_rescaled,threshold_k,dim=-1,largest=False)

            # get actual outlier values (from k vector)
            upper_outliers = k[torch.arange(k.size(0)).unsqueeze(1), upper_outlier_indices]
            lower_outliers = k[torch.arange(k.size(0)).unsqueeze(1), lower_outlier_indices]

            # shift outliers by nearest LUT elements
            numvals = 2 ** self.bits

            # Q-Norm
            if self.norm:
                lut = self.lookup_table2.reshape(-1,numvals)
            else:
                lut = self.lookup_table.reshape(-1,numvals)

            # subtract outliers
            upper_outliers = upper_outliers - lut[upper_outlier_indices,numvals-1]
            lower_outliers = lower_outliers - lut[lower_outlier_indices,0]

            # make sure not storing more than needed
            # if normalized is between -1 and 1, then it isn't an outlier
            upper_outliers_zeros = upper_outliers_tmp <= 1
            lower_outliers_zeros = lower_outliers_tmp >= -1
            outlier_zeros_cat = torch.cat((upper_outliers_zeros, lower_outliers_zeros), dim=-1)

            # concatenate lower / upper outliers (to be appended to arrays)
            outlier_values_cat = torch.cat((upper_outliers, lower_outliers), dim=-1)
            outlier_indices_cat = torch.cat((upper_outlier_indices, lower_outlier_indices), dim=-1)
            outlier_indices,idx = outlier_indices_cat.sort(dim=-1)
            outlier_vals = outlier_values_cat[torch.arange(outlier_values_cat.size(0)).unsqueeze(1), idx]

            # zero out ones that aren't actually outliers
            outlier_zeros_cat = outlier_zeros_cat[torch.arange(outlier_zeros_cat.size(0)).unsqueeze(1), idx]
            outlier_vals[outlier_zeros_cat] = 0

            # need to append to outlier cache
            self.outliers[:self.klen] = outlier_vals
            self.outlier_indices[:self.klen] = outlier_indices

        else:
            assert (False)

# class for managing compressed values
class QuantV(nn.Module):
    def __init__(self, bits=2, hidden_size=4096, num_heads=32, max_position_embeddings=-1, include_sparse=False, sparsity_threshold=0.99, first_few_fp16=0):
        super().__init__()

        """
        bits: number of bits for quantization
        hidden_size: hidden size for the model
        num_heads: number of heads for the model
        max_position_embeddings: max sequence length
        include_sparse: whether to isolate outliers
        sparsity_threshold: what percentage of outliers to remove
        first_few_fp16: how many initial tokens to store in fp16

        This class manages the compressed value cache during generation
        """

        # model parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # quantization parameters
        self.bits = bits
        self.lut = None
        self.zeropoint = None

        # dense-and-sparse quantization parameters
        self.sparsity_threshold = sparsity_threshold
        self.include_sparse = include_sparse
        self.num_threads = -1

        # KV cache parameters
        self.max_len = max_position_embeddings
        self.lookup_table = torch.zeros((self.max_len, 2 ** self.bits), dtype=torch.float).cuda() # per-token LUT (in current implementation)
        self.vcache = torch.zeros((self.num_heads, (self.head_dim // 32) * self.bits, self.max_len), dtype=torch.int).cuda()
        self.vlen = 0

        # outlier params
        if self.include_sparse:
            # only supports 1% outliers
            self.outliers = torch.zeros((self.max_len,42), dtype=torch.float).cuda()
            self.outlier_indices = torch.zeros((self.max_len,42), dtype=torch.int).cuda()

        # first few in fp16
        self.first_few_fp16 = first_few_fp16

        # Q-Norm
        self.norm = False
        self.lookup_table2 = None

    def reset(self):
        """
        This class resets the compressed key cache
        """
        self.vlen = 0

        self.lookup_table[self.lookup_table!=0] = 0
        self.vcache[self.vcache!=0] = 0

        if self.include_sparse:
            self.outliers[self.outliers!=0] = 0
            self.outlier_indices[self.outlier_indices!=0] = 0

        if self.lookup_table2 is not None:
            self.lookup_table2[self.lookup_table2!=0] = 0


    def load_lookup_table(self, quantizer, include_sparse = True, sparsity_threshold = 0.99, norm=False): # need to load using calibration data
        """
        quantizer: calibrated outlier thresholds / LUT for the current layer
        include_sparse: whether to isolate outliers
        sparsity_threshold: what percentage of outliers to remove
        norm: whether to use Q-Norm

        Loads the LUT for the value cache
        """
        self.lut = torch.tensor(quantizer[2][0]).squeeze(-1).to(self.lookup_table.device).float()
        self.lut,_ = self.lut.sort()
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.norm = norm
        if self.norm:
            self.normscale = quantizer[3].cuda()
            self.normoffset = quantizer[4].cuda()
            self.lookup_table2 = torch.zeros((self.max_len, 2 ** self.bits), dtype=torch.float).cuda() # per-token LUT (in current implementation)
        else:
            self.normscale = None
            self.normoffset = None
            self.lookup_table2 = None


    def forward_fused_sparse(self, score, v, upper_outlier_vals, upper_outlier_indices, lower_outlier_vals, lower_outlier_indices):
        """
        score: current score vector
        v: value vector to append
        upper_outlier_vals: upper outlier values for new v token
        upper_outlier_indices: upper outlier indices for new v token
        lower_outlier_vals: lower outlier values for new v token
        lower_outlier_indices: lower outlier indices for new v token

        Performs the forward pass using the compressed value cache (and also appends a new value vector)
        """

        v_shape = v.shape

        score = score.float()
        v = v.flatten()

        if self.include_sparse:
            assert (upper_outlier_vals != None)
            assert (upper_outlier_indices != None)
            assert (lower_outlier_vals != None)
            assert (lower_outlier_indices != None)
            maxval = upper_outlier_vals[-1]
            minval = lower_outlier_vals[-1]
            upper_outlier_vals = upper_outlier_vals[:-1].contiguous() # when using topk(22)
            lower_outlier_vals = lower_outlier_vals[:-1].contiguous() # when using topk(22)
            upper_outlier_indices = upper_outlier_indices[:-1].contiguous()
            lower_outlier_indices = lower_outlier_indices[:-1].contiguous()
            offset = (maxval + minval) / 2
            sf = (maxval - minval) / 2
            outlier_threshold_lower = minval
            outlier_threshold_upper = maxval
        else:
            outlier_mask = None
            # compute max here otherwise
            sf, offset = compute_lut(
                v,
                bits=self.bits,
                qchannel=-1,
                zeropoint=True
            )

        # get per-token LUT
        v = v.float()
        lookup_table = torch.tensor(self.lut).float() * sf.float().item() + offset.float().item()
        self.lookup_table[self.vlen - self.first_few_fp16] = lookup_table
        #Q-Norm
        if self.norm:
            lookup_table2 = (torch.tensor(self.lut).float() * self.normscale + self.normoffset) * sf.float().item() + offset.float().item()
            self.lookup_table2[self.vlen - self.first_few_fp16] = lookup_table2

        score = score.transpose(0,1).contiguous()

        # packing kernel
        if self.include_sparse:

            if self.bits == 4:
                zeropoint = lookup_table[7]
                quant_cuda.vecquant4appendvecVsparse(
                    self.vcache,
                    self.lookup_table,
                    v,
                    zeropoint,
                    outlier_threshold_lower,
                    outlier_threshold_upper,
                    self.vlen - self.first_few_fp16
                )

            elif self.bits == 3:
                zeropoint = lookup_table[3]
                quant_cuda.vecquant3appendvecVsparse(
                    self.vcache,
                    self.lookup_table,
                    v,
                    zeropoint,
                    outlier_threshold_lower,
                    outlier_threshold_upper,
                    self.vlen - self.first_few_fp16
                )

            elif self.bits == 2:
                # for Q-Norm, get zeropoint for post-dequant
                if self.norm:
                    zeropoint = lookup_table2[1]
                else:
                    zeropoint = lookup_table[1]
                quant_cuda.vecquant2appendvecVsparse(
                    self.vcache,
                    self.lookup_table,
                    v,
                    zeropoint,
                    outlier_threshold_lower,
                    outlier_threshold_upper,
                    self.vlen - self.first_few_fp16
                )

            else:
                assert (False)

            # append sparse values
            outlier_values_cat = torch.cat((upper_outlier_vals, lower_outlier_vals), dim=-1) - zeropoint
            outlier_indices_cat = torch.cat((upper_outlier_indices, lower_outlier_indices), dim=-1)
            outlier_indices,idx = outlier_indices_cat.sort()
            outlier_vals = outlier_values_cat[idx]

            # append to value cache
            self.outliers[self.vlen - self.first_few_fp16] = outlier_vals
            self.outlier_indices[self.vlen - self.first_few_fp16] = outlier_indices

        else:
            if self.bits == 4:
                quant_cuda.vecquant4appendvecV(
                    self.vcache,
                    self.lookup_table,
                    v,
                    self.vlen - self.first_few_fp16
                )

            elif self.bits == 3:
                quant_cuda.vecquant3appendvecV(
                    self.vcache,
                    self.lookup_table,
                    v,
                    self.vlen - self.first_few_fp16
                )

            elif self.bits == 2:
                quant_cuda.vecquant2appendvecV(
                    self.vcache,
                    self.lookup_table,
                    v,
                    self.vlen - self.first_few_fp16
                )

            else:
                assert (False)

        self.vlen += 1

        #D+S matvec operation
        mul = torch.zeros((score.shape[0], score.shape[1], self.head_dim), dtype=torch.float, device=score.device)
        if self.include_sparse:

            if self.bits == 4:
                quant_cuda.vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
                    score,
                    self.vcache,
                    mul,
                    self.lookup_table,
                    self.vlen - self.first_few_fp16,
                    self.outliers,
                    self.outlier_indices
                )

            elif self.bits == 3:
                quant_cuda.vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
                    score,
                    self.vcache,
                    mul,
                    self.lookup_table,
                    self.vlen - self.first_few_fp16,
                    self.outliers,
                    self.outlier_indices
                )

            elif self.bits == 2:
                if self.norm:
                    lookup_table = self.lookup_table2
                else:
                    lookup_table = self.lookup_table

                quant_cuda.vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
                    score,
                    self.vcache,
                    mul,
                    lookup_table, # self.lookup_table
                    self.vlen - self.first_few_fp16,
                    self.outliers,
                    self.outlier_indices
                )

            else:
                assert (False)

        else:

            if self.bits == 4:
                quant_cuda.vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
                    score,
                    self.vcache,
                    mul,
                    self.lookup_table,
                    self.vlen - self.first_few_fp16
                )

            elif self.bits == 3:
                quant_cuda.vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
                    score,
                    self.vcache,
                    mul,
                    self.lookup_table,
                    self.vlen - self.first_few_fp16
                )

            elif self.bits == 2:
                if self.norm:
                    lookup_table = self.lookup_table2
                else:
                    lookup_table = self.lookup_table

                quant_cuda.vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
                    score,
                    self.vcache,
                    mul,
                    lookup_table, # self.lookup_table
                    self.vlen - self.first_few_fp16
                )

            else:
                assert (False)

        mul = mul.transpose(0,1).contiguous()
        mul = mul.half()
        return mul

    def parallel_pack(self, v, upper_outlier_vals, upper_outlier_indices, lower_outlier_vals, lower_outlier_indices):
        """
        v: value vectors to append
        upper_outlier_vals: upper outlier values for new v tokens
        upper_outlier_indices: upper outlier indices for new v tokens
        lower_outlier_vals: lower outlier values for new v tokens
        lower_outlier_indices: lower outlier indices for new v tokens

        Pack several value vectors in parallel
        """

        v = v.float()
        self.vlen += v.shape[-1]

        if self.include_sparse:
            assert (upper_outlier_vals != None)
            assert (upper_outlier_indices != None)
            assert (lower_outlier_vals != None)
            assert (lower_outlier_indices != None)
            maxval = upper_outlier_vals[:,-1].contiguous()
            minval = lower_outlier_vals[:,-1].contiguous()
            upper_outlier_vals = upper_outlier_vals[:,:-1].contiguous() # when using topk(22)
            lower_outlier_vals = lower_outlier_vals[:,:-1].contiguous() # when using topk(22)
            upper_outlier_indices = upper_outlier_indices[:,:-1].contiguous()
            lower_outlier_indices = lower_outlier_indices[:,:-1].contiguous()
            offset = (maxval + minval) / 2
            sf = (maxval - minval) / 2
            outlier_threshold_lower = minval
            outlier_threshold_upper = maxval
        else:
            assert (False)

        # get per-token LUT
        lookup_table = torch.tensor(self.lut).float().unsqueeze(0) * sf.float().unsqueeze(-1) + offset.float().unsqueeze(-1)
        self.lookup_table[:self.vlen] = lookup_table

        # packing kernel
        if self.include_sparse:
            self.vcache = self.vcache.contiguous()
            v = v.contiguous()

            if self.bits == 4:
                quant_cuda.vecquant4appendvecVsparseParallel(
                    self.vcache,
                    self.lookup_table,
                    v,
                    outlier_threshold_lower,
                    outlier_threshold_upper
                )
                zpt = 7

            elif self.bits == 3:
                quant_cuda.vecquant3appendvecVsparseParallel(
                    self.vcache,
                    self.lookup_table,
                    v,
                    outlier_threshold_lower,
                    outlier_threshold_upper
                )
                zpt = 3

            elif self.bits == 2:
                quant_cuda.vecquant2appendvecVsparseParallel(
                    self.vcache,
                    self.lookup_table,
                    v,
                    outlier_threshold_lower,
                    outlier_threshold_upper
                )
                zpt = 1

            else:
                assert (False)

            # for Q-Norm, get zeropoint for post-dequant
            if self.norm:
                lookup_table = self.lookup_table2
            else:
                lookup_table = self.lookup_table

            # append sparse values - outliers are (5,21)
            outlier_values_cat = torch.cat((upper_outlier_vals, lower_outlier_vals), dim=-1) - lookup_table[:self.vlen,zpt].unsqueeze(-1)
            outlier_indices_cat = torch.cat((upper_outlier_indices, lower_outlier_indices), dim=-1)
            outlier_indices,idx = outlier_indices_cat.sort()
            outlier_vals = outlier_values_cat[torch.arange(outlier_values_cat.size(0)).unsqueeze(1), idx]

            # need to append to outlier cache
            self.outliers[:self.vlen] = outlier_vals
            self.outlier_indices[:self.vlen] = outlier_indices

        else:
            assert (False)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        assert (self.num_key_value_groups == 1) # kernels don't yet support GQA
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.device = None

        # avoid caching cos / sin for RoPE
        self.dynamicrope = config.dynamicrope
        self._init_rope()

        # argument to use original sparsity configuration
        self.use_orig_sparse = config.use_orig_sparse

        # argument to use attention sink method
        self.first_few_fp16 = config.first_few_fp16

        # set max seqlen for KV cache
        if config.maxseqlen > -1:
            maxseqlen = config.maxseqlen
        else:
            maxseqlen = self.max_position_embeddings

        # load other arguments
        self.abits = config.abits
        self.include_sparse = config.include_sparse

        # arguments to initialize the KV cache are in the load_lookup_table functions
        self.kcache = QuantK(
            bits=self.abits,
            include_sparse=self.include_sparse,
            hidden_size=self.hidden_size,
            max_position_embeddings=maxseqlen,
            rope_theta=self.rope_theta,
            use_orig_sparse=self.use_orig_sparse,
            first_few_fp16=self.first_few_fp16
        )
        self.vcache = QuantV(
            bits=self.abits,
            include_sparse=self.include_sparse,
            hidden_size=self.hidden_size,
            max_position_embeddings=maxseqlen,
            first_few_fp16=self.first_few_fp16
        )

        # fp16 caches
        if self.first_few_fp16 > 0:
            self.kcache_fp16 = torch.zeros((1, self.num_heads, self.head_dim, self.first_few_fp16), dtype=torch.half).cuda()
            self.vcache_fp16 = torch.zeros((1, self.num_heads, self.first_few_fp16, self.head_dim), dtype=torch.half).cuda()

    def _init_rope(self):
        if self.config.rope_scaling is None and self.dynamicrope:
            self.rotary_emb = LlamaRotaryEmbeddingDynamic(
                self.head_dim,
                base=self.rope_theta,
                device="cpu"
            )
        elif self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
                device="cpu"
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    device="cpu"
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    device="cpu"
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        past_key_values_length_inp=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        assert(bsz == 1) # only supports BS=1 for now

        # topk offloading to CPU
        if self.kcache.include_sparse and not (q_len > 1 and self.kcache.klen == 0):
            s2 = torch.cuda.Stream(device="cuda:0")
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = value_states.flatten().float()
            s2.wait_stream(torch.cuda.default_stream(torch.device('cuda:0'))) # needed for correct execution

            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            with torch.cuda.stream(s2):
                v = v.cpu() # asynchronous mem copy, CPU can now run ahead
                threshold_k = int(((1-self.kcache.sparsity_threshold) / 2) * self.hidden_size) + 2
                upper_outlier_vals,upper_outlier_indices = torch.topk(v,threshold_k)
                lower_outlier_vals,lower_outlier_indices = torch.topk(v,threshold_k,largest=False)
                upper_outlier_vals = upper_outlier_vals.cuda()
                lower_outlier_vals = lower_outlier_vals.cuda()
                upper_outlier_indices = upper_outlier_indices.cuda()
                lower_outlier_indices = lower_outlier_indices.cuda()

        elif self.kcache.include_sparse:
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            v = value_states.float().reshape(q_len, self.num_key_value_heads*self.head_dim)
            value_states = value_states.transpose(1, 2)
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # block topK on GPU
            threshold_k = int(((1-self.kcache.sparsity_threshold) / 2) * self.hidden_size) + 2
            upper_outlier_vals,upper_outlier_indices = torch.topk(v,threshold_k,dim=-1)
            lower_outlier_vals,lower_outlier_indices = torch.topk(v,threshold_k,dim=-1,largest=False)

        else:
            upper_outlier_vals = None
            lower_outlier_vals = None
            upper_outlier_indices = None
            lower_outlier_indices = None
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.half()
        value_states = value_states.half()

        if self.kcache.klen == 0:
            kv_seq_len = key_states.shape[-2]
        else:
            kv_seq_len = self.kcache.klen + 1

        # dynamically compute cos / sin for RoPE
        if self.dynamicrope:
            start_pos = self.kcache.klen
            end_pos = self.kcache.klen + q_len
            cos, sin = self.rotary_emb(value_states, start_pos, end_pos)
            query_states = apply_rotary_pos_emb_query_dynamic(query_states, cos, sin, position_ids)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states = apply_rotary_pos_emb_query(query_states, cos, sin, position_ids)

        if q_len > 1 and self.kcache.klen == 0:

            # whether to use dynamic or cached sin/cos
            if self.dynamicrope:
                key_states_rope = apply_rotary_pos_emb_query_dynamic(key_states, cos, sin, position_ids)
            else:
                key_states_rope = apply_rotary_pos_emb_query(key_states, cos, sin, position_ids)

            # parallel version (not using quantized KV cache)
            key_states_rope = key_states_rope.transpose(2, 3)
            attn_weights = torch.matmul(query_states, key_states_rope) / math.sqrt(self.head_dim)

            # support for keeping first few tokens in fp16
            if self.first_few_fp16 > 0:
                if q_len > self.first_few_fp16:
                    # parallel append
                    key_states = key_states[0,:,self.first_few_fp16:,:].transpose(1, 2)
                    if self.use_orig_sparse:
                        self.kcache.parallel_pack_orig(key_states)
                    else:
                        self.kcache.parallel_pack(key_states)

                    # create fp16 k cache
                    self.kcache_fp16[:,:,:,:] = key_states_rope[:,:,:,:self.first_few_fp16]
                    self.kcache.klen += self.first_few_fp16

                else:
                    # only initialize fp16 kcache
                    self.kcache_fp16[:,:,:,:q_len] = key_states_rope
                    self.kcache.klen += q_len

            else: # no fp16 kcache

                # parallel append
                key_states = key_states[0,:,:,:].transpose(1, 2)
                if self.use_orig_sparse:
                    self.kcache.parallel_pack_orig(key_states)
                else:
                    self.kcache.parallel_pack(key_states)
        else:

            if self.kcache.klen < self.first_few_fp16:
                # whether to use dynamic or cached sin/cos
                if self.dynamicrope:
                    key_states_rope = apply_rotary_pos_emb_query_dynamic(key_states, cos, sin, position_ids)
                else:
                    key_states_rope = apply_rotary_pos_emb_query(key_states, cos, sin, position_ids)
                key_states_rope = key_states_rope.transpose(2, 3)

                # append to fp16 k cache
                self.kcache_fp16[:,:,:,self.kcache.klen] = key_states_rope.squeeze(-1)
                self.kcache.klen += 1

                # perform multiplication
                ktensor = self.kcache_fp16[:,:,:,:self.kcache.klen]
                attn_weights = torch.matmul(query_states, ktensor) / math.sqrt(self.head_dim)

            elif self.first_few_fp16 > 0:
                # compute fp16 kcache
                attn_weights_fp16 = torch.matmul(query_states, self.kcache_fp16) / math.sqrt(self.head_dim)

                # fused forward pass
                query_states = query_states[0,:,:,:]
                if self.use_orig_sparse:
                    attn_weights = self.kcache.forward_fused_sparse_orig(query_states, key_states)
                else:
                    attn_weights = self.kcache.forward_fused_sparse(query_states, key_states)
                attn_weights = attn_weights.unsqueeze(0)
                attn_weights = attn_weights / math.sqrt(self.head_dim)

                # append fp16 weights to start
                attn_weights = torch.cat((attn_weights_fp16, attn_weights), dim=-1)

            else:
                query_states = query_states[0,:,:,:]

                # fused forward pass
                if self.use_orig_sparse:
                    attn_weights = self.kcache.forward_fused_sparse_orig(query_states, key_states)
                else:
                    attn_weights = self.kcache.forward_fused_sparse(query_states, key_states)
                attn_weights = attn_weights.unsqueeze(0)
                attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # fused forward pass
        if q_len > 1  and self.vcache.vlen == 0:
            # parallel version
            attn_output = torch.matmul(attn_weights, value_states)

            # parallel append
            if self.first_few_fp16 > 0:

                if q_len > self.first_few_fp16:

                    # create fp16 vcache
                    self.vcache_fp16[:,:,:,:] = value_states[:,:,:self.first_few_fp16,:]

                    # parallel append
                    value_states = value_states[0,:,self.first_few_fp16:,:].transpose(1, 2)
                    self.vcache.parallel_pack(value_states, upper_outlier_vals[self.first_few_fp16:,:], upper_outlier_indices[self.first_few_fp16:,:], lower_outlier_vals[self.first_few_fp16:,:], lower_outlier_indices[self.first_few_fp16:,:])

                    # create fp16 k cache
                    self.vcache.vlen += self.first_few_fp16

                else:
                    # only initialize fp16 kcache
                    self.vcache_fp16[:,:,:q_len,:] = value_states
                    self.vcache.vlen += q_len

            else:
                value_states = value_states[0,:,:,:].transpose(1, 2)
                self.vcache.parallel_pack(value_states, upper_outlier_vals, upper_outlier_indices, lower_outlier_vals, lower_outlier_indices)

        else:
            if self.vcache.vlen < self.first_few_fp16:
                self.vcache_fp16[:,:,self.vcache.vlen,:] = value_states.squeeze(2)
                self.vcache.vlen += 1
                attn_output = torch.matmul(attn_weights, self.vcache_fp16[:,:,:self.vcache.vlen,:])

            elif self.first_few_fp16 > 0:
                # compute fp16 vcache attention
                attn_output_fp16 = torch.matmul(attn_weights[:,:,:,:self.first_few_fp16], self.vcache_fp16)

                # compute quantized vcache attention
                attn_weights = attn_weights.squeeze(0)
                attn_output = self.vcache.forward_fused_sparse(attn_weights[:,:,self.first_few_fp16:], value_states, upper_outlier_vals, upper_outlier_indices, lower_outlier_vals, lower_outlier_indices)
                attn_output = attn_output.unsqueeze(0)

                # sum attention outputs
                attn_output += attn_output_fp16

            else:
                attn_weights = attn_weights.squeeze(0)
                attn_output = self.vcache.forward_fused_sparse(attn_weights, value_states, upper_outlier_vals, upper_outlier_indices, lower_outlier_vals, lower_outlier_indices)
                attn_output = attn_output.unsqueeze(0)


        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, None

class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.test = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        assert(bsz == 1) # only supports BS=1 for now

        if self.kcache.include_sparse and not (q_len > 1 and self.kcache.klen == 0):
            s2 = torch.cuda.Stream(device="cuda:0")
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = value_states.flatten().float()
            s2.wait_stream(torch.cuda.default_stream(torch.device('cuda:0'))) # needed for correct execution

            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            with torch.cuda.stream(s2):
                v = v.cpu() # asynchronous mem copy, CPU can now run ahead
                threshold_k = int(((1-self.kcache.sparsity_threshold) / 2) * self.hidden_size) + 2
                upper_outlier_vals,upper_outlier_indices = torch.topk(v,threshold_k)
                lower_outlier_vals,lower_outlier_indices = torch.topk(v,threshold_k,largest=False)
                upper_outlier_vals = upper_outlier_vals.cuda()
                lower_outlier_vals = lower_outlier_vals.cuda()
                upper_outlier_indices = upper_outlier_indices.cuda()
                lower_outlier_indices = lower_outlier_indices.cuda()

        elif self.kcache.include_sparse:
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            v = value_states.float().reshape(q_len, self.num_key_value_heads*self.head_dim)
            value_states = value_states.transpose(1, 2)
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # block topK on GPU
            threshold_k = int(((1-self.kcache.sparsity_threshold) / 2) * self.hidden_size) + 2
            upper_outlier_vals,upper_outlier_indices = torch.topk(v,threshold_k,dim=-1)
            lower_outlier_vals,lower_outlier_indices = torch.topk(v,threshold_k,dim=-1,largest=False)

        else:
            upper_outlier_vals = None
            lower_outlier_vals = None
            upper_outlier_indices = None
            lower_outlier_indices = None
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.half()
        value_states = value_states.half()

        if self.kcache.klen == 0:
            kv_seq_len = key_states.shape[-2]
        else:
            kv_seq_len = self.kcache.klen + 1

        # dynamically compute cos / sin for RoPE
        if self.dynamicrope:
            start_pos = self.kcache.klen
            end_pos = self.kcache.klen + q_len
            cos, sin = self.rotary_emb(value_states, start_pos, end_pos)
            query_states = apply_rotary_pos_emb_query_dynamic(query_states, cos, sin, position_ids)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states = apply_rotary_pos_emb_query(query_states, cos, sin, position_ids)

        if q_len > 1 and self.kcache.klen == 0:
            # flash attention
            if self.dynamicrope:
                key_states_rope = apply_rotary_pos_emb_query_dynamic(key_states, cos, sin, position_ids)
            else:
                key_states_rope = apply_rotary_pos_emb_query(key_states, cos, sin, position_ids)

            query_states_flash = query_states.transpose(1, 2)
            key_states_flash = key_states_rope.transpose(1, 2)
            value_states_flash = value_states.transpose(1, 2)
            attn_output = self._flash_attention_forward(
                query_states_flash, key_states_flash, value_states_flash, attention_mask, q_len, dropout=0.0
            )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

            # parallel append - K
            if self.first_few_fp16 > 0:
                key_states_rope = key_states_rope.transpose(2, 3)
                if q_len > self.first_few_fp16:
                    # parallel append
                    key_states = key_states[0,:,self.first_few_fp16:,:].transpose(1, 2)
                    if self.use_orig_sparse:
                        self.kcache.parallel_pack_orig(key_states)
                    else:
                        self.kcache.parallel_pack(key_states)

                    # create fp16 k cache
                    self.kcache_fp16[:,:,:,:] = key_states_rope[:,:,:,:self.first_few_fp16]
                    self.kcache.klen += self.first_few_fp16

                else:
                    # only initialize fp16 kcache
                    self.kcache_fp16[:,:,:,:q_len] = key_states_rope
                    self.kcache.klen += q_len

            else: # no fp16 kcache

                # parallel append
                key_states = key_states[0,:,:,:].transpose(1, 2)
                if self.use_orig_sparse:
                    self.kcache.parallel_pack_orig(key_states)
                else:
                    self.kcache.parallel_pack(key_states)

            # parallel append - V
            if self.first_few_fp16 > 0:

                if q_len > self.first_few_fp16:

                    # create fp16 vcache
                    self.vcache_fp16[:,:,:,:] = value_states[:,:,:self.first_few_fp16,:]

                    # parallel append
                    value_states = value_states[0,:,self.first_few_fp16:,:].transpose(1, 2)
                    self.vcache.parallel_pack(value_states, upper_outlier_vals[self.first_few_fp16:,:], upper_outlier_indices[self.first_few_fp16:,:], lower_outlier_vals[self.first_few_fp16:,:], lower_outlier_indices[self.first_few_fp16:,:])

                    # create fp16 k cache
                    self.vcache.vlen += self.first_few_fp16

                else:
                    # only initialize fp16 kcache
                    self.vcache_fp16[:,:,:q_len,:] = value_states
                    self.vcache.vlen += q_len

            else:
                value_states = value_states[0,:,:,:].transpose(1, 2)
                self.vcache.parallel_pack(value_states, upper_outlier_vals, upper_outlier_indices, lower_outlier_vals, lower_outlier_indices)

        else:

            # fused forward pass for query
            if self.kcache.klen < self.first_few_fp16:
                # whether to use dynamic or cached sin/cos
                if self.dynamicrope:
                    key_states_rope = apply_rotary_pos_emb_query_dynamic(key_states, cos, sin, position_ids)
                else:
                    key_states_rope = apply_rotary_pos_emb_query(key_states, cos, sin, position_ids)
                key_states_rope = key_states_rope.transpose(2, 3)

                # append to fp16 k cache
                self.kcache_fp16[:,:,:,self.kcache.klen] = key_states_rope.squeeze(-1)
                self.kcache.klen += 1

                # perform multiplication
                ktensor = self.kcache_fp16[:,:,:,:self.kcache.klen]
                attn_weights = torch.matmul(query_states, ktensor) / math.sqrt(self.head_dim)

            elif self.first_few_fp16 > 0:
                # compute fp16 kcache
                attn_weights_fp16 = torch.matmul(query_states, self.kcache_fp16) / math.sqrt(self.head_dim)

                # fused forward pass
                query_states = query_states[0,:,:,:]
                if self.use_orig_sparse:
                    attn_weights = self.kcache.forward_fused_sparse_orig(query_states, key_states)
                else:
                    attn_weights = self.kcache.forward_fused_sparse(query_states, key_states)
                attn_weights = attn_weights.unsqueeze(0)
                attn_weights = attn_weights / math.sqrt(self.head_dim)

                # append fp16 weights to start
                attn_weights = torch.cat((attn_weights_fp16, attn_weights), dim=-1)

            else:
                query_states = query_states[0,:,:,:]

                # fused forward pass
                if self.use_orig_sparse:
                    attn_weights = self.kcache.forward_fused_sparse_orig(query_states, key_states)
                else:
                    attn_weights = self.kcache.forward_fused_sparse(query_states, key_states)
                attn_weights = attn_weights.unsqueeze(0)
                attn_weights = attn_weights / math.sqrt(self.head_dim)

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

            # fused forward pass - V
            if self.vcache.vlen < self.first_few_fp16:
                self.vcache_fp16[:,:,self.vcache.vlen,:] = value_states.squeeze(2)
                self.vcache.vlen += 1
                attn_output = torch.matmul(attn_weights, self.vcache_fp16[:,:,:self.vcache.vlen,:])

            elif self.first_few_fp16 > 0:
                # compute fp16 vcache attention
                attn_output_fp16 = torch.matmul(attn_weights[:,:,:,:self.first_few_fp16], self.vcache_fp16)

                # compute quantized vcache attention
                attn_weights = attn_weights.squeeze(0)
                attn_output = self.vcache.forward_fused_sparse(attn_weights[:,:,self.first_few_fp16:], value_states, upper_outlier_vals, upper_outlier_indices, lower_outlier_vals, lower_outlier_indices)
                attn_output = attn_output.unsqueeze(0)

                # sum attention outputs
                attn_output += attn_output_fp16

            else:
                attn_weights = attn_weights.squeeze(0)
                attn_output = self.vcache.forward_fused_sparse(attn_weights, value_states, upper_outlier_vals, upper_outlier_indices, lower_outlier_vals, lower_outlier_indices)
                attn_output = attn_output.unsqueeze(0)

            # reshape attn output
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.device = None

    def set_device(self, device):
        self.device = device
        self.self_attn.device = device

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_key_values_length_inp=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_key_values_length_inp=past_key_values_length_inp,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self.num_linear_layers = 7  # q, k, v, o, gate, down, up

    def set_devices(self):
        num_visible_devices = torch.cuda.device_count()
        assert num_visible_devices > 0, "Must use at least one GPU"
        self.split_gpus = num_visible_devices > 1
        print(f"splitting into {num_visible_devices} GPUs")
        if not self.split_gpus:
            # print('max memory(MiB):', torch.cuda.memory_allocated() / 1024 /1024)
            self.cuda()
        else:
            # For larger model, we need to split the model into multiple GPUs
            # assign the embedding and norm onto the 1st devide
            self.embed_tokens.to(f"cuda:0")
            self.norm.to(f"cuda:0")

            # layers are divided into #(num GPUs) chunks
            self.split_indices = []
            prev_device = 0
            nums = len(self.layers) // num_visible_devices
            for i, layer in enumerate(self.layers):
                device = min(num_visible_devices - 1, i // nums)
                if prev_device != device:
                    self.split_indices.append(i)
                print(f"cuda:{device} for", i)
                layer.to(f"cuda:{device}")
                layer.set_device(device)
                prev_device = device

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values_length_inp = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if past_key_values_length_inp is not None:
            past_key_values_length = past_key_values_length_inp

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds
        # hidden_states.requires_grad_(True)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        device = 0
        for idx,decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Move activations to the next device at the split points
            if self.split_gpus and idx in self.split_indices:
                device += 1
                hidden_states = hidden_states.to(f"cuda:{device}")
                attention_mask = attention_mask.to(f"cuda:{device}")

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                # hidden_states.requires_grad_(True)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    past_key_values_length_inp=past_key_values_length,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            # Move activations back to the 1st device at the end
            if self.split_gpus and idx == len(self.layers) - 1:
                hidden_states = hidden_states.to("cuda:0")
                attention_mask = attention_mask.to("cuda:0")

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values_length_inp=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values_length_inp=past_key_values_length_inp,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, kvquant=False, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if kvquant:
            position_ids = position_ids[:,-1:]
            # attention_mask = attention_mask[:,-1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
