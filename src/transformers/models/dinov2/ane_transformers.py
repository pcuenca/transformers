# Copied with adaptations from:
# https://github.com/apple/ml-stable-diffusion
# https://github.com/apple/ml-ane-transformers
# Please, check the corresponding licenses

import torch
import torch.nn as nn

from ...utils import logging

logger = logging.get_logger(__name__)

class ReferenceLayerNormANE(nn.Module):
    """ LayerNorm optimized for Apple Neural Engine (ANE) execution

    Note: This layer only supports normalization over the final dim. It expects `num_channels`
    as an argument and not `normalized_shape` which is used by `torch.nn.LayerNorm`.
    """

    def __init__(self,
                 num_channels,
                 clip_mag=None,
                 eps=1e-5,
                 elementwise_affine=True):
        """
        Args:
            num_channels:       Number of channels (C) where the expected input data format is BC1S. S stands for sequence length.
            clip_mag:           Optional float value to use for clamping the input range before layer norm is applied.
                                If specified, helps reduce risk of overflow.
            eps:                Small value to avoid dividing by zero
            elementwise_affine: If true, adds learnable channel-wise shift (bias) and scale (weight) parameters
        """
        super().__init__()
        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        self.expected_rank = len('BC1S')

        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inputs):
        input_rank = len(inputs.size())

        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        # Migrate the data format from BSC to BC1S (most conducive to ANE)
        if input_rank == 3 and inputs.size(2) == self.num_channels:
            inputs = inputs.transpose(1, 2).unsqueeze(2)
            input_rank = len(inputs.size())

        assert input_rank == self.expected_rank
        assert inputs.size(1) == self.num_channels

        if self.clip_mag is not None:
            inputs.clamp_(-self.clip_mag, self.clip_mag)

        channels_mean = inputs.mean(dim=1, keepdims=True)

        zero_mean = inputs - channels_mean

        zero_mean_sq = zero_mean * zero_mean

        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()

        out = zero_mean * denom

        if self.elementwise_affine:
            out = (out + self.bias.view(1, self.num_channels, 1, 1)
                   ) * self.weight.view(1, self.num_channels, 1, 1)

        return out
    
# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix +
               'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix +
                                                                  'weight']
    return state_dict

class LayerNormANE(ReferenceLayerNormANE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(
            correct_for_bias_scale_order_inversion)

def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """ Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
    """
    # List of substrings to match to convert from Linear to Conv2d
    LINEAR_TO_CONV = [
        "key.weight",
        "query.weight",
        "value.weight",
        "dense.weight",
        "fc1.weight",
        "fc2.weight",
    ]
    for k in state_dict:
        if any([x in k for x in LINEAR_TO_CONV]) and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k][:, :, None, None]

CHUNK_SIZE = 512

def split_einsum(q, k, v, mask, heads, dim_head):
    """ Attention Implementation backing AttentionImplementations.SPLIT_EINSUM

    - Implements https://machinelearning.apple.com/research/neural-engine-transformers
    - Recommended for ANE
    - Marginally slower on GPU
    """
    mh_q = [
        q[:, head_idx * dim_head:(head_idx + 1) *
          dim_head, :, :] for head_idx in range(heads)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    k = k.transpose(1, 3)
    mh_k = [
        k[:, :, :,
          head_idx * dim_head:(head_idx + 1) * dim_head]
        for head_idx in range(heads)
    ]  # (bs, max_seq_length, 1, dim_head) * heads

    mh_v = [
        v[:, head_idx * dim_head:(head_idx + 1) *
          dim_head, :, :] for head_idx in range(heads)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    attn_weights = [
        torch.einsum("bchq,bkhc->bkhq", [qi, ki]) * (dim_head**-0.5)
        for qi, ki in zip(mh_q, mh_k)
    ]  # (bs, max_seq_length, 1, max_seq_length) * heads

    if mask is not None:
        for head_idx in range(heads):
            attn_weights[head_idx] = attn_weights[head_idx] + mask

    attn_weights = [
        aw.softmax(dim=1) for aw in attn_weights
    ]  # (bs, max_seq_length, 1, max_seq_length) * heads
    attn = [
        torch.einsum("bkhq,bchk->bchq", wi, vi)
        for wi, vi in zip(attn_weights, mh_v)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    attn = torch.cat(attn, dim=1)  # (bs, dim, 1, max_seq_length)
    return attn

def split_einsum_v2(q, k, v, mask, heads, dim_head):
    """ Attention Implementation backing AttentionImplementations.SPLIT_EINSUM_V2

    - Implements https://machinelearning.apple.com/research/neural-engine-transformers
    - Recommended for ANE
    - Marginally slower on GPU
    - Chunks the query sequence to avoid large intermediate tensors and improves ANE performance
    """
    query_seq_length = q.size(3)
    num_chunks = query_seq_length // CHUNK_SIZE

    needs_padding = query_seq_length % CHUNK_SIZE != 0
    if needs_padding:
        num_chunks += 1
        pad_length = num_chunks * CHUNK_SIZE - query_seq_length
        z = torch.zeros(q.shape[:-1] + (pad_length,), dtype=q.dtype)
        q = torch.cat((q, z), dim=-1)
        k = torch.cat((k, z), dim=-1)
        v = torch.cat((v, z), dim=-1)
    
    if num_chunks == 0:
        logger.info(
            "AttentionImplementations.SPLIT_EINSUM_V2: query sequence too short to chunk "
            f"({query_seq_length}<{CHUNK_SIZE}), fall back to AttentionImplementations.SPLIT_EINSUM (safe to ignore)")
        return split_einsum(q, k, v, mask, heads, dim_head)
    
    logger.info(
        "AttentionImplementations.SPLIT_EINSUM_V2: Splitting query sequence length of "
        f"{query_seq_length} into {num_chunks} chunks")

    mh_q = [
        q[:, head_idx * dim_head:(head_idx + 1) *
          dim_head, :, :] for head_idx in range(heads)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    # Chunk the query sequence for each head
    mh_q_chunked = [
        [h_q[..., chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE] for chunk_idx in range(num_chunks)]
        for h_q in mh_q
    ]  # ((bs, dim_head, 1, QUERY_SEQ_CHUNK_SIZE) * num_chunks) * heads

    k = k.transpose(1, 3)
    mh_k = [
        k[:, :, :,
          head_idx * dim_head:(head_idx + 1) * dim_head]
        for head_idx in range(heads)
    ]  # (bs, max_seq_length, 1, dim_head) * heads

    mh_v = [
        v[:, head_idx * dim_head:(head_idx + 1) *
          dim_head, :, :] for head_idx in range(heads)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    attn_weights = [
        [
            torch.einsum("bchq,bkhc->bkhq", [qi_chunk, ki]) * (dim_head**-0.5)
            for qi_chunk in h_q_chunked
        ] for h_q_chunked, ki in zip(mh_q_chunked, mh_k)
    ]  # ((bs, max_seq_length, 1, chunk_size) * num_chunks) * heads

    attn_weights = [
        [aw_chunk.softmax(dim=1) for aw_chunk in aw_chunked]
        for aw_chunked in attn_weights
    ]  # ((bs, max_seq_length, 1, chunk_size) * num_chunks) * heads

    attn = [
        [
            torch.einsum("bkhq,bchk->bchq", wi_chunk, vi)
            for wi_chunk in wi_chunked
        ] for wi_chunked, vi in zip(attn_weights, mh_v)
    ]  # ((bs, dim_head, 1, chunk_size) * num_chunks) * heads

    attn = torch.cat([
        torch.cat(attn_chunked, dim=3) for attn_chunked in attn
    ], dim=1)  # (bs, dim, 1, max_seq_length)

    if needs_padding:
        attn = attn[..., :query_seq_length]

    return attn
