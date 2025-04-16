# Copyright (c) 2024-2025, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia License.
# To view a copy of this license, visit
# https://github.com/NVlabs/OmniDrive/blob/main/LICENSE
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
from mmdet.models.utils.builder import TRANSFORMER
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as cp
from .attention import FlashMHA
import warnings


class MultiHeadAttentionwDropout(nn.Module):

    def __init__(self, embed_dims, num_heads, dropout, flash_attn):
        super().__init__()
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._dropout = dropout
        self.flash_attn = flash_attn
        if flash_attn:
            self.attn = FlashMHA(embed_dims, num_heads, dropout, dtype=torch.float16, device='cuda')
        else:
            self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.proj_drop = nn.Dropout(dropout)

        self._count = 0

    def forward(self, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_mask):
        """ Forward function for multi-head attention
        Args:
            query: shape [num_query, batch_size, embed_dims]
            key: shape [num_key, batch_size, embed_dims]
            value: shape [num_value, batch_size, embed_dims]
            query_pos: shape [num_query, batch_size, embed_dims]
            key_pos: shape [num_key, batch_size, embed_dims]
        """
        if query_pos is not None:
            query_w_pos = query + query_pos
        else:
            query_w_pos = query
        if key_pos is not None:
            key_w_pos = key + key_pos
        else:
            key_w_pos = key

        if self.flash_attn:
            out, attn = self.attn(query_w_pos, key_w_pos, value)
        else:
            out, attn = self.attn(query_w_pos, key_w_pos, value, attn_mask=attn_mask)
        out = self.proj_drop(out)

        return out + query, attn


# Feed-forward Network
class FFN(nn.Module):

    def __init__(self, embed_dims, feedforward_dims, dropout):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dims, embed_dims),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self._layers(x) + x

# We use the same decoder as StreamPETR
# which consists of:
# 1. Multi-head Self-attention 
# 2. LayerNorm
# 3. Multi-head Cross-attention
# 4. LayerNorm
# 5. Feed-forward Network
# 6. LayerNorm

class PETRTransformerDecoderLayer(nn.Module):

    def __init__(self, 
                 embed_dims, 
                 num_heads, 
                 feedforward_dims, 
                 dropout=0.1,
                 flash_attn=True,
                 ):
        super().__init__()
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._feedforward_dims = feedforward_dims

        self.transformer_layers = nn.ModuleList()
        # 1. Multi-head Self-attention
        self.transformer_layers.append(
            MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, False)
        )
        # 2. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        # 3. Multi-head Cross-attention
        self.transformer_layers.append(
            MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, flash_attn)
        )
        # 4. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        # 5. Feed-forward Network
        self.transformer_layers.append(
            FFN(embed_dims, feedforward_dims, dropout))
        # 6. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        

    def forward(self, query, key, query_pos, key_pos, attn_mask, temp_memory=None, temp_pos=None):
        """ Forward function for transformer decoder layer
        Args:
            query: shape [num_query, batch_size, embed_dims]
            key: shape [num_key, batch_size, embed_dims]
            value: shape [num_value, batch_size, embed_dims]
            query_pos: shape [num_query, batch_size, embed_dims]
            key_pos: shape [num_key, batch_size, embed_dims]
            attn_mask: shape [batch_size, num_query, num_key]
        """
        # TODO: maybe we shouldn't use hard-code layer here
        # TODO: add temporal query here
        # 1. Multi-head Self-attention (between queries)
        if temp_memory is not None:
            temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
            temp_pos = torch.cat([query_pos, temp_pos], dim=1)
        else:
            temp_key = temp_value = query
            temp_pos = query_pos

        query, attn0 = self.transformer_layers[0](query, temp_key, temp_value, query_pos, temp_pos, attn_mask=attn_mask)
        # 2. LayerNorm
        query = self.transformer_layers[1](query)
        # 3. Multi-head Cross-attention (between queries and keys)
        query, attn1 = self.transformer_layers[2](query, key, key, query_pos, key_pos, attn_mask=None)
        # 4. LayerNorm
        query = self.transformer_layers[3](query)
        # 5. Feed-forward Network
        query = self.transformer_layers[4](query)
        # 6. LayerNorm
        query = self.transformer_layers[5](query)

        return query

@TRANSFORMER.register_module()
class PETRTransformerDecoder(nn.Module):
    def __init__(self, 
                 num_layers,
                 embed_dims,
                 num_heads,
                 feedforward_dims,
                 dropout,
                 with_cp=False,
                 flash_attn=True):
        super().__init__()
        self._num_layers = num_layers
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._feedforward_dims = feedforward_dims
        self._dropout = dropout
        self._with_cp = with_cp
        self._layers = nn.ModuleList()
        for _ in range(num_layers):
            self._layers.append(
                PETRTransformerDecoderLayer(
                    embed_dims,
                    num_heads,
                    feedforward_dims,
                    dropout,
                    flash_attn=flash_attn,
                )
            )

    def forward(self, query, key, query_pos=None, key_pos=None, attn_mask=None, temp_memory=None, temp_pos=None):
        """ Forward function for transformer decoder
        Args:
            query: shape [num_query, batch_size, embed_dims]
            key: shape [num_key, batch_size, embed_dims]
            value: shape [num_value, batch_size, embed_dims]
            query_pos: shape [num_query, batch_size, embed_dims]
            key_pos: shape [num_key, batch_size, embed_dims]
            attn_mask: shape [batch_size, num_query, num_key]
        """
        return_val = []
        for layer in self._layers:
            if self._with_cp and self.training:
                query = cp.checkpoint(layer, query, key, query_pos, key_pos, attn_mask, temp_memory, temp_pos)
            else:
                query = layer(query, key, query_pos, key_pos, attn_mask, temp_memory, temp_pos)
            return_val.append(query)
        return torch.stack(return_val, dim=0)

@TRANSFORMER.register_module()
class PETRTemporalTransformer(nn.Module):
    def __init__(self, 
                 input_dimension,
                 output_dimension,
                 query_number=32,
                 num_layers=6,
                 embed_dims=256,
                 num_heads=8,
                 feedforward_dims=2048,
                 dropout=0.0,
                 with_cp=False,
                 flash_attn=True,):
        
        super().__init__()
        assert output_dimension % embed_dims == 0, "output dimension (language model) must be divisible by the embed dimension"

        # self.query_embedding = nn.Embedding(query_number, embed_dims) # learnable queries
        
        self.input_dimension = embed_dims
        self.output_dimension = output_dimension

        self.query_decoder = PETRTransformerDecoder(
                                            num_layers=num_layers,
                                            embed_dims=embed_dims,
                                            num_heads=num_heads,
                                            feedforward_dims=feedforward_dims,
                                            dropout=dropout,
                                            with_cp=with_cp,
                                            flash_attn=flash_attn)


        # convert with linear layer
        # self.output_projection = nn.Linear(embed_dims, output_dimension)
        # according to shihao, merge the multiple tokens into one to expand dimension

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)


    def forward(self, query, key, query_pos=None, key_pos=None, attn_mask=None, temp_memory=None, temp_pos=None):
        """ Forward function for transformer decoder
        Args:
            vision_tokens: shape [bs, sequence_length, embed_dims]
        Output:
            re-sampled token sequences: [bs, num_queries, embed_dims]
        """

        out = self.query_decoder(query, key, query_pos, key_pos, attn_mask, temp_memory, temp_pos) # feature from the last layer

        return out
