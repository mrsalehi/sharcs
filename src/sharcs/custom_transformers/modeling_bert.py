# coding=utf-8
# 2020.08.28 - (1) Changed fixed sized Transformer layer to support adptive width; and
#              (2) Added modules to rewire the BERT model according to the importance of attention
#                  heads and neurons in the intermediate layer of Feed-forward Network.
#              Huawei Technologies Co., Ltd <houlu3@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys
sys.path.append("..")

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss, BCEWithLogitsLoss

from .modeling_utils import PreTrainedModel
from .configuration_bert import BertConfig

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


ACT2FN = {
    "gelu": gelu, 
    "relu": torch.nn.functional.relu,
    "gelu_new": NewGELUActivation
}


BertLayerNorm = torch.nn.LayerNorm


class AdaptiveLayerNorm(torch.nn.LayerNorm):
    def __init__(self, width_mult, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None, original_weight=None, original_bias=None):
        self.dim = int(width_mult * normalized_shape)
        super().__init__(
            normalized_shape=self.dim, 
            eps=eps, 
            elementwise_affine=elementwise_affine,
            bias=True,
            device=device, 
            dtype=dtype, 
        )
        
        if original_bias is not None and original_weight is not None:
            self.weight.data[:self.dim].copy_(original_weight[:self.dim])
            self.bias.data[:self.dim].copy_(original_bias[:self.dim])

        self.mode = "train"
    

def round_to_nearest(input_size, width_mult, num_heads, min_value=1):
    new_width_mult = round(num_heads * width_mult)*1.0/num_heads
    input_size = int(new_width_mult * input_size)
    new_input_size = max(min_value, input_size)
    return new_input_size


class AdaptiveLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_heads, layer_type, bias=True, adaptive_dim=[True, True]):
        super(AdaptiveLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.num_heads = num_heads
        self.width_mult = 1.
        self.adaptive_dim = adaptive_dim
        self.mode = "train"
        self.layer_type = layer_type

    def forward(self, input):
        if self.adaptive_dim[0]:
            self.in_features = round_to_nearest(self.in_features_max, self.width_mult, self.num_heads)
        if self.adaptive_dim[1]:
            self.out_features = round_to_nearest(self.out_features_max, self.width_mult, self.num_heads)
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mode = "train"

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, qkv_adaptive_dim=[False, True], layer_type="normal"):
        super(BertSelfAttention, self).__init__()
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.orig_num_attention_heads = config.num_attention_heads
        if getattr(config, "attention_head_size", None) is None:
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        else:
            self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # dense layer for adaptive width
        self.query = AdaptiveLinear(config.hidden_size, self.all_head_size, 
                                config.num_attention_heads, adaptive_dim=qkv_adaptive_dim, 
                                layer_type=layer_type)
        self.key = AdaptiveLinear(config.hidden_size, self.all_head_size, 
                              config.num_attention_heads, adaptive_dim=qkv_adaptive_dim,
                              layer_type=layer_type)
        self.value = AdaptiveLinear(config.hidden_size, self.all_head_size, 
                                config.num_attention_heads, adaptive_dim=qkv_adaptive_dim,
                                layer_type=layer_type)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.mode = "train"
        self.layer_type = layer_type

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        self.num_attention_heads = round(self.orig_num_attention_heads * self.query.width_mult)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # output attention scores when needed
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config, layer_type="normal"):
        super(BertSelfOutput, self).__init__()
        if getattr(config, "attention_head_size", None) is not None:
            self_attn_output_dim = config.attention_head_size * config.num_attention_heads 
        else:
            self_attn_output_dim = config.hidden_size
        if layer_type == "adaptive":
            self.dense = AdaptiveLinear(self_attn_output_dim, config.hidden_size,
                                    config.num_attention_heads, adaptive_dim=[True, True], layer_type=layer_type)
        elif layer_type == "dyna":
            self.dense = AdaptiveLinear(self_attn_output_dim, config.hidden_size,
                                    config.num_attention_heads, adaptive_dim=[True, False], layer_type=layer_type)
        elif layer_type in {"normal", "router", "last_non_adaptive_without_router"}:
            self.dense = AdaptiveLinear(self_attn_output_dim, config.hidden_size,
                                    config.num_attention_heads, adaptive_dim=[False, False], layer_type=layer_type)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_type = layer_type
        self.mode = "train"

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        D = hidden_states.size(-1)
        width_mult = D / 768

        if self.layer_type == "adaptive":
            assert D == input_tensor.size(-1)
            layer_norm = getattr(self, 'LayerNorm_{}'.format(str(width_mult).replace(".", "")))
            # hidden_states = layer_norm(hidden_states + input_tensor[..., :dim])
            hidden_states = layer_norm(hidden_states + input_tensor)
        elif self.layer_type in {"normal", "dyna", "router"}:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, layer_type="normal"):
        super(BertAttention, self).__init__()
        if layer_type == "adaptive": 
            self.self = BertSelfAttention(config, qkv_adaptive_dim=[True, True], layer_type=layer_type)
        elif layer_type == "dyna":
            self.self = BertSelfAttention(config, qkv_adaptive_dim=[False, True], layer_type=layer_type)
        elif layer_type in {"normal", "router", "last_non_adaptive_without_router"}:
            self.self = BertSelfAttention(config, qkv_adaptive_dim=[False, False], layer_type=layer_type)

        self.output = BertSelfOutput(config, layer_type=layer_type)
        self.mode = "train" 
        self.layer_type = layer_type

    def reorder_heads(self, idx):
        n, a = self.self.num_attention_heads, self.self.attention_head_size
        index = torch.arange(n*a).reshape(n, a)[idx].view(-1).contiguous().long()

        def reorder_head_matrix(linearLayer, index, dim=0):
            index = index.to(linearLayer.weight.device)
            W = linearLayer.weight.index_select(dim, index).clone().detach()
            if linearLayer.bias is not None:
                if dim == 1:
                    b = linearLayer.bias.clone().detach()
                else:
                    b = linearLayer.bias[index].clone().detach()

            linearLayer.weight.requires_grad = False
            linearLayer.weight.copy_(W.contiguous())
            linearLayer.weight.requires_grad = True
            if linearLayer.bias is not None:
                linearLayer.bias.requires_grad = False
                linearLayer.bias.copy_(b.contiguous())
                linearLayer.bias.requires_grad = True

        reorder_head_matrix(self.self.query, index)
        reorder_head_matrix(self.self.key, index)
        reorder_head_matrix(self.self.value, index)
        reorder_head_matrix(self.output.dense, index, dim=1)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config, layer_type="normal"):
        super(BertIntermediate, self).__init__()
 
        if layer_type in {"normal", "router", "last_non_adaptive_without_router"}:
            self.dense = AdaptiveLinear(config.hidden_size, config.intermediate_size,
                                    config.num_attention_heads, adaptive_dim=[False, False], layer_type=layer_type)        
        elif layer_type == "adaptive":
            self.dense = AdaptiveLinear(config.hidden_size, config.intermediate_size,
                                    config.num_attention_heads, adaptive_dim=[True, True], layer_type=layer_type)
        elif layer_type == "dyna":
            self.dense = AdaptiveLinear(config.hidden_size, config.intermediate_size,
                                    config.num_attention_heads, adaptive_dim=[False, True], layer_type=layer_type)

        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            if config.hidden_act == "gelu_new":
                self.intermediate_act_fn = ACT2FN[config.hidden_act]()
            else:
                self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.mode = "train"
        self.layer_type = layer_type

    def reorder_neurons(self, index, dim=0):
        index = index.to(self.dense.weight.device)
        W = self.dense.weight.index_select(dim, index).clone().detach()
        if self.dense.bias is not None:
            if dim == 1:
                b = self.dense.bias.clone().detach()
            else:
                b = self.dense.bias[index].clone().detach()
        self.dense.weight.requires_grad = False
        self.dense.weight.copy_(W.contiguous())
        self.dense.weight.requires_grad = True
        if self.dense.bias is not None:
            self.dense.bias.requires_grad = False
            self.dense.bias.copy_(b.contiguous())
            self.dense.bias.requires_grad = True

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config, layer_type="normal"):
        super(BertOutput, self).__init__()

        assert layer_type in {"normal", "dyna"}

        if layer_type == "dyna":
            self.dense = AdaptiveLinear(config.intermediate_size, config.hidden_size,
                                    config.num_attention_heads, adaptive_dim=[True, False],
                                    layer_type=layer_type)
        elif layer_type == "normal":
            self.dense = AdaptiveLinear(config.intermediate_size, config.hidden_size,
                                    config.num_attention_heads, adaptive_dim=[False, False],
                                    layer_type=layer_type)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps) 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.layer_type = layer_type
                    
    def reorder_neurons(self, index, dim=1):
        index = index.to(self.dense.weight.device)
        W = self.dense.weight.index_select(dim, index).clone().detach()
        if self.dense.bias is not None:
            if dim == 1:
                b = self.dense.bias.clone().detach()
            else:
                b = self.dense.bias[index].clone().detach()
        self.dense.weight.requires_grad = False
        self.dense.weight.copy_(W.contiguous())
        self.dense.weight.requires_grad = True
        if self.dense.bias is not None:
            self.dense.bias.requires_grad = False
            self.dense.bias.copy_(b.contiguous())
            self.dense.bias.requires_grad = True

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states        


class BertOutputWithRouter(nn.Module):
    def __init__(self, config, layer_type="router"):
        super(BertOutputWithRouter, self).__init__()
        # NOTE: the output of this layer is still 768 dimensional. The router will
        # take this 768d input and output a width multiplier.
        # In val mode, width multiplier will determine the width of the rest of the network
        # and in train mode, the width multiplier will be used to compute the loss.
        # In train mode, the width of the rest of the network will be enforced by labels.
        self.dense = AdaptiveLinear(config.intermediate_size, config.hidden_size,
                                config.num_attention_heads, adaptive_dim=[False, False], layer_type=layer_type)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size_router),
            nn.ReLU(),
            nn.Linear(config.hidden_size_router, len(config.width_mult_list))
            )
        assert layer_type == "router"
        self.layer_type = layer_type
        
        if config.weighted_dim_reduction:
            self.dim_reduction_weights = []
            self.width_mult_to_index = {}
            for it, width_multiplier in enumerate(config.width_mult_list):
                self.width_mult_to_index[width_multiplier] = it
                n_weights = int(1 / width_multiplier)
                self.dim_reduction_weights.append(nn.Parameter(torch.ones(n_weights, 1) * (1/n_weights), requires_grad=True))
            self.dim_reduction_weights = nn.ParameterList(self.dim_reduction_weights)

    def reorder_neurons(self, index, dim=1):
        index = index.to(self.dense.weight.device)
        W = self.dense.weight.index_select(dim, index).clone().detach()
        if self.dense.bias is not None:
            if dim == 1:
                b = self.dense.bias.clone().detach()
            else:
                b = self.dense.bias[index].clone().detach()
        self.dense.weight.requires_grad = False
        self.dense.weight.copy_(W.contiguous())
        self.dense.weight.requires_grad = True
        if self.dense.bias is not None:
            self.dense.bias.requires_grad = False
            self.dense.bias.copy_(b.contiguous())
            self.dense.bias.requires_grad = True

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        B, N, D = hidden_states.size()

        hidden_states_avg = hidden_states.mean(dim=1)
        router_output = self.router(hidden_states_avg)

        # reduce the dimension of the hidden states according to the router output or the gold label in training
        if self.mode == "train":
            # in this case we just return router_output and then loss computation (either CE or gumbel + NLL or BCE) is done in the BERTforClassification model
            if self.config.weighted_dim_reduction: 
                hidden_states = hidden_states.reshape(B, N, int(D * self.width_mult_adaptive_layers), int(1/self.width_mult_adaptive_layers))
                dim_reduction_weight = self.dim_reduction_weights[self.width_mult_to_index[self.width_mult_adaptive_layers]]
                hidden_states = torch.matmul(hidden_states, dim_reduction_weight).squeeze(-1)
            else:
                hidden_states = hidden_states[:,:,:int(self.width_mult_adaptive_layers * D)]
            post_dim_reduction_layer_norm = getattr(
                self, 'LayerNorm_{}'.format(str(self.width_mult_adaptive_layers).replace(".", "")))
            hidden_states = post_dim_reduction_layer_norm(hidden_states)
            outputs = (hidden_states, router_output)
            # else:
            #     # we don't need to run the router here. We just need to reduce the dim.
            #     hidden_states = hidden_states[:,:,:int(self.width_mult_adaptive_layers * D)]
            #     post_dim_reduction_layer_norm = getattr(
            #         self, 'LayerNorm_{}'.format(str(self.width_mult_adaptive_layers).replace(".", "")))
            #     hidden_states = post_dim_reduction_layer_norm(hidden_states)
            #     outputs = (hidden_states, None)
        elif self.mode == "eval":
            predicted_width_mult = self.config.width_mult_list[torch.argmax(router_output, dim=1).item()]
            
            if self.config.weighted_dim_reduction:
                hidden_states = hidden_states.reshape(B, N, int(D * predicted_width_mult), int(1/predicted_width_mult))
                dim_reduction_weight = self.dim_reduction_weights[self.width_mult_to_index[predicted_width_mult]]
                hidden_states = torch.matmul(hidden_states, dim_reduction_weight).squeeze(-1)
            else:
                hidden_states = hidden_states[:,:,:int(predicted_width_mult * D)]

            post_dim_reduction_layer_norm = getattr(
                self, 'LayerNorm_{}'.format(str(predicted_width_mult).replace(".", "")))
            hidden_states = post_dim_reduction_layer_norm(hidden_states)
            
            outputs = (hidden_states, predicted_width_mult)
        elif self.mode == "eval_conf_scores":  # getting conf scores in eval model with preset width mult
            if self.config.weighted_dim_reduction: 
                hidden_states = hidden_states.reshape(B, N, int(D * self.width_mult_adaptive_layers), int(1/self.width_mult_adaptive_layers))
                dim_reduction_weight = self.dim_reduction_weights[self.width_mult_to_index[self.width_mult_adaptive_layers]]
                hidden_states = torch.matmul(hidden_states, dim_reduction_weight).squeeze(-1)
            else:
                hidden_states = hidden_states[:,:,:int(self.width_mult_adaptive_layers * D)] 
            post_dim_reduction_layer_norm = getattr(
                self, 'LayerNorm_{}'.format(str(self.width_mult_adaptive_layers).replace(".", "")))
            hidden_states = post_dim_reduction_layer_norm(hidden_states)
            outputs = (hidden_states, None)
                
        return outputs


class BertLastNonAdaptiveOutputWithoutRouter(nn.Module):
    def __init__(self, config, layer_type="last_non_adaptive_without_router"):
        super(BertLastNonAdaptiveOutputWithoutRouter, self).__init__()
        self.dense = AdaptiveLinear(config.intermediate_size, config.hidden_size,
                                config.num_attention_heads, adaptive_dim=[False, False],
                                layer_type=layer_type)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        assert layer_type == "last_non_adaptive_without_router"
        self.layer_type = layer_type
         
        if config.weighted_dim_reduction:
            self.dim_reduction_weights = []
            self.width_mult_to_index = {}
            for it, width_multiplier in enumerate(config.width_mult_list):
                self.width_mult_to_index[width_multiplier] = it
                n_weights = int(1 / width_multiplier)
                self.dim_reduction_weights.append(nn.Parameter(torch.ones(n_weights, 1) * (1/n_weights), requires_grad=True))
            self.dim_reduction_weights = nn.ParameterList(self.dim_reduction_weights)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        B, N, D = hidden_states.size()

        if self.config.weighted_dim_reduction: 
            hidden_states = hidden_states.reshape(B, N, int(D * self.width_mult_adaptive_layers), int(1/self.width_mult_adaptive_layers))
            dim_reduction_weight = self.dim_reduction_weights[self.width_mult_to_index[self.width_mult_adaptive_layers]]
            hidden_states = torch.matmul(hidden_states, dim_reduction_weight).squeeze(-1)
        else:
            hidden_states = hidden_states[:,:,:int(self.width_mult_adaptive_layers * D)] 
        post_dim_reduction_layer_norm = getattr(
            self, 'LayerNorm_{}'.format(str(self.width_mult_adaptive_layers).replace(".", "")), None)
        if post_dim_reduction_layer_norm is not None: 
            hidden_states = post_dim_reduction_layer_norm(hidden_states)
        outputs = hidden_states

        return outputs


class BertOutputAdaptive(nn.Module):
    def __init__(self, config, layer_type="adaptive"):
        super(BertOutputAdaptive, self).__init__()
        # NOTE: the output of this layer is still 768 dimensional. The router will 
        # take this 768d input and output a width multiplier. 
        # In val mode, width multiplier will determine the width of the rest of the network
        # and in train mode, the width multiplier will be used to compute the loss. 
        # In train mode, the width of the rest of the network will be enforced by labels.
        self.dense = AdaptiveLinear(config.intermediate_size, config.hidden_size,
                                config.num_attention_heads, adaptive_dim=[True, True],
                                layer_type=layer_type)

        # NOTE: later we will remove this layer norm and will have width specific layer norms
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config 
        assert layer_type == "adaptive"
        self.layer_type = layer_type
        
    def forward(self, hidden_states, input_tensor):
        layer_norm = getattr(self, 'LayerNorm_{}'.format(str(self.width_mult).replace(".", "")))
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = layer_norm(hidden_states + input_tensor)

        return hidden_states
    

class BertLayer(nn.Module):
    def __init__(self, config, layer_type="normal"):
        super(BertLayer, self).__init__()
        assert layer_type != "router", "Please use the BertLayerWithRouter class for router layer."
        self.attention = BertAttention(
            config, 
            layer_type=layer_type
        )
        self.intermediate = BertIntermediate(config, layer_type=layer_type)
        if layer_type in {"normal", "dyna"}: 
            self.output = BertOutput(config, layer_type=layer_type)
        elif layer_type == "adaptive":
            self.output = BertOutputAdaptive(config, layer_type=layer_type)
       
        self.output_intermediate = config.output_intermediate
        self.mode = "train"
        self.layer_type = layer_type
        self.config = config

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        hidden_states = self.output(intermediate_output, attention_output)
        
        if self.output_intermediate:
            outputs = (hidden_states,) + attention_outputs[1:] + (intermediate_output,)
        else:
            outputs = (hidden_states,) + attention_outputs[1:]
  
        return outputs


class BertLayerWithRouter(nn.Module):
    def __init__(self, config, layer_type="router"):
        super(BertLayerWithRouter, self).__init__()
        self.attention = BertAttention(config, layer_type=layer_type)

        # self.intermediate = BertIntermediate(config, adaptive_layer=adaptive_layer, qkv_adaptive_dim=[False, False])
        self.intermediate = BertIntermediate(config, layer_type=layer_type)
        self.output = BertOutputWithRouter(
            config,
            layer_type=layer_type
        )  # or you could set it to qkv_adaptive_dim directly
 
        self.output_intermediate = config.output_intermediate
        self.mode = "train"
        assert layer_type == "router"
        self.layer_type = layer_type
        self.config = config

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        hidden_states = layer_output[0]
        
        if self.output_intermediate:
            outputs = (hidden_states,) + attention_outputs[1:] + (intermediate_output,)
        else:
            outputs = (hidden_states,) + attention_outputs[1:]

        # means that there is router_output in the layer_output
        router_output = layer_output[1]  # could be either the logits or the predicted width_mult
        outputs += (router_output,)
        
        return outputs


class BertLastNonAdaptiveLayerWithoutRouter(nn.Module):
    def __init__(self, config, layer_type="last_non_adaptive_without_router"):
        super(BertLastNonAdaptiveLayerWithoutRouter, self).__init__()
        self.attention = BertAttention(config, layer_type=layer_type)

        # self.intermediate = BertIntermediate(config, adaptive_layer=adaptive_layer, qkv_adaptive_dim=[False, False])
        self.intermediate = BertIntermediate(config, layer_type=layer_type)
        self.output = BertLastNonAdaptiveOutputWithoutRouter(
            config,
            layer_type=layer_type
        )  # or you could set it to qkv_adaptive_dim directly
 
        self.output_intermediate = config.output_intermediate
        self.mode = "train"
        assert layer_type == "last_non_adaptive_without_router"
        self.layer_type = layer_type
        self.config = config

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        hidden_states = layer_output
        
        if self.output_intermediate:
            outputs = (hidden_states,) + attention_outputs[1:] + (intermediate_output,)
        else:
            outputs = (hidden_states,) + attention_outputs[1:]
 
        return outputs

 
class BertEncoder(nn.Module):
    # def __init__(self, config, qkv_adaptive_dims, adaptive_layer):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.output_intermediate = config.output_intermediate

        layer = []
        if config.adaptive_layer_idx is not None:
            # k layers full width, 12-k layers adaptive width, router at the end of layer k
            if config.use_router:
                for layer_idx in range(config.num_hidden_layers):
                    if layer_idx < config.adaptive_layer_idx - 1:
                        layer.append(BertLayer(config, layer_type="normal"))
                    elif layer_idx == config.adaptive_layer_idx - 1:
                        layer.append(BertLayerWithRouter(config, layer_type="router"))
                    else: 
                        layer.append(BertLayer(config, layer_type="adaptive"))
            else:
                # not having a router will be useful in multiple cases:
                # 1. Individual network that has fix width for adaptive layers
                # 2. Data augmentation where you want to distill knowledge to all of the subnetworks
                if any([width_ for width_ in config.width_mult_list if width_ != 1.0]):
                    for layer_idx in range(config.num_hidden_layers):
                        if layer_idx < config.adaptive_layer_idx - 1:
                            layer.append(BertLayer(config, layer_type="normal"))
                        elif layer_idx == config.adaptive_layer_idx - 1:
                            layer.append(BertLastNonAdaptiveLayerWithoutRouter(config, layer_type="last_non_adaptive_without_router"))
                        else: 
                            layer.append(BertLayer(config, layer_type="adaptive")) 
                else:
                    # only having width 1.0 so there is no need for adaptive layer and layer with router
                    for layer_idx in range(config.num_hidden_layers):
                        layer.append(BertLayer(config, layer_type="normal"))
        else:
            # original model (not dynamic and not adaptive)
            for layer_idx in range(config.num_hidden_layers):
                layer.append(BertLayer(config, layer_type="normal"))
         
        self.layer = nn.ModuleList(layer) 

        self.depth_mult = 1.
        self.mode = "train"
        self.config = config
    
    def make_layer_norms_private(self, config, *args, **kwargs):
        """
        similar to slimmable networks (https://arxiv.org/pdf/1812.08928.pdf) we use a separate layer norm for each width multiplier
        """
        init_private_layernorms_from_scratch = kwargs.pop("init_private_layernorms_from_scratch", False)

        for i, layer in enumerate(self.layer):
            if hasattr(layer, "layer_type") and layer.layer_type == "adaptive":
                layer_norm_weight = layer.attention.output.LayerNorm.weight.clone().detach() \
                    if not init_private_layernorms_from_scratch else None
                layer_norm_bias = layer.attention.output.LayerNorm.bias.clone().detach() \
                    if not init_private_layernorms_from_scratch else None

                for width_mult in config.width_mult_list:
                    setattr(
                        layer.attention.output, 
                        f'LayerNorm_{str(width_mult).replace(".", "")}',  
                        AdaptiveLayerNorm(width_mult, config.hidden_size, eps=config.layer_norm_eps, \
                            original_weight=layer_norm_weight, original_bias=layer_norm_bias))

                delattr(layer.attention.output, 'LayerNorm')
 
                layer_norm_weight = layer.output.LayerNorm.weight.clone().detach() \
                    if not init_private_layernorms_from_scratch else None
                layer_norm_bias = layer.output.LayerNorm.bias.clone().detach() \
                    if not init_private_layernorms_from_scratch else None

                for width_mult in config.width_mult_list:
                    setattr(
                        layer.output,
                        f'LayerNorm_{str(width_mult).replace(".", "")}',
                        AdaptiveLayerNorm(width_mult, config.hidden_size, eps=config.layer_norm_eps,
                                      original_weight=layer_norm_weight, original_bias=layer_norm_bias))
                delattr(layer.output, 'LayerNorm')
            elif hasattr(layer, "layer_type") and layer.layer_type in {"router", "last_non_adaptive_without_router"}:
                layer_norm_weight = layer.output.LayerNorm.weight.clone().detach() \
                    if not init_private_layernorms_from_scratch else None
                layer_norm_bias = layer.output.LayerNorm.bias.clone().detach() \
                    if not init_private_layernorms_from_scratch else None
                for width_mult in config.width_mult_list:
                    setattr(
                        layer.output,
                        f'LayerNorm_{str(width_mult).replace(".", "")}',
                        AdaptiveLayerNorm(width_mult, config.hidden_size, eps=config.layer_norm_eps,
                                      original_weight=layer_norm_weight, original_bias=layer_norm_bias))
                # not deleting the original layer norm because it is used before the router
                # private layer norms are used after the router to make the input normalized again

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        all_intermediate = ()

        # uniformly remove layers
        depth = round(self.depth_mult * len(self.layer))
        kept_layers_index = []

        # (0,2,4,6,8,10)
        for i in range(depth):
            kept_layers_index.append(math.floor(i/self.depth_mult))

        predicted_width_mult, router_output = None, None
        for i in kept_layers_index:
            layer_module = self.layer[i]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if predicted_width_mult is not None:  # router in the eval mode has predicted the width of the adaptive layers.
                assert layer_module.layer_type == "adaptive"
                layer_module.apply(lambda m: setattr(m, 'width_mult', predicted_width_mult))

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            # if len(layer_outputs) > 1: 
            if isinstance(layer_module, BertLayerWithRouter): 
                if self.mode == "train":
                    router_output = layer_outputs[-1]  # router logits
                elif self.mode == "eval":
                    # run the router on the hidden states to get the width and then run the rest of the network with that width
                    predicted_width_mult = layer_outputs[-1]
                elif self.mode == "eval_conf_scores":
                    pass
 
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if self.output_intermediate:
                all_intermediate = all_intermediate + (layer_outputs[2],)
                
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = {"hidden_states": hidden_states}
        if self.output_hidden_states:
            # outputs = outputs + (all_hidden_states,)
            outputs.update({"all_hidden_states": all_hidden_states})
        if self.output_attentions:
            # outputs = outputs + (all_attentions,)
            outputs.update({"all_attentions": all_attentions})
        if self.output_intermediate:
            # outputs = outputs + (all_intermediate,)
            outputs.update({"all_intermediate": all_intermediate})
        
        if router_output is not None:
            # Training phase of the router
            # outputs = outputs + (router_output,)
            outputs.update({"router_logits": router_output})
        elif predicted_width_mult is not None:
            # Eval phase where the router has predicted the width for the rest of the network
            # outputs = outputs + (predicted_width_mult,))
            outputs.update({"router_predicted_width_mult": predicted_width_mult})

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        if config.adaptive_layer_idx is not None: 
            self.dense = AdaptiveLinear(config.hidden_size, config.hidden_size, config.num_attention_heads, 
                                    adaptive_dim=[True, False], layer_type="adaptive")
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = nn.Tanh()
        self.mode = "train"

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mode = "train"

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # if self.config.model_type in {"bert", "tinybert", "shallow_deep_bert", "pabee_bert", "branchy_bert"}:
        # if "roberta" not in self.config.model_type and (self.config.model_type in {"bert", "tinybert"} or any([x in self.config.model_type for x in {"shallow_deep", "pabee", "branchy"}])):
        if "roberta" not in self.config.model_type and "distilbert" not in self.config.model_type and \
            (self.config.model_type in {"bert", "tinybert"} or any([x in self.config.model_type for x in {"shallow_deep", "pabee", "branchy", "dee", "fast", "berxit"}])):
            self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        if self.config.use_router:
            if self.mode == "eval":
                predicted_width_mult = encoder_outputs["router_predicted_width_mult"]
                if self.config.model_type in {"bert", "tinybert"}:
                    self.pooler.apply(lambda m: setattr(m, 'width_mult', predicted_width_mult))
            elif self.mode == "eval_conf_scores":
                # getting conf scores in eval model
                for layer in self.encoder.layer:
                    if layer.layer_type == "adaptive":
                        predicted_width_mult = layer.width_mult
                        break
                if self.config.model_type == "bert":
                    self.pooler.apply(lambda m: setattr(m, 'width_mult', predicted_width_mult))
            elif self.mode == "train":
                # router_logits = encoder_outputs["router_logits"]
                pass

        # sequence_output = encoder_outputs[0]
        outputs = encoder_outputs
        sequence_output = encoder_outputs["hidden_states"]
        # if self.config.model_type in {"bert", "tinybert", "shallow_deep_bert", "pabee_bert", "branchy_bert"}:
        if "roberta" not in self.config.model_type and "distilbert" not in self.config.model_type and \
            (self.config.model_type in {"bert", "tinybert"} or any([x in self.config.model_type for x in {"shallow_deep", "pabee", "branchy", "dee", "fast", "berxit"}])):
            if self.config.finetuning_task == "squad":
                # for qa, we don't need to apply pooler on CLS token.
                pass 
            else:
                pooled_output = self.pooler(sequence_output) 
                outputs.update({"pooler_output": pooled_output})

        return outputs


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            Attentions scores before softmax.
        **intermediates**: (`optional`, returned when ``config.output_intermediate=True``)
            representation in the intermediate layer after nonlinearity.
    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        if config.use_router:
            # binary with sigmoid loss function
            self.router_loss_fct = BCEWithLogitsLoss()

        self.config = config 
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, width_mult_label=None, batch_width=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)  # (last_hidden_states, pooler_output, router_output) or (last_hidden_states, pooler_output)


        # pooled_output = outputs[1]
        pooled_output = outputs.pop("pooler_output")
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if self.config.use_router:
            if self.mode == "train": 
                router_logits = outputs["router_logits"]
                # in distillation we are not training the router
                if width_mult_label is not None: 
                    loss_router = self.router_loss_fct(router_logits, width_mult_label.float())
                    corrects_router = (torch.argmax(router_logits, dim=1) == batch_width).int()
            elif self.mode in {"eval", "eval_conf_scores"}:
                pass
                # router_predicted_width_mult = outputs["router_predicted_width_mult"]

        # outputs = (logits,) + outputs[2:]  # (logits, router_outputs) or (logits,)
        outputs.update({"logits": logits})

        if labels is not None:
            if self.num_labels == 1:
                #  regression task
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if self.config.use_router:
                    probs = torch.softmax(logits.detach(), dim=1)
                    probs_ground_truth_class = probs[range(probs.shape[0]), labels]

                preds = logits.argmax(dim=-1)
                corrects = (preds == labels).int()

            outputs.update({"loss": loss, "corrects": corrects})
            
            if self.config.use_router:
                if self.mode == "train" and width_mult_label is not None:
                    outputs.update({"loss_router": loss_router, "corrects_router": corrects_router})
            
            if self.config.use_router:
                outputs.update({"probs_ground_truth_class": probs_ground_truth_class})

        return outputs 


class BertForQuestionAnswering(BertPreTrainedModel):
    r"""
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        if config.adaptive_layer_idx is not None:
            self.qa_outputs = AdaptiveLinear(config.hidden_size, config.num_labels, \
                config.num_attention_heads, adaptive_dim=[True, False], layer_type="adaptive")
        else:
            self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        if config.use_router:
            # binary with sigmoid loss function
            self.router_loss_fct = BCEWithLogitsLoss()

        self.config = config 
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, 
                token_type_ids=None, start_positions=None, 
                end_positions=None, position_ids=None, 
                head_mask=None, labels=None, 
                width_mult_label=None, batch_width=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)  # (last_hidden_states, pooler_output, router_output) or (last_hidden_states, pooler_output)


        # pooled_output = outputs.pop("pooler_output")
        sequence_output = outputs.pop("hidden_states")
        if self.config.use_router:
            if self.mode == "eval":
                predicted_width_mult = outputs["router_predicted_width_mult"]
                self.qa_outputs.apply(lambda m: setattr(m, 'width_mult', predicted_width_mult))
                    
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        outputs.update({"start_logits": start_logits, "end_logits": end_logits})
        
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if self.config.use_router:
                probs_start = torch.softmax(start_logits.detach(), dim=1)
                probs_end = torch.softmax(end_logits.detach(), dim=1) 
                probs_ground_truth_start = probs_start[range(probs_start.shape[0]), start_positions]
                probs_ground_truth_end = probs_end[range(probs_end.shape[0]), end_positions]
                probs_ground_truth = (probs_ground_truth_start + probs_ground_truth_end) / 2 
                outputs.update({"probs_ground_truth": probs_ground_truth})

                if self.mode == "train": 
                    router_logits = outputs["router_logits"]
                    # in distillation we are not training the router
                    if width_mult_label is not None: 
                        loss_router = self.router_loss_fct(router_logits, width_mult_label.float())
                        # corrects_router = (torch.argmax(router_logits, dim=1) == batch_width).int()
                        outputs.update({"loss_router": loss_router})
                elif self.mode in {"eval", "eval_conf_scores"}:
                    pass 

            outputs.update({"loss": total_loss})

        return outputs 