# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_utils import (
    # PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
# from transformers.models.bert.configuration_bert import BertConfig
from .configuration_bert import BertConfig
from .modeling_utils import PreTrainedModel

from dataclasses import dataclass
import torch
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple

@dataclass
class BaseModelOutputWithPastAndCrossAttentionsSkim(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsSkim(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None


@dataclass
class SequenceClassifierOutputSkim(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None
    skim_loss: Optional[torch.FloatTensor] = None
    classification_loss: Optional[torch.FloatTensor] = None
    tokens_remained: Optional[torch.FloatTensor] = None
    layer_tokens_remained: Optional[Tuple[torch.FloatTensor]] = None


def convert_softmax_mask_to_digit(skim_mask):
    # skim_mask [batch, from, to, seq_len]
    return (skim_mask == 0).to(dtype=torch.int64).unsqueeze(1).unsqueeze(1)

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = {
    "gelu": gelu, 
    "relu": torch.nn.functional.relu,
    # "gelu_new": NewGELUActivation
}


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


class AdaptiveLayerNorm(torch.nn.LayerNorm):
    def __init__(self, width_mult, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None, original_weight=None, original_bias=None):
        self.dim = int(width_mult * normalized_shape)
        super().__init__(self.dim, eps, elementwise_affine, device, dtype)
        
        if original_bias is not None and original_weight is not None:
            self.weight.data[:self.dim].copy_(original_weight[:self.dim])
            self.bias.data[:self.dim].copy_(original_bias[:self.dim])

        self.mode = "train"


def round_to_nearest(input_size, width_mult, num_heads, min_value=1):
    new_width_mult = round(num_heads * width_mult)*1.0/num_heads
    input_size = int(new_width_mult * input_size)
    new_input_size = max(min_value, input_size)
    return new_input_size


# def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
#     """Load tf checkpoints in a pytorch model."""
#     try:
#         import re

#         import numpy as np
#         import tensorflow as tf
#     except ImportError:
#         logger.error(
#             "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
#             "https://www.tensorflow.org/install/ for installation instructions."
#         )
#         raise
#     tf_path = os.path.abspath(tf_checkpoint_path)
#     logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
#     # Load weights from TF model
#     init_vars = tf.train.list_variables(tf_path)
#     names = []
#     arrays = []
#     for name, shape in init_vars:
#         logger.info(f"Loading TF weight {name} with shape {shape}")
#         array = tf.train.load_variable(tf_path, name)
#         names.append(name)
#         arrays.append(array)

#     for name, array in zip(names, arrays):
#         name = name.split("/")
#         # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
#         # which are not required for using pretrained model
#         if any(
#             n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
#             for n in name
#         ):
#             logger.info(f"Skipping {'/'.join(name)}")
#             continue
#         pointer = model
#         for m_name in name:
#             if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
#                 scope_names = re.split(r"_(\d+)", m_name)
#             else:
#                 scope_names = [m_name]
#             if scope_names[0] == "kernel" or scope_names[0] == "gamma":
#                 pointer = getattr(pointer, "weight")
#             elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
#                 pointer = getattr(pointer, "bias")
#             elif scope_names[0] == "output_weights":
#                 pointer = getattr(pointer, "weight")
#             elif scope_names[0] == "squad":
#                 pointer = getattr(pointer, "classifier")
#             else:
#                 try:
#                     pointer = getattr(pointer, scope_names[0])
#                 except AttributeError:
#                     logger.info(f"Skipping {'/'.join(name)}")
#                     continue
#             if len(scope_names) >= 2:
#                 num = int(scope_names[1])
#                 pointer = pointer[num]
#         if m_name[-11:] == "_embeddings":
#             pointer = getattr(pointer, "weight")
#         elif m_name == "kernel":
#             array = np.transpose(array)
#         try:
#             assert (
#                 pointer.shape == array.shape
#             ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
#         except AssertionError as e:
#             e.args += (pointer.shape, array.shape)
#             raise
#         logger.info(f"Initialize PyTorch weight {name}")
#         pointer.data = torch.from_numpy(array)
#     return model


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


def init_skim_predictor(module_list, mean_bias=5.0):
    for module in module_list:
        if not isinstance(module, torch.nn.Linear):
            raise ValueError("only support initialization of linear skim predictor")

        # module.bias.data[1].fill_(5.0)
        # module.bias.data[0].fill_(-5.0)
        # module.weight.data.zero_()
        module.bias.data[1].normal_(mean=mean_bias, std=0.02)
        module.bias.data[0].normal_(mean=-mean_bias, std=0.02)
        module.weight.data.normal_(mean=0.0, std=0.02)

        module._skim_initialized = True


class SkimPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None):
        super().__init__()
        
        self.hidden_size = hidden_size if hidden_size else input_size

        self.predictor = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, self.hidden_size),
            # nn.GELU(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_size),
        )

        init_skim_predictor([self.predictor[-1]])

    def forward(self, hidden_states):
        return self.predictor(hidden_states)


class SharcsSkimPredictor(nn.Module):
    def __init__(self, input_size, output_size, config, hidden_size=None, layer_type="normal"):
        super().__init__() 

        self.hidden_size = hidden_size if hidden_size else input_size
        self.LayerNorm = nn.LayerNorm(input_size)
        self.linear = AdaptiveLinear(config.hidden_size, config.hidden_size, config.num_attention_heads,
                                   adaptive_dim=[True, False], layer_type=layer_type)
        self.LayerNorm2 = nn.LayerNorm(self.hidden_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(self.hidden_size, output_size)
        self.layer_type = layer_type

        # init_skim_predictor([self.predictor[-1]])
        init_skim_predictor([self.linear2])

    def make_layer_norms_private(self, config, *args, **kwargs):
        """
        Just making the first layernorm private
        """
        layer_norm_weight = self.LayerNorm.weight.clone().detach()
        layer_norm_bias = self.LayerNorm.bias.clone().detach()

        if self.layer_type == "adaptive":
            for width_mult in config.width_mult_list:
                setattr(
                    self,
                    f'LayerNorm_{str(width_mult).replace(".", "")}',
                    AdaptiveLayerNorm(width_mult, config.hidden_size, eps=config.layer_norm_eps, \
                        original_weight=layer_norm_weight, original_bias=layer_norm_bias))

            delattr(self, 'LayerNorm')

            # layer_norm_weight = layer.output.LayerNorm.weight.clone().detach()
            # layer_norm_bias = layer.output.LayerNorm.bias.clone().detach()

            # for width_mult in config.width_mult_list:
                # setattr(
                    # layer.output,
                    # f'LayerNorm_{str(width_mult).replace(".", "")}',
                    # AdaptiveLayerNorm(width_mult, config.hidden_size, eps=config.layer_norm_eps,
                                    # original_weight=layer_norm_weight, original_bias=layer_norm_bias))
            # delattr(layer.output, 'LayerNorm')

    def forward(self, hidden_states):
        if self.layer_type in {"normal"}:
            hidden_states = self.LayerNorm(hidden_states)
            # hidden_states = self.linear(hidden_states)
            # hidden_states = self.LayerNorm2(hidden_states)
            # hidden_states = self.act(hidden_states)
            # hidden_states = self.linear2(hidden_states)
        elif self.layer_type in {"adaptive"}:
            # hidden_states = self.LayerNorm(hidden_states)
            if self.mode == "train":
                layernorm = getattr(
                self, 'LayerNorm_{}'.format(str(self.width_mult).replace(".", "")))
                hidden_states = layernorm(hidden_states)
            elif self.mode == "eval":
                layernorm =getattr(
                    self, 
                    'LayerNorm_{}'.format(str(self.width_mult).replace(".", "")))
                hidden_states = layernorm(hidden_states)

        hidden_states = self.linear(hidden_states)
        hidden_states = self.LayerNorm2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear2(hidden_states)

        return hidden_states
            
        # if self.mode == "train":
            # pass
        # else:
        # return self.predictor(hidden_states)


def test_init_skim_predictor():
    num_layers = 12

    skim_predictors = torch.nn.ModuleList([torch.nn.Linear(768,2) for _ in range(num_layers)])
    init_skim_predictor(skim_predictors)

    print(skim_predictors[0].weight, skim_predictors[0].bias)

    rand_input = torch.rand((4, 16, 768))
    print(skim_predictors[0](rand_input))



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        skim_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # mask attention probs during training for skimming
        attention_probs = attention_probs * skim_mask[:, None, None, :]

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

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class SharcsBertSelfAttention(nn.Module):
    def __init__(
        self, 
        config,
        qkv_adaptive_dim=[False, True],
        layer_type="normal"
    ):
        super().__init__()
        self.output_attentions = config.output_attentions

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.orig_num_attention_heads = config.num_attention_heads
        ###
        # self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        if getattr(config, "attention_head_size", None) is None:
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        else:
            self.attention_head_size = config.attention_head_size
        ###
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # self.value = nn.Linear(config.hidden_size, self.all_head_size)
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

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # self.max_position_embeddings = config.max_position_embeddings
            # self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # past_key_value=None,
        # output_attentions=False,
        skim_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        # is_cross_attention = encoder_hidden_states is not None

        # if is_cross_attention and past_key_value is not None:
        #     # reuse k,v, cross_attentions
        #     key_layer = past_key_value[0]
        #     value_layer = past_key_value[1]
        #     attention_mask = encoder_attention_mask
        # elif is_cross_attention:
        #     key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        #     value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        #     attention_mask = encoder_attention_mask
        # elif past_key_value is not None:
        #     key_layer = self.transpose_for_scores(self.key(hidden_states))
        #     value_layer = self.transpose_for_scores(self.value(hidden_states))
        #     key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        #     value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        # else:
        self.num_attention_heads = round(self.orig_num_attention_heads * self.query.width_mult)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # if self.is_decoder:
        #     # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        #     # Further calls to cross_attention layer can then reuse all cross-attention
        #     # key/value_states (first "if" case)
        #     # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        #     # all previous decoder key/value_states. Further calls to uni-directional self-attention
        #     # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        #     # if encoder bi-directional self-attention `past_key_value` is always `None`
        #     past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        #     seq_length = hidden_states.size()[1]
        #     position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        #     position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        #     distance = position_ids_l - position_ids_r
        #     positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        #     positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

        #     if self.position_embedding_type == "relative_key":
        #         relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #         attention_scores = attention_scores + relative_position_scores
        #     elif self.position_embedding_type == "relative_key_query":
        #         relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #         relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
        #         attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # mask attention probs during training for skimming
        attention_probs = attention_probs * skim_mask[:, None, None, :]

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

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)

        # if self.is_decoder:
        #     outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SharcsBertSelfOutput(nn.Module):
    def __init__(
        self, 
        config,
        layer_type="normal"):
        super().__init__()
        if getattr(config, "attention_head_size", None) is not None:
            self_attn_output_dim = config.attention_head_size * config.num_attention_heads 
        else:
            self_attn_output_dim = config.hidden_size

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if layer_type == "adaptive":
            self.dense = AdaptiveLinear(self_attn_output_dim, config.hidden_size,
                                    config.num_attention_heads, adaptive_dim=[True, True], layer_type=layer_type)
        elif layer_type in {"normal", "router"}:
            self.dense = AdaptiveLinear(self_attn_output_dim, config.hidden_size,
                                    config.num_attention_heads, adaptive_dim=[False, False], layer_type=layer_type)

        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        skim_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            skim_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SharcsBertAttention(nn.Module):
    def __init__(
        self, 
        config,
        layer_type="normal"
    ):
        super().__init__()
        if layer_type == "adaptive":
            self.self = SharcsBertSelfAttention(config, qkv_adaptive_dim=[True, True], layer_type=layer_type)
        elif layer_type in {"normal", "router"}:
            self.self = SharcsBertSelfAttention(config, qkv_adaptive_dim=[False, False], layer_type=layer_type)
        self.output = SharcsBertSelfOutput(config, layer_type=layer_type)
        self.mode = "train"
        self.layer_type = layer_type
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # past_key_value=None,
        # output_attentions=False,
        skim_mask=None,
    ):
        self_outputs = self.self(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            # encoder_hidden_states,
            # encoder_attention_mask,
            # past_key_value,
            # output_attentions,
            skim_mask=skim_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SharcsBertIntermediate(nn.Module):
    def __init__(
        self, 
        config,
        layer_type="normal"
    ):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if layer_type in {"normal", "router", "last_non_adaptive_without_router"}:
            self.dense = AdaptiveLinear(config.hidden_size, config.intermediate_size,
                                    config.num_attention_heads, adaptive_dim=[False, False], layer_type=layer_type)        
        elif layer_type == "adaptive":
            self.dense = AdaptiveLinear(config.hidden_size, config.intermediate_size,
                                    config.num_attention_heads, adaptive_dim=[True, True], layer_type=layer_type)
        elif layer_type == "dyna":
            self.dense = AdaptiveLinear(config.hidden_size, config.intermediate_size,
                                    config.num_attention_heads, adaptive_dim=[False, True], layer_type=layer_type)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SharcsBertOutput(nn.Module):
    def __init__(self, config, layer_type="normal"):
        super().__init__()
        assert layer_type == "normal"

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SharcsBertOutputWithRouter(nn.Module):
    def __init__(self, config, layer_type="router"):
        super(SharcsBertOutputWithRouter, self).__init__()
        # NOTE: the output of this layer is still 768 dimensional. The router will
        # take this 768d input and output a width multiplier.
        # In val mode, width multiplier will determine the width of the rest of the network
        # and in train mode, the width multiplier will be used to compute the loss.
        # In train mode, the width of the rest of the network will be enforced by labels.
        self.dense = AdaptiveLinear(config.intermediate_size, config.hidden_size,
                                config.num_attention_heads, adaptive_dim=[False, False], layer_type=layer_type)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size_router),
            nn.ReLU(),
            nn.Linear(config.hidden_size_router, len(config.width_mult_list))
            )
        assert layer_type == "router"
        self.layer_type = "router"
        
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
            hidden_states = hidden_states[:,:,:int(self.width_mult_adaptive_layers * D)]
            post_dim_reduction_layer_norm = getattr(
                self, 'LayerNorm_{}'.format(str(self.width_mult_adaptive_layers).replace(".", "")))
            hidden_states = post_dim_reduction_layer_norm(hidden_states)
            outputs = (hidden_states, router_output)
        elif self.mode == "eval":
            predicted_width_mult = self.config.width_mult_list[torch.argmax(router_output, dim=1).item()]
            hidden_states = hidden_states[:,:,:int(predicted_width_mult * D)]
            post_dim_reduction_layer_norm = getattr(
                self, 'LayerNorm_{}'.format(str(predicted_width_mult).replace(".", "")))
            hidden_states = post_dim_reduction_layer_norm(hidden_states)            
            outputs = (hidden_states, predicted_width_mult)
                
        return outputs


class SharcsBertOutputAdaptive(nn.Module):
    def __init__(self, config, layer_type="adaptive"):
        super(SharcsBertOutputAdaptive, self).__init__()
        assert layer_type == "adaptive"
        # NOTE: the output of this layer is still 768 dimensional. The router will 
        # take this 768d input and output a width multiplier. 
        # In val mode, width multiplier will determine the width of the rest of the network
        # and in train mode, the width multiplier will be used to compute the loss. 
        # In train mode, the width of the rest of the network will be enforced by labels.
        self.dense = AdaptiveLinear(config.intermediate_size, config.hidden_size,
                                config.num_attention_heads, adaptive_dim=[True, True],
                                layer_type=layer_type)

        # NOTE: later we will remove this layer norm and will have width specific layer norms
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        skim_mask=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            skim_mask=skim_mask,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SharcsBertLayer(nn.Module):
    def __init__(
        self, 
        config,
        layer_type="normal"
    ):
        super().__init__()
        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SharcsBertAttention(config, layer_type=layer_type)
        # self.is_decoder = config.is_decoder
        # self.add_cross_attention = config.add_cross_attention
        # if self.add_cross_attention:
        #     assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            # self.crossattention = BertAttention(config)
        if layer_type in {"normal"}:
            self.output = SharcsBertOutput(config, layer_type=layer_type)
        elif layer_type == "adaptive":
            self.output = SharcsBertOutputAdaptive(config, layer_type=layer_type)

        self.intermediate = SharcsBertIntermediate(config, layer_type=layer_type)
        self.output_intermediate = config.output_intermediate
        self.mode = "train"
        self.layer_type=layer_type
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # past_key_value=None,
        # output_attentions=False,
        skim_mask=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            # output_attentions=output_attentions,
            # past_key_value=self_attn_past_key_value,
            skim_mask=skim_mask,
        )
        attention_output = attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        # if self.is_decoder:
            # outputs = self_attention_outputs[1:-1]
            # present_key_value = self_attention_outputs[-1]
        # else:
        # outputs = attention_outputs[1:]  # add self attentions if we output attention weights

        # cross_attn_present_key_value = None
        # if self.is_decoder and encoder_hidden_states is not None:
            # assert hasattr(
                # self, "crossattention"
            # ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            # cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # cross_attention_outputs = self.crossattention(
            #     attention_output,
            #     attention_mask,
            #     head_mask,
            #     encoder_hidden_states,
            #     encoder_attention_mask,
            #     cross_attn_past_key_value,
            #     output_attentions,
            # )
            # attention_output = cross_attention_outputs[0]
            # outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            # cross_attn_present_key_value = cross_attention_outputs[-1]
            # present_key_value = present_key_value + cross_attn_present_key_value

        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        # )
        intermediate_output = self.intermediate(attention_output)
        hidden_states = self.output(intermediate_output, attention_output)

        # outputs = (layer_output,) + outputs
        if self.output_intermediate:
            outputs = (hidden_states,) + attention_outputs[1:] + (intermediate_output,)
        else:
            outputs = (hidden_states,) + attention_outputs[1:]

        # if decoder, return the attn key/values as the last output
        # if self.is_decoder:
            # outputs = outputs + (present_key_value,)

        return outputs

    # def feed_forward_chunk(self, attention_output):
        # return layer_output


class SharcsBertLayerWithRouter(nn.Module):
    def __init__(self, config, layer_type="router"):
        super(SharcsBertLayerWithRouter, self).__init__()
        self.attention = SharcsBertAttention(config, layer_type=layer_type)

        # self.intermediate = BertIntermediate(config, adaptive_layer=adaptive_layer, qkv_adaptive_dim=[False, False])
        self.intermediate = SharcsBertIntermediate(config, layer_type=layer_type)
        self.output = SharcsBertOutputWithRouter(
            config,
            layer_type=layer_type
        )  # or you could set it to qkv_adaptive_dim directly
 
        self.output_intermediate = config.output_intermediate
        self.mode = "train"
        assert layer_type == "router"
        self.layer_type = layer_type
        self.config = config

    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        head_mask=None,
        skim_mask=None
    ):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, skim_mask=skim_mask)
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


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        # skim predictors for each layer
        self.skim_predictors = nn.ModuleList([SkimPredictor(config.hidden_size, 2) for _ in range(config.num_hidden_layers)])
        # init_skim_predictor(self.skim_predictors)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_skim_mask = ()

        next_decoder_cache = () if use_cache else None

        forward_hidden_states = hidden_states.clone()

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            skim_mask = nn.functional.gumbel_softmax(self.skim_predictors[i](hidden_states[:,1:,:]), hard=True, tau=1)
            skim_mask = skim_mask[:,:,1]
            skim_mask_with_cls = torch.ones(skim_mask.shape[0], skim_mask.shape[1]+1, device=skim_mask.device)
            skim_mask_with_cls[:,1:] = skim_mask
            skim_mask = skim_mask_with_cls
            # multiple current layer skim mask with last layer skim mask
            # to gurantee skimmed tokens are never recovered
            if all_skim_mask:
                skim_mask = skim_mask * all_skim_mask[-1]
            all_skim_mask += (skim_mask, )

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    skim_mask,
                )

            hidden_states = layer_outputs[0]
            forward_hidden_states = forward_hidden_states * (1-skim_mask.view(*skim_mask.shape,1)) + hidden_states * skim_mask.view(*skim_mask.shape,1)
            # binary_skim_mask = skim_mask.to(dtype=torch.bool)
            # forward_hidden_states[binary_skim_mask] = hidden_states[binary_skim_mask]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)


        if not return_dict:
            return tuple(
                v
                for v in [
                    forward_hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentionsSkim(
            last_hidden_state=forward_hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask=attention_mask,
            skim_mask=all_skim_mask,
        )


class SharcsBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.output_intermediate = config.output_intermediate

        assert config.adaptive_layer_idx is not None and config.use_router
        layer, skim_predictors = [], []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx < config.adaptive_layer_idx - 1:
                layer.append(SharcsBertLayer(config, layer_type="normal"))
                skim_predictors.append(SharcsSkimPredictor(config.hidden_size, 2, config, layer_type="normal"))
            elif layer_idx == config.adaptive_layer_idx - 1:
                layer.append(SharcsBertLayerWithRouter(config, layer_type="router"))
                skim_predictors.append(SharcsSkimPredictor(config.hidden_size, 2, config, layer_type="normal"))
            else:
                layer.append(SharcsBertLayer(config, layer_type="adaptive"))
                skim_predictors.append(SharcsSkimPredictor(config.hidden_size, 2, config, layer_type="adaptive"))

        self.layer = nn.ModuleList(layer)
        self.skim_predictors = nn.ModuleList(skim_predictors)

        # skim predictors for each layer
        # self.skim_predictors = nn.ModuleList([SkimPredictor(config.hidden_size, 2) for _ in range(config.num_hidden_layers)])
        # self.skim_predictors = nn.ModuleList([SkimPredictor(config.hidden_size, 2, config) for _ in range(config.num_hidden_layers)])
        self.config = config
        # init_skim_predictor(self.skim_predictors)


    def make_layer_norms_private(self, config, *args, **kwargs):
        for skim_predictor in self.skim_predictors:
            skim_predictor.make_layer_norms_private(config, *args, **kwargs)

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
            elif hasattr(layer, "layer_type") and layer.layer_type in {"router"}:
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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # past_key_values=None,
        # use_cache=None,
        # output_attentions=False,
        # output_hidden_states=False,
        # return_dict=True,
    ):
        ###
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attentions = () if output_attentions else None
        # all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = ()
        all_attentions = ()
        # all_cross_attentions = ()
        all_intermediate = ()
        all_skim_mask = ()

        # next_decoder_cache = () if use_cache else None

        forward_hidden_states = hidden_states.clone()
        predicted_width_mult, router_output = None, None
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if predicted_width_mult is not None:  # router in the eval mode has predicted the width of the adaptive layers.
                assert layer_module.layer_type == "adaptive"
                layer_module.apply(lambda m: setattr(m, 'width_mult', predicted_width_mult))
                self.skim_predictors[i].apply(lambda m: setattr(m, 'width_mult', predicted_width_mult))
 
            skim_mask = nn.functional.gumbel_softmax(self.skim_predictors[i](hidden_states[:,1:,:]), hard=True, tau=1)
            skim_mask = skim_mask[:,:,1]
            skim_mask_with_cls = torch.ones(skim_mask.shape[0], skim_mask.shape[1]+1, device=skim_mask.device)
            skim_mask_with_cls[:,1:] = skim_mask
            skim_mask = skim_mask_with_cls
            # multiple current layer skim mask with last layer skim mask
            # to gurantee skimmed tokens are never recovered
            if all_skim_mask:
                skim_mask = skim_mask * all_skim_mask[-1]
            all_skim_mask += (skim_mask, )

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                # encoder_hidden_states,
                # encoder_attention_mask,
                # past_key_value,
                # output_attentions=,
                skim_mask=skim_mask,
            )

            hidden_states = layer_outputs[0]

            if isinstance(layer_module, SharcsBertLayerWithRouter):
                if self.mode == "train":
                    router_output = layer_outputs[-1]  # router logits
                elif self.mode == "eval":
                    # run the router on the hidden states to get the width and then run the rest of the network with that width
                    predicted_width_mult = layer_outputs[-1]
                forward_hidden_states = forward_hidden_states[:,:,:hidden_states.size(-1)] 
                 
            forward_hidden_states = forward_hidden_states * (1-skim_mask.view(*skim_mask.shape,1)) + hidden_states * skim_mask.view(*skim_mask.shape,1)
            # binary_skim_mask = skim_mask.to(dtype=torch.bool)
            # forward_hidden_states[binary_skim_mask] = hidden_states[binary_skim_mask]

            # if use_cache:
                # next_decoder_cache += (layer_outputs[-1],)
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if self.output_intermediate:
                all_intermediate = all_intermediate + (layer_outputs[2],)
                # all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # if self.config.add_cross_attention:
                    # all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        outputs = {"hidden_states": hidden_states}
        if self.output_hidden_states:
            # all_hidden_states = all_hidden_states + (hidden_states,)
            outputs.update({"all_hidden_states": all_hidden_states})
        if self.output_attentions:
            # outputs = outputs + (all_attentions,)
            outputs.update({"all_attentions": all_attentions})
        if self.output_intermediate:
            # outputs = outputs + (all_intermediate,)
            outputs.update({"all_intermediate": all_intermediate})

        if router_output is not None:
            outputs.update({"router_logits": router_output})
        elif predicted_width_mult is not None:
            outputs.update({"router_predicted_width_mult": predicted_width_mult})

        outputs.update({"skim_mask": all_skim_mask})
        return outputs


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SharcsBertPooler(nn.Module):
    def __init__(self, config):
        super(SharcsBertPooler, self).__init__()
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
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    # load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if hasattr(module, '_skim_initialized') and module._skim_initialized:
            return
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPoolingAndCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentionsSkim(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            attention_mask=encoder_outputs.attention_mask,
            skim_mask=encoder_outputs.skim_mask,
        )


class SharcsBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        # self.encoder = BertEncoder(config)
        self.encoder = SharcsBertEncoder(config)

        self.pooler = SharcsBertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if self.config.is_decoder:
        #     use_cache = use_cache if use_cache is not None else self.config.use_cache
        # else:
        #     use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.is_decoder and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #     encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # else:
        #     encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_extended_attention_mask,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        ### added by Reza
        if self.config.use_router:
            if self.mode == "eval":
                predicted_width_mult = encoder_outputs["router_predicted_width_mult"]
                if self.config.model_type in {"sharcstranskimer"}:
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
        if self.config.model_type == "sharcstranskimer":
            pooled_output = self.pooler(sequence_output)
            outputs.update({"pooler_output": pooled_output})

        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        outputs.update({
            "skim_mask": encoder_outputs['skim_mask'],
            # "attention_mask": encoder_outputs.attention_mask, 
            "hidden_states": encoder_outputs['hidden_states'],
        })
        # if not return_dict:
            # return (sequence_output, pooled_output) + encoder_outputs[1:]

        # return BaseModelOutputWithPoolingAndCrossAttentionsSkim(
        # return dict(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        #     attention_mask=encoder_outputs.attention_mask,
        #     skim_mask=encoder_outputs.skim_mask,
        # )
        return outputs
        ###


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.skim_coefficient = config.skim_coefficient if hasattr(config, 'skim_coefficient') else 1

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        skim_loss, neat_mac = 0.0, 0.0
        layer_neat_mac = list()
        all_tokens_length = torch.mean(torch.sum(attention_mask.to(torch.float32),dim=-1))
        for mask in outputs.skim_mask:
            accumulated_skim_mask = torch.mean(torch.sum(mask,dim=1))
            skim_loss += accumulated_skim_mask/mask.shape[1]
            layer_neat_mac.append(accumulated_skim_mask/all_tokens_length)
            neat_mac += accumulated_skim_mask/all_tokens_length
        skim_loss /= self.config.num_hidden_layers
        neat_mac /= self.config.num_hidden_layers
        classification_loss = loss
        # print(skim_loss, neat_mac, loss)
        # loss = skim_loss
        if labels is not None:
            loss = self.skim_coefficient * skim_loss + loss

        return SequenceClassifierOutputSkim(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=outputs.attention_mask,
            skim_mask=outputs.skim_mask,
            skim_loss=skim_loss,
            classification_loss=classification_loss,
            tokens_remained=neat_mac,
            layer_tokens_remained=layer_neat_mac,
        )


class SharcsBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # self.bert = BertModel(config)
        self.bert = SharcsBertModel(config)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        ### added by Reza
        if config.use_router:
            self.router_loss_fct = BCEWithLogitsLoss()
        self.config = config 
        ### 

        self.skim_coefficient = config.skim_coefficient if hasattr(config, 'skim_coefficient') else 1

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        ###
        width_mult_label=None,
        batch_width=None,
        ###
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        ### added by Reza 
        # pooled_output = outputs[1]
        pooled_output = outputs.pop("pooler_output")
        ###
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        ### added by Reza 
        if self.config.use_router:        
            if self.mode == "train": 
                router_logits = outputs["router_logits"]
                # in distillation we are not training the router
                if width_mult_label is not None: 
                    loss_router = self.router_loss_fct(router_logits, width_mult_label.float())
                    corrects_router = (torch.argmax(router_logits, dim=1) == batch_width).int()
            elif self.mode in {"eval", "eval_conf_scores"}:
                pass
        ### 
        outputs.update({"logits": logits})
        loss = None
        if labels is not None:
            ### added by Reza
            self.config.problem_type = "single_label_classification"
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


def test_BertEncoder():
    import transformers

    logging.debug(f'Start unit test for BertEncoder')

    config = transformers.BertConfig.from_pretrained('bert-base-uncased')
    # config.output_attentions = False
    encoder = BertEncoder(config)

    rand_hidden_states = torch.rand((1,8,768))
    # rand_hidden_states = torch.rand((4,128,768))

    encoder_outputs = encoder(rand_hidden_states)

    logging.debug(f'output attention: {config.output_attentions}, {encoder_outputs[-1][0].shape}')

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    test_BertEncoder()