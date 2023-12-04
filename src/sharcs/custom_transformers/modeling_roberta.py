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
"""PyTorch RoBERTa model. """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss, BCEWithLogitsLoss

from .modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from .configuration_roberta import RobertaConfig
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

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


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size,
                                                padding_idx=self.padding_idx)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      position_ids=position_ids)


# ROBERTA_START_DOCSTRING = r"""    The RoBERTa model was proposed in
#     `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
#     by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
#     Veselin Stoyanov. It is based on Google's BERT model released in 2018.
    
#     It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
#     objective and training with much larger mini-batches and learning rates.
    
#     This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained 
#     models.
#     This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
#     refer to the PyTorch documentation for all matter related to general usage and behavior.
#     .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
#         https://arxiv.org/abs/1907.11692
#     .. _`torch.nn.Module`:
#         https://pytorch.org/docs/stable/nn.html#module
#     Parameters:
#         config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the 
#             model. Initializing with a config file does not load the weights associated with the model, only the configuration.
#             Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
# """


class RobertaModel(BertModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if input_ids[:, 0].sum().item() != 0:
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your tokenize.encode()"
                           "or tokenizer.convert_tokens_to_ids().")
        return super(RobertaModel, self).forward(input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 position_ids=position_ids,
                                                 head_mask=head_mask)


class RobertaForMaskedLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMaskedLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head.decoder, self.roberta.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return x


class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        
        if config.use_router:
            # binary with sigmoid loss function
            self.router_loss_fct = BCEWithLogitsLoss()

        self.config = config
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, \
        position_ids=None, head_mask=None, width_mult_label=None, \
            batch_width=None, labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        # sequence_output = outputs[0]
        # pooled_output = outputs.pop("pooler_output")
        sequence_output = outputs.pop("hidden_states")
        if self.config.use_router:
            if self.mode == "eval":
                predicted_width_mult = outputs["router_predicted_width_mult"]
                self.classifier.dense.apply(lambda m: setattr(m, 'width_mult', predicted_width_mult))
        logits = self.classifier(sequence_output)

        if self.config.use_router:
            if self.mode == "train":
                router_logits = outputs["router_logits"]
                # in distillation we are not training the router
                if width_mult_label is not None:
                    loss_router = self.router_loss_fct(router_logits, width_mult_label.float())
                    corrects_router = (torch.argmax(router_logits, dim=1) == batch_width).int()
            elif self.mode in {"eval", "eval_conf_scores"}:
                pass

        # outputs = (logits,) + outputs[2:]
        outputs.update({"logits": logits})

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
                outputs.update({"loss": loss})
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if self.config.use_router:
                    if not self.config.use_entropy_hardness:
                        probs = torch.softmax(logits.detach(), dim=1)
                        probs_ground_truth_class = probs[range(probs.shape[0]), labels]
                    else:
                        probs = torch.softmax(logits.detach(), dim=1)
                        entropy = -torch.sum(probs * torch.log(probs), dim=1)

                preds = logits.argmax(dim=-1)
                corrects = (preds == labels).int()
                 
                outputs.update({"loss": loss, "corrects": corrects})
              
            if self.config.use_router:
                if not self.config.use_entropy_hardness:
                    outputs.update({"probs_ground_truth_class": probs_ground_truth_class})
                else:
                    outputs.update({"entropy": entropy})

                if self.mode == "train" and width_mult_label is not None:
                    outputs.update({"loss_router": loss_router, "corrects_router": corrects_router})
            
            # if self.config.use_router:
                # outputs.update({"probs_ground_truth_class": probs_ground_truth_class})

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        if config.adaptive_layer_idx is not None: 
            # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense = AdaptiveLinear(config.hidden_size, config.hidden_size, config.num_attention_heads,
                                    adaptive_dim=[True, False], layer_type="adaptive")
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForQuestionAnswering(BertPreTrainedModel):
    r"""
    """
    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)

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

        outputs = self.roberta(
            input_ids,
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