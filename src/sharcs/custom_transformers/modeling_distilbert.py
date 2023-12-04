import torch.nn as nn
import torch
from .modeling_bert import AdaptiveLinear, BertPreTrainedModel, BertEncoder, BertModel
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss, BCEWithLogitsLoss


class DistilBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, input_embeds=None):
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
            input_embeds (*optional*, torch.Tensor):
                The pre-computed word embeddings. Can only be passed if the input ids are `None`.


        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        if input_ids is not None:
            input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        seq_length = input_embeds.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class DistilBertForSequenceClassification(BertPreTrainedModel):
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
        super(DistilBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.embeddings = DistilBertEmbeddings(config)
        self.distilbert = BertEncoder(config)
            
        if config.adaptive_layer_idx is not None: 
            self.pre_classifier = AdaptiveLinear(config.hidden_size, config.hidden_size, 
                                             config.num_attention_heads, adaptive_dim=[True, False], 
                                             layer_type="adaptive")
        else:
            self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels) 

        if config.use_router:
            # binary with sigmoid loss function
            self.router_loss_fct = BCEWithLogitsLoss()

        self.config = config
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, 
                width_mult_label=None, batch_width=None):
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

        # embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        embedding_output = self.embeddings(input_ids)
        
        outputs = self.distilbert(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
        )  # (last_hidden_states, pooler_output, router_output) or (last_hidden_states, pooler_output)

        sequence_output = outputs.pop("hidden_states")
        if self.config.use_router:
            if self.mode == "eval":
                predicted_width_mult = outputs["router_predicted_width_mult"]
                self.pre_classifier.apply(lambda m: setattr(m, 'width_mult', predicted_width_mult)) 
        pooled_output = sequence_output[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.config.use_router:
            if self.mode == "train":
                router_logits = outputs["router_logits"]
                if width_mult_label is not None:
                    loss_router = self.router_loss_fct(router_logits, width_mult_label.float())
                    corrects_router = (torch.argmax(router_logits, dim=1) == batch_width).int()
            elif self.mode in {"eval", "eval_conf_scores"}:
                pass

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