# in this baseline, you attach a classifier to one of the internal layers in the model.



# To determine the layer you want to attach the classifier, you can compare the number of flops 
# of your method for an input sequqnece of length 128 with the number of flops of this model. 

from .modeling_roberta import RobertaForSequenceClassification
from .modeling_bert import BertForSequenceClassification
from .modeling_distilbert import DistilBertForSequenceClassification
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch
import numpy as np


class BerxitRoberta(RobertaForSequenceClassification):
    def __init__(self, config):
        config.output_hidden_states = True
        super().__init__(config)
        assert getattr(config, "internal_classifier_layer", None) is not None
        assert getattr(config, "internal_classifier_all_layers", False) is True
        self.lte = nn.Linear(config.hidden_size, 1)
        self.lte_loss = nn.MSELoss()
        self.config.internal_classifier_layer = config.internal_classifier_layer
        
        self.internal_classifiers = []

        # freeze all the weights in the model 
        # for param in self.parameters():
        #     param.requires_grad = False

        for _ in range(self.config.internal_classifier_layer + 1):
            self.internal_classifiers.append(nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_labels)
            ))
        self.internal_classifiers = nn.ModuleList(self.internal_classifiers)
        self.internal_classifier_thresh = config.internal_classifier_thresh
        self.config = config 
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, \
        labels=None, position_ids=None, head_mask=None):
        outputs = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels
        )
        all_hidden_states = outputs["all_hidden_states"]

        logits_internal, probs, lte_preds = [], [], []
        for i in range(1, self.config.internal_classifier_layer + 2):
            internal_hidden_state = all_hidden_states[i]
            lte_preds.append(torch.sigmoid(self.lte(internal_hidden_state[:, 0])))
            logits_internal.append(self.internal_classifiers[i - 1](internal_hidden_state.mean(dim=1)))
            probs.append(torch.softmax(logits_internal[-1].detach(), dim=1))

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits_internal.view(-1), labels.view(-1))
                outputs.update({"loss": loss})
            else:
                loss_fct = CrossEntropyLoss()
                loss_internal = 0
                loss_lte = 0
                for i in range(len(logits_internal)):
                    preds = logits_internal[i].argmax(dim=-1)
                    labels_lte = (preds == labels).int()
                    loss_lte += self.lte_loss(lte_preds[i].squeeze(1), labels_lte.float())
                    loss_internal += loss_fct(logits_internal[i].view(-1, self.num_labels), labels.view(-1))
                
                loss_internal += loss_lte
 
                if not self.training:
                    # if the maximum probability is greater than the threshold, then we predict the class with the maximum probability
                    outputs["internal_prediction"] = False
                    for i in range(len(logits_internal)):
                        # entropy = - (probs[i] * torch.log(probs[i])).sum(dim=1)
                        if lte_preds[i].item() > self.internal_classifier_thresh:
                            # overwriting the predictions of full depth model
                            outputs["logits"] = logits_internal[i]
                            outputs["internal_prediction"] = True
                            outputs["depth"] = i + 1
                            preds = logits_internal[i].argmax(dim=-1)
                            corrects = (preds == labels).int()
                            outputs.update({"corrects": corrects})
                            break
                elif self.training:
                    outputs["loss_internal"] = loss_internal

                     
        outputs.update({"logits_internal": logits_internal})

        return outputs

         
class BerxitBert(BertForSequenceClassification):
    def __init__(self, config):
        config.output_hidden_states = True
        super().__init__(config)
        assert getattr(config, "internal_classifier_layer", None) is not None
        assert getattr(config, "internal_classifier_all_layers", False) is True
        self.lte = nn.Linear(config.hidden_size, 1)
        self.lte_loss = nn.MSELoss()
        self.config.internal_classifier_layer = config.internal_classifier_layer
        
        self.internal_classifiers = []

        # freeze all the weights in the model 
        # for param in self.parameters():
        #     param.requires_grad = False

        for _ in range(self.config.internal_classifier_layer + 1):
            self.internal_classifiers.append(nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_labels)
            ))
        self.internal_classifiers = nn.ModuleList(self.internal_classifiers)
        self.internal_classifier_thresh = config.internal_classifier_thresh
        self.config = config 
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, \
        labels=None, position_ids=None, head_mask=None):
        outputs = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels
        )
        all_hidden_states = outputs["all_hidden_states"]

        logits_internal, probs, lte_preds = [], [], []
        for i in range(1, self.config.internal_classifier_layer + 2):
            internal_hidden_state = all_hidden_states[i]
            lte_preds.append(torch.sigmoid(self.lte(internal_hidden_state[:, 0])))
            logits_internal.append(self.internal_classifiers[i - 1](internal_hidden_state.mean(dim=1)))
            probs.append(torch.softmax(logits_internal[-1].detach(), dim=1))

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits_internal.view(-1), labels.view(-1))
                outputs.update({"loss": loss})
            else:
                loss_fct = CrossEntropyLoss()
                loss_internal = 0
                loss_lte = 0
                for i in range(len(logits_internal)):
                    preds = logits_internal[i].argmax(dim=-1)
                    labels_lte = (preds == labels).int()
                    loss_lte += self.lte_loss(lte_preds[i].squeeze(1), labels_lte.float())
                    loss_internal += loss_fct(logits_internal[i].view(-1, self.num_labels), labels.view(-1))
                
                loss_internal += loss_lte
 
                if not self.training:
                    # if the maximum probability is greater than the threshold, then we predict the class with the maximum probability
                    outputs["internal_prediction"] = False
                    for i in range(len(logits_internal)):
                        # entropy = - (probs[i] * torch.log(probs[i])).sum(dim=1)
                        if lte_preds[i].item() > self.internal_classifier_thresh:
                            # overwriting the predictions of full depth model
                            outputs["logits"] = logits_internal[i]
                            outputs["internal_prediction"] = True
                            outputs["depth"] = i + 1
                            preds = logits_internal[i].argmax(dim=-1)
                            corrects = (preds == labels).int()
                            outputs.update({"corrects": corrects})
                            break
                elif self.training:
                    outputs["loss_internal"] = loss_internal

                     
        outputs.update({"logits_internal": logits_internal})

        return outputs


class BerxitDistilBert(DistilBertForSequenceClassification):
    def __init__(self, config):
        config.output_hidden_states = True
        super().__init__(config)
        assert getattr(config, "internal_classifier_layer", None) is not None
        assert getattr(config, "internal_classifier_all_layers", False) is True
        self.lte = nn.Linear(config.hidden_size, 1)
        self.lte_loss = nn.MSELoss()
        self.config.internal_classifier_layer = config.internal_classifier_layer
        
        self.internal_classifiers = []

        # freeze all the weights in the model 
        # for param in self.parameters():
        #     param.requires_grad = False

        for _ in range(self.config.internal_classifier_layer + 1):
            self.internal_classifiers.append(nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_labels)
            ))
        self.internal_classifiers = nn.ModuleList(self.internal_classifiers)
        self.internal_classifier_thresh = config.internal_classifier_thresh
        self.config = config 
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, \
        labels=None, position_ids=None, head_mask=None):
        outputs = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels
        )
        all_hidden_states = outputs["all_hidden_states"]

        logits_internal, probs, lte_preds = [], [], []
        for i in range(1, self.config.internal_classifier_layer + 2):
            internal_hidden_state = all_hidden_states[i]
            lte_preds.append(torch.sigmoid(self.lte(internal_hidden_state[:, 0])))
            logits_internal.append(self.internal_classifiers[i - 1](internal_hidden_state.mean(dim=1)))
            probs.append(torch.softmax(logits_internal[-1].detach(), dim=1))

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits_internal.view(-1), labels.view(-1))
                outputs.update({"loss": loss})
            else:
                loss_fct = CrossEntropyLoss()
                loss_internal = 0
                loss_lte = 0
                for i in range(len(logits_internal)):
                    preds = logits_internal[i].argmax(dim=-1)
                    labels_lte = (preds == labels).int()
                    loss_lte += self.lte_loss(lte_preds[i].squeeze(1), labels_lte.float())
                    loss_internal += loss_fct(logits_internal[i].view(-1, self.num_labels), labels.view(-1))
                
                loss_internal += loss_lte
 
                if not self.training:
                    # if the maximum probability is greater than the threshold, then we predict the class with the maximum probability
                    outputs["internal_prediction"] = False
                    for i in range(len(logits_internal)):
                        # entropy = - (probs[i] * torch.log(probs[i])).sum(dim=1)
                        if lte_preds[i].item() > self.internal_classifier_thresh:
                            # overwriting the predictions of full depth model
                            outputs["logits"] = logits_internal[i]
                            outputs["internal_prediction"] = True
                            outputs["depth"] = i + 1
                            preds = logits_internal[i].argmax(dim=-1)
                            corrects = (preds == labels).int()
                            outputs.update({"corrects": corrects})
                            break
                elif self.training:
                    outputs["loss_internal"] = loss_internal

                     
        outputs.update({"logits_internal": logits_internal})

        return outputs