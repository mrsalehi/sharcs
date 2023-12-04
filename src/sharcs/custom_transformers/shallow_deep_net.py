# in this baseline, you attach a classifier to one of the internal layers in the model.



# To determine the layer you want to attach the classifier, you can compare the number of flops 
# of your method for an input sequqnece of length 128 with the number of flops of this model. 

from .modeling_roberta import RobertaForSequenceClassification
from .modeling_bert import BertForSequenceClassification
from .modeling_distilbert import DistilBertForSequenceClassification
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch


class ShallowDeepRoberta(RobertaForSequenceClassification):
    def __init__(self, config):
        config.output_hidden_states = True
        super().__init__(config)
        assert getattr(config, "internal_classifier_layer", None) is not None
        self.config.internal_classifier_layer = config.internal_classifier_layer

        if config.internal_classifier_all_layers:
            self.internal_classifiers = []
            for i in range(self.config.internal_classifier_layer + 1):
                self.internal_classifiers.append(nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, config.num_labels)
                ))
            self.internal_classifiers = nn.ModuleList(self.internal_classifiers)
        else:
            self.internal_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_labels)
            )
        self.internal_classifier_thresh = config.internal_classifier_thresh
        
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

        if self.config.internal_classifier_all_layers:
            logits_internal, probs = [], []
            for i in range(1, self.config.internal_classifier_layer + 2):
                internal_hidden_state = all_hidden_states[i]
                logits_internal.append(self.internal_classifiers[i - 1](internal_hidden_state.mean(dim=1))) 
                probs.append(torch.softmax(logits_internal[-1].detach(), dim=1))
        else: 
            internal_hidden_states = all_hidden_states[self.config.internal_classifier_layer + 1]
            logits_internal = self.internal_classifier(internal_hidden_states.mean(dim=1))
            probs = torch.softmax(logits_internal.detach(), dim=1)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits_internal.view(-1), labels.view(-1))
                outputs.update({"loss": loss})
            else:
                loss_fct = CrossEntropyLoss()
                if self.config.internal_classifier_all_layers:
                    loss_internal = 0
                    sum_layer_inds = 0
                    for i in range(len(logits_internal)):
                        loss_internal += (i+1) * loss_fct(logits_internal[i].view(-1, self.num_labels), labels.view(-1))
                        sum_layer_inds += (i+1)
                    loss_internal /= sum_layer_inds
                else:
                    loss_internal = loss_fct(logits_internal.view(-1, self.num_labels), labels.view(-1))
 
                if not self.training:
                    # if the maximum probability is greater than the threshold, then we predict the class with the maximum probability
                    if self.config.internal_classifier_all_layers:
                        outputs["internal_prediction"] = False
                        for i in range(len(logits_internal)):
                            if probs[i].max(dim=1)[0].item() > self.internal_classifier_thresh:
                                # overwriting the predictions of full depth model
                                outputs["logits"] = logits_internal[i]
                                outputs["internal_prediction"] = True
                                outputs["depth"] = i + 1
                                preds = logits_internal[i].argmax(dim=-1)
                                corrects = (preds == labels).int()
                                outputs.update({"corrects": corrects})
                                break
                    else:
                        outputs["internal_prediction"] = False
                        if probs.max(dim=1)[0].item() > self.internal_classifier_thresh:
                            # overwriting the predictions of full depth model
                            outputs["logits"] = logits_internal
                            outputs["internal_prediction"] = True
                            preds = logits_internal.argmax(dim=-1)
                            corrects = (preds == labels).int()
                            outputs.update({"corrects": corrects})
                elif self.training:
                    outputs["loss_internal"] = loss_internal
                     
        outputs.update({"logits_internal": logits_internal})

        return outputs

         
class ShallowDeepBert(BertForSequenceClassification):
    def __init__(self, config):
        config.output_hidden_states = True
        super().__init__(config)
        assert getattr(config, "internal_classifier_layer", None) is not None
        self.config.internal_classifier_layer = config.internal_classifier_layer

        if config.internal_classifier_all_layers:
            self.internal_classifiers = []
            for i in range(self.config.internal_classifier_layer + 1):
                self.internal_classifiers.append(nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, config.num_labels)
                ))
            self.internal_classifiers = nn.ModuleList(self.internal_classifiers)
        else:
            self.internal_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_labels)
            )
        self.internal_classifier_thresh = config.internal_classifier_thresh
        
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

        if self.config.internal_classifier_all_layers:
            logits_internal, probs = [], []
            for i in range(1, self.config.internal_classifier_layer + 2):
                internal_hidden_state = all_hidden_states[i]
                logits_internal.append(self.internal_classifiers[i - 1](internal_hidden_state.mean(dim=1))) 
                probs.append(torch.softmax(logits_internal[-1].detach(), dim=1))
        else: 
            internal_hidden_states = all_hidden_states[self.config.internal_classifier_layer + 1]
            logits_internal = self.internal_classifier(internal_hidden_states.mean(dim=1))
            probs = torch.softmax(logits_internal.detach(), dim=1)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits_internal.view(-1), labels.view(-1))
                outputs.update({"loss": loss})
            else:
                loss_fct = CrossEntropyLoss()
                if self.config.internal_classifier_all_layers:
                    loss_internal = 0
                    sum_layer_inds = 0
                    for i in range(len(logits_internal)):
                        loss_internal += (i+1) * loss_fct(logits_internal[i].view(-1, self.num_labels), labels.view(-1))
                        sum_layer_inds += (i+1)
                    loss_internal /= sum_layer_inds
                else:
                    loss_internal = loss_fct(logits_internal.view(-1, self.num_labels), labels.view(-1))
 
                if not self.training:
                    # if the maximum probability is greater than the threshold, then we predict the class with the maximum probability
                    if self.config.internal_classifier_all_layers:
                        outputs["internal_prediction"] = False
                        for i in range(len(logits_internal)):
                            if probs[i].max(dim=1)[0].item() > self.internal_classifier_thresh:
                                # overwriting the predictions of full depth model
                                outputs["logits"] = logits_internal[i]
                                outputs["internal_prediction"] = True
                                outputs["depth"] = i + 1
                                preds = logits_internal[i].argmax(dim=-1)
                                corrects = (preds == labels).int()
                                outputs.update({"corrects": corrects})
                                break
                    else:
                        outputs["internal_prediction"] = False
                        if probs.max(dim=1)[0].item() > self.internal_classifier_thresh:
                            # overwriting the predictions of full depth model
                            outputs["logits"] = logits_internal
                            outputs["internal_prediction"] = True
                            preds = logits_internal.argmax(dim=-1)
                            corrects = (preds == labels).int()
                            outputs.update({"corrects": corrects})
                elif self.training:
                    outputs["loss_internal"] = loss_internal
                     
        outputs.update({"logits_internal": logits_internal})

        return outputs


class ShallowDeepDistilBert(DistilBertForSequenceClassification):
    def __init__(self, config):
        config.output_hidden_states = True
        super().__init__(config)
        assert getattr(config, "internal_classifier_layer", None) is not None
        self.config.internal_classifier_layer = config.internal_classifier_layer

        if config.internal_classifier_all_layers:
            self.internal_classifiers = []
            for i in range(self.config.internal_classifier_layer + 1):
                self.internal_classifiers.append(nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, config.num_labels)
                ))
            self.internal_classifiers = nn.ModuleList(self.internal_classifiers)
        else:
            self.internal_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_labels)
            )
        self.internal_classifier_thresh = config.internal_classifier_thresh
        
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

        if self.config.internal_classifier_all_layers:
            logits_internal, probs = [], []
            for i in range(1, self.config.internal_classifier_layer + 2):
                internal_hidden_state = all_hidden_states[i]
                logits_internal.append(self.internal_classifiers[i - 1](internal_hidden_state.mean(dim=1))) 
                probs.append(torch.softmax(logits_internal[-1].detach(), dim=1))
        else: 
            internal_hidden_states = all_hidden_states[self.config.internal_classifier_layer + 1]
            logits_internal = self.internal_classifier(internal_hidden_states.mean(dim=1))
            probs = torch.softmax(logits_internal.detach(), dim=1)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits_internal.view(-1), labels.view(-1))
                outputs.update({"loss": loss})
            else:
                loss_fct = CrossEntropyLoss()
                if self.config.internal_classifier_all_layers:
                    loss_internal = 0
                    sum_layer_inds = 0
                    for i in range(len(logits_internal)):
                        loss_internal += (i+1) * loss_fct(logits_internal[i].view(-1, self.num_labels), labels.view(-1))
                        sum_layer_inds += (i+1)
                    loss_internal /= sum_layer_inds
                else:
                    loss_internal = loss_fct(logits_internal.view(-1, self.num_labels), labels.view(-1))
 
                if not self.training:
                    # if the maximum probability is greater than the threshold, then we predict the class with the maximum probability
                    if self.config.internal_classifier_all_layers:
                        outputs["internal_prediction"] = False
                        for i in range(len(logits_internal)):
                            if probs[i].max(dim=1)[0].item() > self.internal_classifier_thresh:
                                # overwriting the predictions of full depth model
                                outputs["logits"] = logits_internal[i]
                                outputs["internal_prediction"] = True
                                outputs["depth"] = i + 1
                                preds = logits_internal[i].argmax(dim=-1)
                                corrects = (preds == labels).int()
                                outputs.update({"corrects": corrects})
                                break
                    else:
                        outputs["internal_prediction"] = False
                        if probs.max(dim=1)[0].item() > self.internal_classifier_thresh:
                            # overwriting the predictions of full depth model
                            outputs["logits"] = logits_internal
                            outputs["internal_prediction"] = True
                            preds = logits_internal.argmax(dim=-1)
                            corrects = (preds == labels).int()
                            outputs.update({"corrects": corrects})
                elif self.training:
                    outputs["loss_internal"] = loss_internal
                     
        outputs.update({"logits_internal": logits_internal})

        return outputs
