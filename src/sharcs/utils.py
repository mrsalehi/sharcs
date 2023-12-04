import argparse
from time import perf_counter
import wandb 
import numpy as np
import io
import datetime
import os
import time
from collections import defaultdict, deque
import datetime
from pathlib import Path
import shutil
import json
from collections import Counter
from loguru import logger
from torch.utils.data import DataLoader, SequentialSampler
import sys
import pandas as pd
from custom_transformers.modeling_bert import AdaptiveLinear, BertForQuestionAnswering
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from tqdm import tqdm
import pickle
from custom_transformers import glue_convert_examples_to_features as convert_examples_to_features
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset
from custom_transformers import glue_output_modes as output_modes
import random
from data import OnlineBucketsTensorDataset

from custom_transformers import glue_processors as processors



def convert_distilbert_ckpt(ckpt_fpath, delete_classifier=False):
    mapping = {
        "attention.q_lin.weight": "attention.self.query.weight",
        "attention.q_lin.bias": "attention.self.query.bias",
        "attention.k_lin.weight": "attention.self.key.weight",
        "attention.k_lin.bias": "attention.self.key.bias",
        "attention.v_lin.weight": "attention.self.value.weight",
        "attention.v_lin.bias": "attention.self.value.bias",
        "attention.out_lin.weight": "attention.output.dense.weight",
        "attention.out_lin.bias": "attention.output.dense.bias",
        "sa_layer_norm.weight": "attention.output.LayerNorm.weight",
        "sa_layer_norm.bias": "attention.output.LayerNorm.bias",
        "ffn.lin1.weight": "intermediate.dense.weight",
        "ffn.lin1.bias": "intermediate.dense.bias",
        "ffn.lin2.weight": "output.dense.weight",
        "ffn.lin2.bias": "output.dense.bias",
        "output_layer_norm.weight": "output.LayerNorm.weight",
        "output_layer_norm.bias": "output.LayerNorm.bias",
    }
    sd = torch.load(ckpt_fpath, map_location="cpu")
    sd_copy = sd.copy()
    for k, v in sd.items():
        if k.startswith("distilbert.transformer.layer."):
            new_k = k.replace("distilbert.transformer.layer.", "distilbert.layer.")
            for m1, m2 in mapping.items():
                if m1 in k: 
                    new_k = new_k.replace(m1, m2)
                    break
            sd_copy[new_k] = v
            del(sd_copy[k])
        elif k.startswith("vocab_"):
            del(sd_copy[k])
        elif k.startswith("distilbert.embeddings."):
            sd_copy[k.replace("distilbert.embeddings.", "embeddings.")] = v
            del(sd_copy[k])
        elif k.startswith("pre_classifier."):
            pass
        elif k.startswith("classifier."):
            if delete_classifier:
                del(sd_copy[k])
            else:
                pass
        
    for key in sd_copy.keys():
        print(key)

    torch.save(sd_copy, ckpt_fpath)


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()


def remove_pads(batch, args):
    unique_ids = batch[0]
    input_ids = batch[1]
    attention_mask = batch[2]
    labels = batch[4]
    token_type_ids = batch[3]
    
    # keep non zero elements in input_ids
    if ("bert" in args.model_type or "sharcstranskimer" in args.model_type) and not "roberta" in args.model_type:
        input_ids = input_ids[input_ids != 0].unsqueeze(0)
        attention_mask = attention_mask[attention_mask != 0].unsqueeze(0)
        token_type_ids = token_type_ids[:,:input_ids.shape[1]]
        input_length = input_ids.shape[1]
    # elif args.model_type in {"roberta", "shallow_deep_roberta", "pabee_roberta", "branchy_roberta"}:
    elif "roberta" in args.model_type:
        input_ids = input_ids[input_ids != 1].unsqueeze(0)  # apparently pads in roberta are 1..
        attention_mask = attention_mask[attention_mask != 0].unsqueeze(0)
        token_type_ids = token_type_ids[:,:input_ids.shape[1]]
        input_length = input_ids.shape[1]
    
    return (unique_ids, input_ids, attention_mask, token_type_ids, labels), input_length


def measure_qqp_latency(args, model, tokenizer, num_warmup=20):
    """
    During training, every time you get a new best checkpoint, you can use this function to measure the latency of the model 
    1. Loads the model from the give ckpt
    2. Measures the latency of the model on cpu with batch size 1
    
    We are going to use this function later to comapre our network with DynaBERT
    """
    # create a copy of the model on cpu
    eval_task_names = (args.task_name,)
    if hasattr(model, "module"):
        model = model.module

    model = model.cpu()
    model = model.eval()

    for eval_task in eval_task_names:
        eval_dataset = load_and_cache_examples(
            args, 
            eval_task, 
            tokenizer, 
            evaluate=True,
        )
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.per_gpu_eval_batch_size)

        latencies = [] 
        with torch.no_grad(): 
            for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                # batch = tuple(t.to(args.device) for t in batch) 
                inputs = {'input_ids': batch[1], 'attention_mask': batch[2], 'labels': batch[4]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[3] if args.model_type in ['bert'] else None 
                start = perf_counter()
                outputs = model(**inputs)
                end = perf_counter()
                latencies.append(end - start)
    
    logger.info(f"Avg. latency of the model on cpu with batch size 1 is {np.mean(latencies) * 1000} ms.")
    logger.info(f"Sum of latencies of the model on cpu with batch size 1 is {np.sum(latencies)} seconds.")
    # model.to(device)


def measure_mnli_latency(args, model, device, tokenizer, num_warmup=20):
    """
    During training, every time you get a new best checkpoint, you can use this function to measure the latency of the model 
    1. Loads the model from the give ckpt
    2. Measures the latency of the model on cpu with batch size 1
    
    We are going to use this function later to comapre our network with DynaBERT
    """
    # create a copy of the model on cpu
    eval_task_names = (args.task_name,)
    if hasattr(model, "module"):
        model = model.module

    model = model.cpu()
    model = model.eval()

    for eval_task in eval_task_names:
        eval_dataset = load_and_cache_examples(
            args, 
            eval_task, 
            tokenizer, 
            evaluate=True,
        )
        eval_sampler = SequentialSampler(eval_dataset)
        
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.per_gpu_eval_batch_size)

        latencies = [] 
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # batch = tuple(t.to(args.device) for t in batch) 
                inputs = {'input_ids': batch[1], 'attention_mask': batch[2], 'labels': batch[4]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[3] if args.model_type in ['bert'] else None 
                start = perf_counter()
                outputs = model(**inputs)
                end = perf_counter()
                latencies.append(end - start)
    
    logger.info(f"Avg. latency of the model on cpu with batch size 1 is {np.mean(latencies) * 1000} ms.")
    model.to(device)


def create_binary_width_label(width_mult_list, query_widths):
    """
    width_mult_list: a list of width multipliers that are used in the model. In most cases [0.25, 0.5, 1.0]
    query_widths: a query list for which the label should be returned.

    a label is a binary vector of size len(width_mult_list) and element i is 1 if width_mult_list[i] is in query_widths
    example: 
    width_mult_list: [0.25, 0.5, 1.0]
    query_widths: [0.25, 0.5]
    returns [1, 1, 0]
    """
    return torch.tensor([1 if width in query_widths else 0 for width in width_mult_list]).long()


def create_dynabert_subwidth_checkpoints(tasks, ckpt_root_path, width_mults):
    # read the ckpt file
    for task in tasks:
        for width_mult in width_mults:
            ckpt_fpath = os.path.join(ckpt_root_path, task, "best/pytorch_model.bin")
            state_dict = torch.load(ckpt_fpath, map_location="cpu")
            new_state_dict = state_dict.copy()
            for k, v in state_dict.items():
                # for key and query and value make the second dim of the weight slimmer
                if "attention.self" in k:
                    shape = v.shape
                    if "weight" in k: 
                        new_state_dict[k] = v[:int(shape[0] * width_mult), :]
                    elif "bias" in k:
                        new_state_dict[k] = v[:int(shape[0] * width_mult)]
                elif "attention.output.dense" in k:
                    shape = v.shape
                    if "weight" in k:
                        new_state_dict[k] = v[:, :int(shape[1] * width_mult)] 
                elif "intermediate.dense" in k:
                    shape = v.shape
                    if "weight" in k:
                        new_state_dict[k] = v[:int(shape[0] * width_mult), :] 
                    elif "bias" in k:
                        new_state_dict[k] = v[:int(shape[0] * width_mult)]
                elif "output.dense" in k: 
                    shape = v.shape 
                    if "weight" in k:
                        new_state_dict[k] = v[:, :int(shape[1] * width_mult)]
        
            if width_mult == 0.25:
                width_mult_str = "025"
            elif width_mult == 0.5:
                width_mult_str = "05"
            elif width_mult == 0.75:
                width_mult_str = "075"
            elif width_mult == 1.0:
                width_mult = "1"
            
            # creating the model with fine tuned classifier 
            os.makedirs(f"/home/user/adaptive-width/models/dynabert/{task}_{width_mult_str}_width", exist_ok=True)
            torch.save(new_state_dict, f"/home/user/adaptive-width/models/dynabert/{task}_{width_mult_str}_width/pytorch_model.bin")
            shutil.copy(
                "/home/user/adaptive-width/models/dynabert/MNLI_025_width/tokenizer.json",
                f"/home/user/adaptive-width/models/dynabert/{task}_{width_mult_str}_width/tokenizer.json"
            )
            shutil.copy(
                "/home/user/adaptive-width/models/dynabert/MNLI_025_width/vocab.txt",
                f"/home/user/adaptive-width/models/dynabert/{task}_{width_mult_str}_width/vocab.txt"
            )

            # creating the model without fine tuned classifier
            del(new_state_dict['classifier.weight'])
            del(new_state_dict['classifier.bias'])
            os.makedirs(f"/home/user/adaptive-width/models/dynabert/{task}_{width_mult_str}_width_no_classifier", exist_ok=True)
            torch.save(new_state_dict, f"/home/user/adaptive-width/models/dynabert/{task}_{width_mult_str}_width_no_classifier/pytorch_model.bin") 
            shutil.copy(
                "/home/user/adaptive-width/models/dynabert/MNLI_025_width/tokenizer.json",
                f"/home/user/adaptive-width/models/dynabert/{task}_{width_mult_str}_width_no_classifier/tokenizer.json"
            )
            shutil.copy(
                "/home/user/adaptive-width/models/dynabert/MNLI_025_width/vocab.txt",
                f"/home/user/adaptive-width/models/dynabert/{task}_{width_mult_str}_width_no_classifier/vocab.txt"
            )


def write_num_samples_per_width(args, num_samples_per_width, epoch, result_dir):
    """
    This is just for debugging purposes 
    """
    # find the process number
    if args.distributed:
        dist.barrier()

    rank = args.rank 
    # create a dir under the result dir to save the number of samples per width info to that
    dir_ = Path(result_dir) / f"rank_{rank}"
    dir_.mkdir(exist_ok=True, parents=True)

    # dumping the samples per width
    with open(dir_ / f"num_samples_per_width_epoch{epoch}.json", "w") as f:
        json.dump(num_samples_per_width, f)
    
    if args.distributed:
        dist.barrier()


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def check_conditions(args):
    if args.use_router:
        assert args.adaptive_layer_idx is not None
        if args.do_train:
            assert args.train_router_window_size > 0
    
    if args.remove_pads_in_eval:
        assert args.measure_flops
    
    if args.measure_flops:
        assert args.remove_pads_in_eval 
    
    if args.write_val_preds:
        assert args.task_name.lower() == "mnli"
        
    if not args.init_private_layernorms_from_scratch: # may be later you can remove this condition
        assert not args.weighted_dim_reduction
    
    if args.weighted_dim_reduction:
        assert args.init_private_layernorms_from_scratch
         
    if args.model_type == "distilbert" and args.use_router:
        assert args.gradient_accumulation_steps == 2, "for some reason distributed training is not working for distilbert. Instead do it with gradient accumulation of 2."
    
    if args.measure_latency:
        # NOTE: When measuring latency, will just load the ckpt and will not do any training
        assert not args.do_train
    

def update_config(config, args):
    # for arg in vars(args):
    #     if arg in config:
    #         config[arg] = getattr(args, arg)
    setattr(config, 'width_mult_list', args.width_mult_list)
    setattr(config, 'adaptive_layer_idx', args.adaptive_layer_idx)
    setattr(config, 'hidden_size_router', args.hidden_size_router)
    setattr(config, 'use_router', args.use_router)
    setattr(config, 'train_router_window_size', args.train_router_window_size)
    setattr(config, "lambda_loss_task", args.lambda_loss_task)
    setattr(config, "lambda_loss_router", args.lambda_loss_router)
    setattr(config, "repeat_smaller_sized_buckets", args.repeat_smaller_sized_buckets)
    setattr(config, "weighted_dim_reduction", args.weighted_dim_reduction)
    setattr(config, "model_type", args.model_type)
    setattr(config, "use_entropy_hardness", args.use_entropy_hardness)

    # setattr(config, "internal_classifier_all_layers", args.internal_classifier_all_layers)


def get_eval_data(args, task):
    processor = processors[task]()
    examples = processor.get_dev_examples(args.data_dir)

    return examples


def convert_transkimer_skim_predictor(ckpt_fpath):
    wts = torch.load(ckpt_fpath, map_location="cpu")
    new_wts = {}
    key_map = {
        "predictor.0.weight": "LayerNorm.weight",
        "predictor.0.bias": "LayerNorm.bias",
        "predictor.1.weight": "linear.weight",
        "predictor.1.bias": "linear.bias",
        "predictor.2.weight": "LayerNorm2.weight",
        "predictor.2.bias": "LayerNorm2.bias",
        "predictor.4.weight": "linear2.weight",
        "predictor.4.bias": "linear2.bias",
    }
    for key in wts.keys():
        if "bert.encoder.skim_predictors" in key:
            layer_idx = key.split(".")[3]
            second_part = ".".join(key.split(".")[4:])
            second_part = key_map[second_part]
            new_key = f"bert.encoder.skim_predictors.{layer_idx}.{second_part}"
            new_wts[new_key] = wts[key]
        else:
            new_wts[key] = wts[key]
    torch.save(new_wts, ckpt_fpath.replace("pytorch_model.bin", "pytorch_model_converted.bin")) 


def load_and_cache_examples(args, task, tokenizer, evaluate=False, offline_width_label_fpath=None):
    features_path = "/home/user/adaptive-width/glue"

    # if args.model_type in {"bert", "distilbert", "shallow_deep_bert", "branchy_bert", "pabee_bert"}:
    if "bert" in args.model_type and not "roberta" in args.model_type:
        pkl_fname = f"features_{task}_eval.pkl" if evaluate else f"features_{task}_train.pkl"
    elif args.model_type == "sharcstranskimer":
        pkl_fname = f"features_{task}_eval.pkl" if evaluate else f"features_{task}_train.pkl"
    # elif args.model_type in {"roberta", "shallow_deep_roberta", "pabee_roberta", "branchy_roberta"}:
    elif "roberta" in args.model_type:
        pkl_fname = f"features_{task}_roberta_eval.pkl" if evaluate else f"features_{task}_roberta_train.pkl"
    elif args.model_type in {"albert"}:
        pkl_fname = f"features_{task}_albert_eval.pkl" if evaluate else f"features_{task}_albert_train.pkl"
    else:
        raise NotImplementedError
     
    features_path = os.path.join(features_path, pkl_fname)
    if os.path.exists(features_path):
        logger.info(f"Loading features from cached file {pkl_fname}")
        with open(features_path, 'rb') as f:
            features = pickle.load(f)

        # if args.name == "debug":
            # features = features[:100]

        processor = processors[task]()
        output_mode = output_modes[task]
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            label_list[1], label_list[2] = label_list[2], label_list[1]
    else: 
        processor = processors[task]()
        output_mode = output_modes[task]
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                args=args,
                                                task=task,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                evaluate=evaluate,
        )

    # Convert to Tensors and build dataset
    unique_ids = torch.arange(len(features))
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    
    if args.use_router:
        if not evaluate:
            dataset = OnlineBucketsTensorDataset(args, unique_ids, all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        else:
            dataset = TensorDataset(unique_ids, all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    elif offline_width_label_fpath is not None:
        width_mult_to_idx = {width_mult: idx for idx, width_mult in enumerate(args.width_mult_list)}
        # slimmable network
        width_labels = json.load(open(offline_width_label_fpath, 'r'))
        width_labels = torch.tensor([width_labels[str(id_.item())] for id_ in unique_ids], dtype=torch.float)
        width_labels_ids = torch.tensor([width_mult_to_idx[width_label.item()] for width_label in width_labels], dtype=torch.long)
        if args.use_router:
            dataset = TensorDataset(unique_ids, all_input_ids, all_attention_mask, all_token_type_ids, all_labels, width_labels, width_labels_ids)
        else:
            dataset = TensorDataset(unique_ids, all_input_ids, all_attention_mask, all_token_type_ids, all_labels, width_labels)
    else:
        dataset = TensorDataset(unique_ids, all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def reorder_neuron_head(model, head_importance, neuron_importance):
    """ reorder neurons based on their importance.
        Arguments:
            model: bert model
            head_importance: 12*12 matrix for head importance in 12 layers
            neuron_importance: list for neuron importance in 12 layers.
    """
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)

    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        base_model.encoder.layer[layer].attention.reorder_heads(idx)
        # reorder neurons
        idx = torch.sort(current_importance, descending=True)[-1]
        base_model.encoder.layer[layer].intermediate.reorder_neurons(idx)
        base_model.encoder.layer[layer].output.reorder_neurons(idx)


def compute_neuron_head_importance(args, model, tokenizer):
    """ This method shows how to compute:
        - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
    """
    # prepare things for heads
    model = model.module if hasattr(model, 'module') else model
    model.apply(lambda m: setattr(m, 'mode', 'eval'))
    base_model = getattr(model, model.base_model_prefix, model)
    n_layers, n_heads = base_model.config.num_hidden_layers, base_model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    head_mask = torch.ones(n_layers, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    # collect weights
    intermediate_weight = []
    intermediate_bias = []
    output_weight = []
    for name, w in model.named_parameters():
        if 'intermediate' in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'output' in name and 'attention' not in name:
            if w.dim() > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).to(args.device))

    model.to(args.device)

    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + 'MM') if args.task_name == "mnli" else (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, 1)  # let's assume that number of gpus are 1
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)
            # input_ids, input_mask, _, label_ids = batch
            unique_ids, input_ids, input_mask, _, label_ids = batch

            # segment_ids = batch[2] if args.model_type=='bert' else None  # RoBERTa does't use segment_ids
            segment_ids = batch[3] if args.model_type=='bert' else None  # RoBERTa does't use segment_ids

            # calculate head importance
            outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids,
                            head_mask=head_mask)
            # loss = outputs[0]
            loss = outputs['loss']

            loss.backward()
            head_importance += head_mask.grad.abs().detach()

            # calculate  neuron importance
            for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
                current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
                current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()

    return head_importance, neuron_importance


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs


def set_width_mult(model, config, width_mult=None):
    """
    width_mult_last_non_adaptive_layer_output: the width mult of the dense layer in BertOutput of last layer
    width_mult_adaptive_layers: width mult used in all of the adaptive layers
    """ 
    assert width_mult is not None

    if hasattr(model, 'module'):
        model = model.module
     
    if config.model_type in {"tinybert", "sharcstranskimer"}:
        model_ = getattr(model, "bert")
    else:
        model_ = getattr(model, config.model_type) 
    
    if config.model_type == "distilbert":
        encoder = model_
    else:
        encoder = model_.encoder 
    
    if not hasattr(encoder, "skim_predictors"):
        encoder.skim_predictors = [None] * len(encoder.layer)
      
    for layer in encoder.layer:
        if layer.layer_type in "normal":
            layer.apply(lambda m: setattr(m, 'width_mult', 1.0)) 
        elif layer.layer_type in {"router", "last_non_adaptive_without_router"}:
            setattr(layer.output, "width_mult_adaptive_layers", width_mult)
        elif layer.layer_type == "adaptive":
            layer.apply(lambda m: setattr(m, 'width_mult', width_mult))
    
 
    if config.model_type in {"bert", "tinybert", "albert", "sharcstranskimer"}:
        model_.pooler.apply(lambda m: setattr(m, 'width_mult', width_mult))
    elif config.model_type == "roberta":
        model.classifier.dense.apply(lambda m: setattr(m, 'width_mult', width_mult))
    elif config.model_type == "distilbert":
        model.pre_classifier.apply(lambda m: setattr(m, 'width_mult', width_mult))
    
    if isinstance(model, BertForQuestionAnswering):
        model.qa_outputs.apply(lambda m: setattr(m, 'width_mult', width_mult))


def reorder_neuron_head(args, model, head_importance, neuron_importance):
    """ reorder neurons based on their importance.
        Arguments:
            model: bert model
            head_importance: 12*12 matrix for head importance in 12 layers
            neuron_importance: list for neuron importance in 12 layers.
    """
    task = args.task_name
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)

    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        base_model.encoder.layer[layer].attention.reorder_heads(idx)
        # reorder neurons
        idx = torch.sort(current_importance, descending=True)[-1]
        base_model.encoder.layer[layer].intermediate.reorder_neurons(idx)
        base_model.encoder.layer[layer].output.reorder_neurons(idx)

    # save model state dict to a file
    save_dir = Path(f"/home/user/adaptive-width/models/bert/{task}_reordered_heads")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def convert_albert_to_bert_ckpt(ckpt_fpath):
    sd = torch.load(ckpt_fpath, map_location='cpu')
    n_layers = 12
    mapping = {
        "attention.query.weight": "attention.self.query.weight",
        "attention.query.bias": "attention.self.query.bias",

        "attention.key.weight": "attention.self.key.weight",
        "attention.key.bias": "attention.self.key.bias",

        "attention.value.weight": "attention.self.value.weight",
        "attention.value.bias": "attention.self.value.bias",

        "attention.LayerNorm.weight": "attention.output.LayerNorm.weight",
        "attention.LayerNorm.bias": "attention.output.LayerNorm.bias",

        "attention.dense.weight": "attention.output.dense.weight",
        "attention.dense.bias": "attention.output.dense.bias",

        "ffn.weight": "intermediate.dense.weight",
        "ffn.bias": "intermediate.dense.bias",

        "ffn_output.weight": "output.dense.weight",
        "ffn_output.bias": "output.dense.bias",

        "full_layer_layer_norm.weight": "output.LayerNorm.weight",
        "full_layer_layer_norm.bias": "output.LayerNorm.bias",
    }

    new_sd = {}

    for key in sd.keys():
        if key.startswith("albert.embeddings."):
            new_sd[key] = sd[key]
        elif key.startswith("albert.encoder.embedding_hidden_mapping"):
            new_sd[key.replace("albert.encoder", "albert")] = sd[key]
        elif key.startswith('albert.pooler'):
            new_sd[key.replace('albert.pooler', 'albert.pooler.dense')] = sd[key] 
        elif key.startswith('classifier'):
            new_sd[key] = sd[key]

    for i in range(n_layers):
        for key in sd.keys():
            if key.startswith('albert.encoder.albert_layer_groups'):
                new_key = key.replace('albert.encoder.albert_layer_groups.0.albert_layers.0', 'albert.encoder.layer.{}'.format(i))
                for k, v in mapping.items():
                    new_key = new_key.replace(k, v)
                new_sd[new_key] = sd[key]

    
    # save new_sd
    torch.save(new_sd, ckpt_fpath)


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    if is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)

    return device


def get_last_checkpoint(checkpoint_dir):
    files = Path(checkpoint_dir).glob('*.pt')
    # sort the files based on getmtime and then return the last created file
    files = sorted(files, key=os.path.getmtime)
    last_ckpt = files[-1] if files else None
    return last_ckpt


def save_pred_results_per_width_train(args, train_result, result_dir, filename):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename, args.rank))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    json.dump(train_result, open(result_file,'w'))

    if args.distributed:
        dist.barrier()
        
    if is_master(args):
        result = defaultdict(list)
        for rank in range(args.world_size):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            for width_ in args.width_mult_list:
                result[width_].extend(res[str(width_)])  # so stupid that when you read it the keys are strings
            os.remove(result_file)
        json.dump(result, open(final_result_file,'w'))
    
    if args.distributed:
        dist.barrier()


def save_and_update_train_history(args, train_result, result_dir, filename, epoch, train_history, best_epoch_with_router):
    """
    1. Gets the confidence scores from all processes and save them in files
    2. The main process reads the file and write it into a single file called train_confidence_scores_ep<epoch>.json
    3. The main process also updates the train_history and assign the new buckets
    """
    # saving the train history per process first. Writing to files is what dynaBERT does.
    # Could do it with gathering instead of writing to files.
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename, args.rank))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    json.dump(train_result, open(result_file,'w'))
    
    if args.distributed:
        dist.barrier()
 
    if is_master(args):
        result = {}
        for rank in range(args.world_size):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            # convert keys into int
            res = {int(k): v for k, v in res.items()}
            result.update(res) 
            os.remove(result_file)

        json.dump(result, open(final_result_file,'w')) 
        logger.info(f"save_and_update_train_history: Length of results saved in save_confidence_scores: {len(result)}")

        new_train_history = train_history.copy()
        # n_epochs_in_train_history = len(next(iter(train_history.values())))

        # update the train history based on the confidences of the epoch
        for uid in result.keys():
            new_train_history[uid][epoch] = result[uid]
        
        n_epochs_in_train_history = epoch + 1  # including 0th epoch and this one that has finished

        freeze_bucket_0_25 = getattr(args, 'freeze_bucket_0_25', False)
        have_frozen_bucket_0_25 = getattr(args, 'have_frozen_bucket_0_25', False)  # Is this the first time setting the buckets after reaching the patience?

        if args.model_type == 'distilbert':
            logger.info("after creating new train history in save_and_update_train_history.")

        if n_epochs_in_train_history >= args.train_router_window_size:
            confidence_threshs = args.confidence_threshs
            if args.use_entropy_hardness:
                entropy_threshs = args.confidence_threshs

            logger.info(f"save_confidence_scores_train: epoch {epoch}. Samples in buckets 0.25, 0.5 and 1.0 will bounce."
                        "You should see this msg only before freezing bucket 0.25!!")
            
            buckets_for_next_epoch = {width_mult: [] for width_mult in args.width_mult_list}
            uid_to_width_mults = {}  # will be used in creating labels to train the router

            for uid, confidence_history in new_train_history.items():

                if args.use_entropy_hardness:
                    entropy_history = confidence_history

                buckets_for_this_sample = []
                # just taking the confidences of last window_size epochs
                if not args.use_entropy_hardness:
                    recent_confidence_history = {ep: conf for ep, conf in confidence_history.items() \
                        if ep >= n_epochs_in_train_history - args.train_router_window_size}
                else:
                    recent_entropies = {ep: conf for ep, conf in entropy_history.items() \
                        if ep >= n_epochs_in_train_history - args.train_router_window_size} 

                if not args.use_entropy_hardness:
                    for width_, thresh in confidence_threshs.items():
                        if min(recent_confidence_history.values()) >= thresh[0] and max(recent_confidence_history.values()) < thresh[1]:
                            buckets_for_this_sample.append(width_)
                else:
                    for width_, thresh in entropy_threshs.items():
                        if min(recent_entropies.values()) >= thresh[0] and max(recent_entropies.values()) < thresh[1]:
                            buckets_for_this_sample.append(width_)

                # If the sample is not assigned any buckets then put it in bucket 1.0
                if len(buckets_for_this_sample) == 0:
                    buckets_for_this_sample.append(1.0)

                uid_to_width_mults[uid] = buckets_for_this_sample

                width_ = None
                if len(buckets_for_this_sample) > 1:
                    # sample one of the assigned widths
                    width_ = random.choice(buckets_for_this_sample)
                else:
                    width_ = buckets_for_this_sample[0]
                
                buckets_for_next_epoch[width_].append(uid)
 
            json.dump(buckets_for_next_epoch, open(os.path.join(result_dir, f'online_buckets_for_ep{epoch+1}.json'),'w'))
            json.dump(new_train_history, open(os.path.join(result_dir, f'train_history_ep{epoch}.json'),'w'))
            json.dump(uid_to_width_mults, open(os.path.join(result_dir, f'uid_to_width_mults_for_ep{epoch+1}.json'),'w'))
     
    if args.distributed:
        dist.barrier()
 
    return None if not is_master(args) else new_train_history


def save_pred_results_train(args, train_result, result_dir, filename):
    # for qid, sample_loss in train_result.items():
    #     # result[str(qid.item())] = '%.3f' % sample_loss.item()
    #     result[str(qid.item())] = sample_loss.item()
        # floating point with 2 decimal places
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename, args.rank))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    json.dump(train_result, open(result_file,'w'))

    if args.distributed:
        dist.barrier()
        
    if is_master(args):
        result = {}
        for rank in range(args.world_size):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result.update(res) 
            os.remove(result_file)

        json.dump(result, open(final_result_file,'w')) 
    
    if args.distributed:
        dist.barrier()


def gather_tensor(tensor, args):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(args.world_size)]
    if args.distributed:
        dist.all_gather(gathered_tensors, tensor)
        return torch.stack(gathered_tensors, dim=0).sum(dim=0)
    else:
        return tensor 


def analyze_prob_samples_that_change_bucket(result_dir):
    src_bucket = 1
    dst_bucket = 0.25
    for ep in range(1, 60):
        # find the samples that are in bucket i in epoch ep and are in bucket j in epoch ep+1
        pass 


@torch.no_grad()
def evaluation_latency(model, device, data_loader, tokenizer, config, label2ans, args):
    """
    This one just measure the latency for the case where you don't use the width classifier in the pipeline
    """
    # test
    # override_width_multiplier = None if args.override_width_multiplier == 0. else args.override_width_multiplier 
    assert args.no_width_classifier
    if args.override_width_multiplier != 0.:
        override_width_multiplier = args.override_width_multiplier
    elif args.no_width_classifier:
        override_width_multiplier = None # I am going to change the override_width_multiplier for each sample in the validation loop below
    else:
        override_width_multiplier = None
 
    if override_width_multiplier:
        enable_adaptive=True
    else:
        #even if offline widths are used then if disable adaptive width is on then enable adaptive is off 
        enable_adaptive = not args.disable_adaptive_width 
    
    model.eval()
    result = []
    result_predicted_widths = {}
    logger.info("Measure latency (no width classifier in the piepline)")

    # warmup phase
    for i in range(100):
        dummy_image = torch.randn((1, 3, 384, 384)).to(device)
        dummy_question_input = tokenizer(['Is this train at the station?'], padding='longest', return_tensors='pt').to(device)
        dummy_labels = torch.zeros((1, 3129)).to(device)
        dummy_batch_width_multiplier = torch.tensor(1.).to(device)
        override_width_multiplier = dummy_batch_width_multiplier.item()
        _, _ = model(dummy_image, dummy_question_input.input_ids, dummy_question_input.attention_mask, 
                     dummy_labels, enable_adaptive=enable_adaptive, override_width_multiplier=override_width_multiplier, 
                     width_multipliers=None, train=False, k=config['k_test'])

    timings= []
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for step, batch in enumerate(tqdm(data_loader, disable=not is_master(args))):
        if args.offline_adaptive_width:  # which should be always true in this function
            image, question, question_id, labels, batch_width_multiplier = batch
            # override_width_multiplier = batch_width_multiplier[0].item()
            override_width_multiplier = 0.25
        
        image, labels = image.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
        # starter.record()
        start = perf_counter()
        preds, predicted_width_multipliers = model(image, question_input.input_ids, question_input.attention_mask, 
                                                   labels, enable_adaptive=enable_adaptive, override_width_multiplier=override_width_multiplier,
                                                   width_multipliers=None, train=False, k=config['k_test']) 
        # ender.record()
        end = perf_counter()
        # torch.cuda.synchronize()
        curr_time = end - start
        # curr_time = starter.elapsed_time(ender)
        timings.append((override_width_multiplier, curr_time))
    
    mean_syn = np.sum([el[1] for el in timings]) / len(timings)
    std_syn = np.std([el[1] for el in timings])

    logger.info(f"Mean latency: {mean_syn} ms, std: {std_syn} ms")

 
if __name__ == "__main__":
    # Testing SHARCS + Transkimmer
    convert_transkimer_skim_predictor(
        ckpt_fpath="/home/user/Transkimmer/model/glue/qnli/pytorch_model.bin")