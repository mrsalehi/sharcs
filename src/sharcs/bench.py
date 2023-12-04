from fvcore.nn import (
    ActivationCountAnalysis,
    FlopCountAnalysis,
    flop_count_str,
    flop_count_table,
    )
from collections import Counter
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from torch.nn import TransformerEncoderLayer
from bench_utils import CustomTransformerEncoderLayer
import torch
from typing import List
from torch import nn
import numpy as np
from loguru import logger
import sys
from fvcore.nn import flop_count_table
import warnings
from itertools import chain

warnings.filterwarnings("ignore", category=UserWarning, module="fvcore")

import math


def bench_flops_dyna(width_mult):
    """
    12 layers of dynabert network with width mult of width_mult
    n_val: number of validation samples
    """
    input_dim = 768 # this is original input dim without any adaptive width  
    seq_len = 128    

    mlp_ratio = int(4 * width_mult)
    nhead = int(12 * width_mult)
    
    original_input = torch.randn(1, seq_len, input_dim)
    # block = TransformerEncoderLayer(
    block = CustomTransformerEncoderLayer(
        d_model=input_dim, 
        nhead=nhead, 
        dim_head=64,
        dim_feedforward=mlp_ratio * input_dim, 
        dropout=0.1, 
        activation='gelu',
        batch_first=True,
        device='cpu'
    )

    flops = FlopCountAnalysis(block, original_input)
    flops = flops.unsupported_ops_warnings(False)
    print(flop_count_table(flops))
    original_flops = flops.total()
    original_flops_total = original_flops * 12

    return original_flops_total


def bench_flops_original():
    """
    widths: list of width multlpliers that should be applied to the dimensions of layer layer_index. each eleement is for one sample in the validation set.
    layer_index: the index of the layer

    the point is that you don't need to crate a network with that exact width multiplier applied. You could just create a block
    and then  compute the ration between the flops of that block and the flops of the original block. then you should multiply 
    this by the number of blocks after the original block.
    """
    input_dim = 768 # this is original input dim without any adaptive width  
    seq_len = 128    

    mlp_ratio = 4
    nhead = 12
    
    original_input = torch.randn(1, seq_len, input_dim)
    block = CustomTransformerEncoderLayer(
        d_model=input_dim, 
        nhead=nhead, 
        dim_feedforward=mlp_ratio * input_dim, 
        dropout=0.1, 
        dim_head=64,
        activation='gelu',
        batch_first=True,
        device='cpu'
    )

    flops = FlopCountAnalysis(block, original_input)
    flops = flops.unsupported_ops_warnings(False)
    print(flop_count_table(flops))
    original_flops = flops.total()
    original_flops_total = original_flops * 12

    return original_flops_total


def bench_flops_adaptive_with_router(width_mults: List, hidden_dim_router: int, n_vals: List, adaptive_layer_idx: int):
    """
    compute the average number of flops for adaptive network with different width multipliers
    router_layer_idx: the index of the layer that the router is located in.
    """
    assert len(n_vals) == len(width_mults)
    input_dim = 768 # this is original input dim without any adaptive width
    seq_len = 128  
    mlp_ratio = 4
    nhead = 12
    original_input = torch.randn(1, seq_len, input_dim)
    block_original = CustomTransformerEncoderLayer(
        d_model=input_dim, 
        nhead=nhead, 
        dim_feedforward=mlp_ratio * input_dim, 
        dim_head=64,
        dropout=0.1, 
        activation='gelu',
        batch_first=True,
        device='cpu'
    )

    flops = FlopCountAnalysis(block_original, original_input)
    flops = flops.unsupported_ops_warnings(False)
    original_flops = (flops.total() / 1e9) * adaptive_layer_idx * sum(n_vals)
    # print("normal layer flops:", flops.total() / 1e9)
    # print("normal layer flops") 
    # print(flop_count_table(flops))

    
    router = nn.Sequential(
        nn.Linear(input_dim, hidden_dim_router),
        nn.ReLU(),
        nn.Linear(hidden_dim_router, len(width_mults))
    )
    flops = FlopCountAnalysis(router, original_input.mean(dim=1))
    flops = flops.unsupported_ops_warnings(False)
    router_flops = (flops.total() / 1e9) * sum(n_vals)
    # print("router flops", flops.total() / 1e9)
    # print(flop_count_table(flops))

    adaptive_flops = 0
    for width_mult, n_val in zip(width_mults, n_vals): 
        mlp_ratio = int(4 * width_mult)
        nhead = int(12 * width_mult) 
        input_ = torch.randn(1, seq_len, int(input_dim * width_mult))
        print(f"adaptive layer width mult {width_mult} flops")
        block = CustomTransformerEncoderLayer(
            d_model=int(input_dim * width_mult), 
            nhead=nhead, 
            dim_feedforward=mlp_ratio * input_dim, 
            dropout=0.1, 
            dim_head=64,
            activation='gelu',
            batch_first=True,
            device='cpu'
        )
        flops = FlopCountAnalysis(block, input_)
        # print(flop_count_table(flops))
        # print(flops.total())
        # print(f"adaptive flops {width_mult}:", flops.total() / 1e9)
        adaptive_flops += (flops.total() / 1e9) * (12 - adaptive_layer_idx) * n_val
        
    total_flops = original_flops + router_flops + adaptive_flops
    avg_flops = total_flops / sum(n_vals)
    
    return avg_flops


def bench_flops_adaptive_no_router(width_mult: List, adaptive_layer_idx: int):
    """
    compute the average number of flops for adaptive network with different width multipliers
    router_layer_idx: the index of the layer that the router is located in.
    """
    input_dim = 768 # this is original input dim without any adaptive width
    seq_len = 128  
    mlp_ratio = 4
    nhead = 12
    n_layers = 12
    original_input = torch.randn(1, seq_len, input_dim)
    block_original = CustomTransformerEncoderLayer(
        d_model=input_dim, 
        nhead=nhead, 
        dim_feedforward=mlp_ratio * input_dim, 
        dim_head=64,
        dropout=0.1, 
        activation='gelu',
        batch_first=True,
        device='cpu'
    )

    flops = FlopCountAnalysis(block_original, original_input)
    flops = flops.unsupported_ops_warnings(False)
    original_flops = (flops.total() / 1e9) * adaptive_layer_idx
    # print("normal layer flops:", flops.total() / 1e9)
    # print("normal layer flops") 
    # print(flop_count_table(flops))

    adaptive_flops = 0
    mlp_ratio = int(4 * width_mult)
    nhead = int(12 * width_mult) 
    input_ = torch.randn(1, seq_len, int(input_dim * width_mult))
    print(f"adaptive layer width mult {width_mult} flops")
    block = CustomTransformerEncoderLayer(
        d_model=int(input_dim * width_mult), 
        nhead=nhead, 
        dim_feedforward=mlp_ratio * input_dim, 
        dropout=0.1, 
        dim_head=64,
        activation='gelu',
        batch_first=True,
        device='cpu'
    )
    flops = FlopCountAnalysis(block, input_)
    flops = flops.unsupported_ops_warnings(False)
    # print(flop_count_table(flops))
    adaptive_flops += (flops.total() / 1e9) * (n_layers - adaptive_layer_idx)
        
    total_flops = original_flops + adaptive_flops
    
    return total_flops


def bench_flops_dyna_adaptive_with_router(dyna_width_mult: int, adaptive_width_mults: List, hidden_dim_router: int, n_vals: List, adaptive_layer_idx: int):
    """
    give a dynaBERT network with dyna_width_mult as its width factor, and a router that makes the network adaptive at some layer,
    compute the average flops for one task. 
    """
    # first dyna-layers
    input_dim = 768 # this is original input dim without any adaptive width  
    seq_len = 128    

    mlp_ratio_dyna_layers = int(4 * dyna_width_mult)
    nhead_dyna_layers = int(12 * dyna_width_mult)
    
    original_input = torch.randn(1, seq_len, input_dim)
    block_dyna = CustomTransformerEncoderLayer(
        d_model=input_dim, 
        nhead=nhead_dyna_layers, 
        dim_head=64,
        dim_feedforward=int(mlp_ratio_dyna_layers * input_dim), 
        dropout=0.1, 
        activation='gelu',
        batch_first=True,
        device='cpu'
    )

    flops_dyna = FlopCountAnalysis(block_dyna, original_input)
    print(flop_count_table(flops_dyna))

    assert len(n_vals) == len(adaptive_width_mults)
    flops_dyna = (flops_dyna.total() / 1e9) * adaptive_layer_idx * sum(n_vals)
    
    router = nn.Sequential(
        nn.Linear(input_dim, hidden_dim_router),
        nn.ReLU(),
        nn.Linear(hidden_dim_router, len(adaptive_width_mults))
    )
    flops = FlopCountAnalysis(router, original_input.mean(dim=1))
    flops = flops.unsupported_ops_warnings(False)
    router_flops = (flops.total() / 1e9) * sum(n_vals)
    # print("router flops", flops.total() / 1e9)
    # print(flop_count_table(flops))

    adaptive_flops = 0
    for width_mult, n_val in zip(adaptive_width_mults, n_vals): 
        mlp_ratio = mlp_ratio_dyna_layers
        # nhead = int(12 * width_mult) 
        nhead = int(nhead_dyna_layers * width_mult) 
        input_ = torch.randn(1, seq_len, int(input_dim * width_mult))
        print(f"adaptive layer width mult {width_mult} flops")
        block = CustomTransformerEncoderLayer(
            d_model=int(input_dim * width_mult), 
            nhead=nhead, 
            dim_feedforward=int(mlp_ratio * input_dim * width_mult), 
            dropout=0.1, 
            dim_head=64,
            activation='gelu',
            batch_first=True,
            device='cpu'
        )
        flops = FlopCountAnalysis(block, input_)
        flops = flops.unsupported_ops_warnings(False)
        print(flop_count_table(flops))
        # print(flops.total())
        # print(f"adaptive flops {width_mult}:", flops.total() / 1e9)
        adaptive_flops += (flops.total() / 1e9) * (12 - adaptive_layer_idx) * n_val
        
    total_flops = flops_dyna + router_flops + adaptive_flops
    avg_flops = total_flops / sum(n_vals)

    return avg_flops


# def bench_flops_total(seq_lens, width_mults, args, depths=None, tokens_remained=None):
def bench_flops_total(seq_lens, width_mults, args, depths=None, **kwargs):
    if args.use_router:
        logger.info("measuring total flops of evaluation")

        if "dynabert" in args.model_dir:
            logger.info("Using router + dynabert")
            input_dim = 768 # this is original input dim without any adaptive width
            seq_len = 128
            mlp_ratio = 4
            nheads = 12
            adaptive_layer_idx = args.adaptive_layer_idx

            # TODO: for now I am going to find the dynawidth mult from the name of the file but change this later
            dyna_width_mult = float(Path(args.model_dir).stem.split("_")[1])
            mapping = {25.: 0.25, 5.: 0.5, 75.: 0.75, 1: 1.0}
            dyna_width_mult = mapping[dyna_width_mult]
            assert dyna_width_mult in [0.25, 0.5, 0.75, 1.0]
  
            block_dyna = CustomTransformerEncoderLayer(
                d_model=input_dim,
                nhead=int(nheads * dyna_width_mult),
                dim_head=64,
                dim_feedforward=int(mlp_ratio * dyna_width_mult * input_dim),
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                device='cpu'
            )
            hidden_dim_router = args.hidden_size_router 
            router = nn.Sequential(
                nn.Linear(input_dim, hidden_dim_router),
                nn.ReLU(),
                nn.Linear(hidden_dim_router, len(args.width_mult_list))
            )
             
            blocks_dyna_adaptive = {}            
            total_flops_per_width_mult = {}
            for adaptive_width_mult in np.unique(width_mults):
                total_flops_per_width_mult[adaptive_width_mult] = 0
                blocks_dyna_adaptive[adaptive_width_mult] = CustomTransformerEncoderLayer(
                    d_model=int(input_dim * adaptive_width_mult),
                    nhead=int(nheads * dyna_width_mult * adaptive_width_mult),
                    dim_head=64,
                    dim_feedforward=int(mlp_ratio * dyna_width_mult * input_dim * adaptive_width_mult),
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True,
                    device='cpu'
                )
             
            total_flops = 0
            for adaptive_width_mult, seq_len in tqdm(list(zip(width_mults, seq_lens))):
                flops_sample = 0
                original_input = torch.randn(1, seq_len, input_dim)
                flops = FlopCountAnalysis(block_dyna, original_input)
                flops = flops.unsupported_ops_warnings(False)
                # print(flop_count_table(flops_dyna))
                flops_sample += (flops.total() / 1e9) * adaptive_layer_idx

                flops = FlopCountAnalysis(router, original_input.mean(dim=1))
                flops = flops.unsupported_ops_warnings(False)
                flops_sample += (flops.total() / 1e9) 
                 
                input_ = torch.randn(1, seq_len, int(input_dim * adaptive_width_mult)) 
                flops = FlopCountAnalysis(blocks_dyna_adaptive[adaptive_width_mult], input_)
                flops = flops.unsupported_ops_warnings(False)
                # print(flop_count_table(flops))
                flops_sample += (flops.total() / 1e9) * (12 - adaptive_layer_idx)  # for now we are assuming that we are just running with dynabert adaptive width models
                total_flops_per_width_mult[adaptive_width_mult] += flops_sample
                total_flops += flops_sample
                
            return total_flops, total_flops_per_width_mult
         
        input_dim = 768 # this is original input dim without any adaptive width
        mlp_ratio = 4
        nhead = 12
        hidden_dim_router = args.hidden_size_router

        if args.model_type in {"distilbert", "tinybert"}:
            n_layers = 6
        else:
            n_layers = 12
        
        
        block_original = CustomTransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=mlp_ratio * input_dim,
            dim_head=64,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            device='cpu'
        )
        router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_router),
            nn.ReLU(),
            nn.Linear(hidden_dim_router, len(width_mults))
        )
        tokens_remained = kwargs.pop("tokens_remained", None)
        if tokens_remained is not None:
            token_len_pred_width_mult_for_skim_predictors = kwargs.pop("token_len_pred_width_mult_for_skim_predictors", None)
            skim_predictor_non_adaptive = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 2)
            )
         
        total_flops = 0

        blocks_adaptive, skim_predictors_adaptive = {}, {}

        total_flops_per_width_mult = {}
        for width_mult in np.unique(width_mults):        
            total_flops_per_width_mult[width_mult] = 0
            blocks_adaptive[width_mult] = CustomTransformerEncoderLayer(
                d_model=int(input_dim * width_mult),
                nhead=int(nhead * width_mult),
                dim_feedforward=int(mlp_ratio * input_dim * width_mult),
                dropout=0.1,
                dim_head=64,
                activation='gelu',
                batch_first=True,
                device='cpu'
            )
            if tokens_remained is not None:
                skim_predictors_adaptive[width_mult] = nn.Sequential(
                    nn.LayerNorm(int(input_dim * width_mult)),
                    nn.Linear(int(input_dim * width_mult), input_dim),
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, 2)
                )
        
        if tokens_remained is not None:
            tokens_remained_non_adaptive = Counter(chain(*[t[:args.adaptive_layer_idx] for t in tokens_remained]))
            tokens_remained_adaptive = Counter(chain(*[t[args.adaptive_layer_idx:] for t in tokens_remained]))

            # non-adaptive layers FLOPS
            for token_len, freq in tokens_remained_non_adaptive.items():
                adaptive_layer_idx = args.adaptive_layer_idx
                original_input = torch.randn(1, token_len, input_dim)
                flops = FlopCountAnalysis(block_original, original_input)
                flops = flops.unsupported_ops_warnings(False)
                total_flops += (flops.total() / 1e9) * freq
                flops = FlopCountAnalysis(skim_predictor_non_adaptive, original_input)
                flops = flops.unsupported_ops_warnings(False)
                total_flops += (flops.total() / 1e9) * freq
           
            # adaptive layers FLOPS 
            for token_len, freq in tokens_remained_adaptive.items():
                input_ = torch.randn(1, token_len, int(input_dim * width_mult))
                flops = FlopCountAnalysis(blocks_adaptive[width_mult], input_)
                flops = flops.unsupported_ops_warnings(False)
                total_flops += (flops.total() / 1e9) * freq

            # router FLOPs
            original_input = torch.randn(1, input_dim)
            flops = FlopCountAnalysis(router, original_input)
            flops = flops.unsupported_ops_warnings(False)
            total_flops += (flops.total() / 1e9) * len(seq_lens)
        
            # skim predictors FLOPs
            for width_mult, token_len in token_len_pred_width_mult_for_skim_predictors:
                input_ = torch.randn(1, token_len, int(input_dim * width_mult))
                flops = FlopCountAnalysis(skim_predictors_adaptive[width_mult], input_)
                flops = flops.unsupported_ops_warnings(False)
                total_flops += (flops.total() / 1e9)
            
            return total_flops, dict()
 
        for seq_len, width_mult in tqdm(list(zip(seq_lens, width_mults))):
            flops_sample = 0
            adaptive_layer_idx = args.adaptive_layer_idx
            original_input = torch.randn(1, seq_len, input_dim)
            flops = FlopCountAnalysis(block_original, original_input)
            flops = flops.unsupported_ops_warnings(False)
            # print(flop_count_table(flops))
            # total_flops += (flops.total() / 1e9) * adaptive_layer_idx
            flops_sample += (flops.total() / 1e9) * adaptive_layer_idx

            flops = FlopCountAnalysis(router, original_input.mean(dim=1))
            flops = flops.unsupported_ops_warnings(False)
            # print(flop_count_table(flops)) 
            # total_flops += (flops.total() / 1e9)
            flops_sample += (flops.total() / 1e9)

            input_ = torch.randn(1, seq_len, int(input_dim * width_mult))
            flops = FlopCountAnalysis(blocks_adaptive[width_mult], input_)
            # print(flop_count_table(flops))
            flops = flops.unsupported_ops_warnings(False)
            # total_flops += (flops.total() / 1e9) * (n_layers - adaptive_layer_idx)
            flops_sample += (flops.total() / 1e9) * (n_layers - adaptive_layer_idx)

            # total_flops = original_flops + router_flops + adaptive_flops
            # avg_flops = total_flops / sum(n_vals)
            total_flops_per_width_mult[width_mult] += flops_sample
            total_flops += flops_sample
 
        return total_flops, total_flops_per_width_mult
    else:
        if args.adaptive_layer_idx is not None:
            # adaptive no router flops
            logger.info("Measuring the flops of adaptive + no router")
            input_dim = 768 # this is original input dim without any adaptive width
            seq_len = 128
            n_layers = 12
            adaptive_layer_idx = args.adaptive_layer_idx
            assert len(args.width_mult_list) == 1, "only one width mult is supported when using adaptive no router"
            width_mult = args.width_mult_list[0]
            mlp_ratio = 4
            nhead = 12 

            block_original = CustomTransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=mlp_ratio * input_dim,
                dim_head=64,
                dropout=0.1, 
                activation='gelu',
                batch_first=True,
                device='cpu'
            )

            # print(flop_count_table(flops)) 
            block_adaptive = CustomTransformerEncoderLayer(
                d_model=int(input_dim * width_mult),
                nhead=int(nhead * width_mult),
                dim_feedforward=int(mlp_ratio * width_mult * input_dim),
                dropout=0.1, 
                dim_head=64,
                activation='gelu',
                batch_first=True,
                device='cpu'
            )
            
            total_flops = 0
            for seq_len in tqdm(seq_lens):
                original_input = torch.randn(1, seq_len, input_dim)
                flops = FlopCountAnalysis(block_original, original_input)
                flops = flops.unsupported_ops_warnings(False)
                total_flops += (flops.total() / 1e9) * adaptive_layer_idx
                # print(flop_count_table(flops))
                input_ = torch.randn(1, seq_len, int(input_dim * width_mult))
                flops = FlopCountAnalysis(block_adaptive, input_)
                flops = flops.unsupported_ops_warnings(False)
                total_flops += (flops.total() / 1e9) * (n_layers - adaptive_layer_idx)
  
            return total_flops
        elif any([x in args.model_type for x in {"shallow_deep", "branchy", "pabee", "dee", "fast", "berxit"}]):
            assert depths is not None
            logger.info("Measuring the flops of shallow deep or pabee or branchy baselines")
            if "dynabert" in args.model_dir:
                logger.info("Using dynabert with adaptive depth methods...")
                input_dim = 768 # this is original input dim without any adaptive width
                seq_len = 128
                mlp_ratio = 4
                nheads = 12
                num_labels = 3
                hidden_size = 768

                # TODO: for now I am going to find the dynawidth mult from the name of the file but change this later
                dyna_width_mult = float(Path(args.model_dir).stem.split("_")[1])
                mapping = {25.: 0.25, 5.: 0.5, 75.: 0.75, 1: 1.0}
                dyna_width_mult = mapping[dyna_width_mult]
                assert dyna_width_mult in [0.25, 0.5, 0.75, 1.0]
    
                block_dyna = CustomTransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=int(nheads * dyna_width_mult),
                    dim_head=64,
                    dim_feedforward=int(mlp_ratio * dyna_width_mult * input_dim),
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True,
                    device='cpu'
                )
                internal_classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, num_labels)
                )
                flops_per_depth = {}
                for depth in np.unique(depths):
                    flops_per_depth[depth] = 0

                total_flops = 0

                for seq_len, depth in tqdm(list(zip(seq_lens, depths))):
                    flops_sample = 0
                    original_input = torch.randn(1, seq_len, input_dim)
                    flops = FlopCountAnalysis(block_dyna, original_input)
                    flops = flops.unsupported_ops_warnings(False)
                    flops_sample += (flops.total() / 1e9) * depth

                    flops = FlopCountAnalysis(internal_classifier, original_input.mean(dim=1))
                    flops = flops.unsupported_ops_warnings(False)
                    if args.internal_classifier_all_layers:
                        flops_sample += (flops.total() / 1e9) * args.internal_classifier_layer
                    else:
                        flops_sample += (flops.total() / 1e9)
                    total_flops += flops_sample
                    flops_per_depth[depth] += flops_sample
                    
                return total_flops, flops_per_depth

            input_dim = 768 # this is original input dim without any adaptive width
            seq_len = 128
            n_layers = 6 if "distilbert" in args.model_type else 12
            adaptive_layer_idx = args.adaptive_layer_idx
            assert len(args.width_mult_list) == 1, "only one width mult is supported when using adaptive no router"
            width_mult = args.width_mult_list[0]
            mlp_ratio = 4
            nhead = 12 
            num_labels = 3
            hidden_size = 768

            block_original = CustomTransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=mlp_ratio * input_dim,
                dim_head=64,
                dropout=0.1, 
                activation='gelu',
                batch_first=True,
                device='cpu'
            )
            internal_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_labels)
            )
            flops_per_depth = {}
            for depth in np.unique(depths):
                flops_per_depth[depth] = 0

            total_flops = 0

            for seq_len, depth in tqdm(list(zip(seq_lens, depths))):
                flops_sample = 0
                original_input = torch.randn(1, seq_len, input_dim)
                flops = FlopCountAnalysis(block_original, original_input)
                flops = flops.unsupported_ops_warnings(False)
                flops_sample += (flops.total() / 1e9) * depth

                flops = FlopCountAnalysis(internal_classifier, original_input.mean(dim=1))
                flops = flops.unsupported_ops_warnings(False)
                if args.internal_classifier_all_layers:
                    flops_sample += (flops.total() / 1e9) * args.internal_classifier_layer
                else:
                    flops_sample += (flops.total() / 1e9)
                total_flops += flops_sample
                flops_per_depth[depth] += flops_sample
            
            return total_flops, flops_per_depth


if __name__ == "__main__":
    # flops = bench_flops_dyna(width_mult=1.0)
    # avg_flops = bench_flops_adaptive_with_router(
    #     width_mults=[0.25, 1.0],
    #     hidden_dim_router=1024,
    #     # n_vals=[6307, 3508],  # MNLI
    #     # n_vals=[572, 471],    # CoLA
    #     # n_vals = [6490, 3325],
    #     # n_vals = [23612, 16818],
    #     # n_vals = [28596, 11834],
    #     # n_vals = [29405, 11025],
    #     # n_vals = [5689, 4126],
    #     # n_vals = [5420, 4395],
    #     # n_vals = [5986, 3829],
    #     # n_vals = [3302, 2161],
    #     # n_vals = [3263, 2200],
    #     # n_vals = [5893, 3922],
    #     # n_vals = [3390, 2073],
    #     # n_vals = [3603, 1860],
    #     # n_vals = [701, 171],
    #     # n_vals = [3660, 1803],
    #     # n_vals = [517, 355],
    #     # n_vals = [613, 259],
    #     # n_vals = [25392, 15038],
    #     # n_vals = [617, 255],
    #     # n_vals = [27301, 13129],
    #     # n_vals = [27759, 12671],
    #     # n_vals = [30163, 10267],
    #     # n_vals = [3848, 1615],
    #     # n_vals = [6074, 3741],
    #     # n_vals = [723, 149],
    #     n_vals=[4675, 5140],
    #     adaptive_layer_idx=2,
    # )
    # avg_flops = bench_flops_adaptive_no_router(
        # width_mult=0.25,
        # adaptive_layer_idx=9,
    # )

    # FLOPS DYNABERT + ADAPTIVE WIDTH WITH ROUTER
    # avg_flops = bench_flops_dyna_adaptive_with_router(
    #    dyna_width_mult=0.5,
    #    adaptive_width_mults=[2/3, 1.0],
    #    hidden_dim_router=1024,
    #    n_vals=[5937, 3878],
    #    adaptive_layer_idx=9,
    # )

    avg_flops = bench_flops_adaptive_no_router(
       width_mult=0.25,
       adaptive_layer_idx=1,
    )
    print("avg flops", avg_flops)
    # bench_flops_original()
    # flops = FlopCountAnalysis(layer, torch.randn(1, 768))
    # print(flop_count_table(flops))
    
    # flops = bench_flops_original()
    # print(flops / 1e9)