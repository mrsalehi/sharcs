from pathlib import Path
import math
import json
import shutil
import yaml
from loguru import logger
import os
import numpy as np
from collections import defaultdict
import torch
from collections import Counter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from torch.nn import  MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
import wandb
from utils import remove_pads, compute_neuron_head_importance, reorder_neuron_head
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from sharcs.sampler import SingleWidthBatchSampler
from sharcs.bench import bench_flops_total
from sharcs.custom_transformers import MODEL_CLASSES
from sharcs.custom_transformers import (
    AdamW, WarmupLinearSchedule,
    glue_compute_metrics as compute_metrics,
    glue_output_modes as output_modes,
    glue_processors as processors
)
from sharcs.args_glue import get_args
from sharcs.data import collate_fn
from sharcs.utils import (
    world_info_from_env, init_distributed_device, is_master, 
    save_pred_results_train, save_and_update_train_history, 
    load_and_cache_examples, update_config, set_width_mult,
    check_conditions, random_seed, get_eval_data, 
    gather_tensor, write_num_samples_per_width, measure_mnli_latency, 
    soft_cross_entropy, measure_qqp_latency,
)


TASK_METRICS = {
    "mnli": "acc",
    "cola": "mcc",
    "sst-2": "acc",
    "mrpc": "acc",
    "sts-b": "pearson",
    "qqp": "acc",
    "mnli-mm": "acc",
    "qnli": "acc",
    "rte": "acc",
    "wnli": "acc",
}

loss_mse = MSELoss()

# logger = logging.getLogger(__name__)
# logging.basicConfig(stream=sys.stdout, level=logger.info)

 
def evaluate(args, model, tokenizer, prefix=""):
    """ Evaluate the model """

    if args.measure_latency:
        assert args.task_name == "qqp"
        measure_qqp_latency(args, model, tokenizer)

    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.result_dir,)

    model.apply(lambda m: setattr(m, 'mode', 'eval'))

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args,
            eval_task,
            tokenizer,
            evaluate=True,
        )
        
        if is_master(args):
            if not os.path.exists(eval_output_dir):
                os.makedirs(eval_output_dir)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.per_gpu_eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        all_corrects = {}
        all_corrects_per_width = defaultdict(list)


        eval_data = get_eval_data(args, eval_task)  # for viz purposes
        pred_router_widths = []  # the width that the router has predicted, for viz purposes

        if args.measure_flops:
            seq_lens = []
            if any([x in args.model_type for x in ["shallow_deep", "pabee", "branchy", "dee", "fast", "berxit"]]):
                depths = []
            tokens_remained = None
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            if args.remove_pads_in_eval:
                batch, input_length = remove_pads(batch, args)
                seq_lens.append(input_length)

            batch = tuple(t.to(args.device) for t in batch)
             
            with torch.no_grad():
                unique_ids = batch[0]
                inputs = {'input_ids': batch[1], 'attention_mask': batch[2], 'labels': batch[4],\
                    'token_type_ids': batch[3] if args.model_type in ["bert", "deebert", "fastbert",
                                                                      "sharcstranskimer", "berxitbert",
                                                                      "shallow_deep_bert", "pabee_bert",
                                                                      "albert", "deedynabert",
                                                                      "berxitdynabert", "fastdynabert",
                                                                      "pabee_dynabert", "branchy_dynabert",
                                                                      "shallow_deep_dynabert"] else None}

                outputs = model(**inputs)

                tmp_eval_loss, logits = outputs['loss'], outputs['logits']

                corrects = None
                if args.use_router:
                    corrects = outputs['corrects']
                    predicted_width_mult = outputs['router_predicted_width_mult']
                    pred_router_widths.append(predicted_width_mult)
                    all_corrects_per_width[predicted_width_mult].append(corrects[0].item())
                else:
                    if outputs.get('corrects', None) is not None:
                        corrects = outputs['corrects']
                        all_corrects_per_width[args.width_mult_list[0]].extend(corrects.detach().cpu().numpy().tolist())

                if corrects is not None:
                    all_corrects.update({uid.item(): correct.item() for uid, correct in zip(unique_ids, corrects)})
                eval_loss += tmp_eval_loss.mean().item()

                if args.measure_flops and any([x in args.model_type for x in {"shallow_deep", "pabee", "branchy", "dee", "fast", "berxit"}]):
                    if outputs['internal_prediction']:
                        if outputs.get("depth", None) is not None:
                            # using internal classifier at every layer
                            depths.append(outputs['depth'])
                        else:
                            depths.append(args.internal_classifier_layer + 1)
                    else:
                        depth_ = 6 if "distilbert" in args.model_type else 12
                        depths.append(depth_)
                
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        # if args.measure_flops and args.model_type in {"shallow_deep_roberta", "shallow_deep_bert", "pabee_roberta", "pabee_bert", "branchy_roberta", "branchy_bert"}: 
        if args.measure_flops and any([x in args.model_type for x in {"shallow_deep", "pabee", "branchy", "dee", "fast", "berxit"}]):
            logger.info("Samples solved by shallower models: {}".format(np.sum(np.array(depths) != 12)))
            logger.info("Samples solved by deep model: {}".format(np.sum(np.array(depths) == 12)))

        
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        if eval_task == 'mnli-mm':
            results.update({'acc_mm':result['acc']})
        else: 
            for width_ in args.width_mult_list:
                corrects_width = all_corrects_per_width[width_]
                result['corrects_per_width_{}'.format(width_)] = f"{int(np.sum(corrects_width))}/{len(corrects_width)}"
                if len(corrects_width) > 0:
                    result['acc_{}'.format(width_)] = np.mean(corrects_width)
            results.update(result)

        output_eval_file = os.path.join(eval_output_dir, f"eval_results.txt") # wirte all the results to the same file

        if args.measure_flops:
            if args.use_router:
                total_flops, total_flops_per_width_mult = bench_flops_total(
                    seq_lens,
                    width_mults=pred_router_widths,
                    args=args,
                    tokens_remained=tokens_remained,
                )
                for width_ in total_flops_per_width_mult.keys():
                    result['total_flops_{}'.format(width_)] = total_flops_per_width_mult[width_]
                result['total_flops'] = total_flops  # gigaflops
                results.update(result)
            else:
                if any([x in args.model_type for x in {"shallow_deep", "pabee", "branchy", "dee", "fast", "berxit"}]):
                    total_flops, flops_per_depth = bench_flops_total(
                        seq_lens, 
                        width_mults=None, 
                        args=args, 
                        depths=depths
                    )
                    for depth_ in flops_per_depth.keys():
                        result['total_flops_{}'.format(depth_)] = flops_per_depth[depth_]
                    result['total_flops'] = total_flops  # gigaflops
                    results.update(result)
                    depth_to_n_samples = Counter(depths)
                    for depth_ in depth_to_n_samples.keys():
                        logger.info(f"Depth {depth_}: {depth_to_n_samples[depth_]} samples")
                else:
                    if args.cur_epoch == 0:
                        total_flops = bench_flops_total(seq_lens, width_mults=None, args=args)
                        result['total_flops'] = total_flops  # gigaflops
                        results.update(result)

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {f} *****")
            for key in sorted(result.keys()):
                logger.info(f"  {key} = {str(result[key])}")
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write("\n")
        
        if args.use_router and args.write_val_preds:
            assert eval_task == "mnli"
            pred2label = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
            with open(os.path.join(eval_output_dir, f'val_preds_{args.current_epoch}.txt'), 'w') as f:
                f.write("text_a\ntext_b\nlabel\npred\npred_router_width")
                f.write("\n")
                for uid, pred in enumerate(preds):
                    f.write(f"{eval_data[uid].text_a}\t{eval_data[uid].text_b}\t{eval_data[uid].label}\t{pred2label[pred]}\t{pred_router_widths[uid]}")
                    f.write("\n")
 
        return results, all_corrects


def train(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    check_conditions(args)
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device = init_distributed_device(args)

    # make the random seed same for all processes
    random_seed(args.seed, 0)

    args.output_dir = os.path.join(args.output_dir, args.name)
    args.result_dir = os.path.join(args.output_dir, 'result')
    args.checkpoint_path = os.path.join(args.output_dir, 'checkpoint')

    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_dir, num_labels=num_labels, finetuning_task=args.task_name)
    if args.custom_dropout_rate is not None:
        config.attention_probs_dropout_prob = args.custom_dropout_rate
        config.hidden_dropout_prob = args.custom_dropout_rate
        
    update_config(config, args)  # update the config with the new args

    config.output_attentions, config.output_hidden_states, config.output_intermediate = False, False, False

    tokenizer = tokenizer_class.from_pretrained(args.model_dir, do_lower_case=args.do_lower_case)

    if is_master(args):
        if args.name == "debug":
            shutil.rmtree(args.output_dir, ignore_errors=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
        wandbid_file = Path(args.output_dir) / "wandb-id.txt"

    # if args.model_type in {"shallow_deep_roberta", "shallow_deep_bert", "shallow_deep_distilbert"}:
    if any([x in args.model_type for x in {"shallow_deep", "branchy", "dee", "fast", "berxit"}]):
        config.internal_classifier_all_layers = args.internal_classifier_all_layers
        config.internal_classifier_layer = args.internal_classifier_layer
        config.internal_classifier_thresh = args.internal_classifier_thresh
        config.output_attentions, config.output_hidden_states, config.output_intermediate = False, True, False
     
    if "pabee" in args.model_type:
        config.internal_classifier_all_layers = args.internal_classifier_all_layers
        config.internal_classifier_layer = args.internal_classifier_layer
        config.output_attentions, config.output_hidden_states, config.output_intermediate = False, True, False
        config.patience = args.patience
    
    model = model_class.from_pretrained(args.model_dir, config=config)

    if config.adaptive_layer_idx is not None:
        if any([width_ for width_ in args.width_mult_list if width_ != 1.0]):  # only do it if there are widths that are not 1.0
            if args.model_type == "distilbert":
                encoder = getattr(model, args.model_type)
            elif args.model_type in {"tinybert"}:
                encoder = getattr(model, "bert").encoder
            else:
                encoder = getattr(model, args.model_type).encoder

            encoder.make_layer_norms_private(
                config,
                init_private_layernorms_from_scratch=args.init_private_layernorms_from_scratch)
     
    if isinstance(device, str):
        device = torch.device(str)
    model.to(device)
        
    random_seed(args.seed, args.rank)
 
    if is_master(args):
        args.wandb_id = wandbid_file.read_text().strip() if wandbid_file.exists() else wandb.util.generate_id()
        wandb_run = (
            wandb.init(
                project="bert-adaptive-width",
                name=args.name,
                id=args.wandb_id,
                config=args,
                settings=wandb.Settings(code_dir="/home/user/ALBEF-adaptive/src"),
                ) if args.report_to_wandb else None
        )

    def wandb_log(values, step=None):
        if is_master(args) and wandb_run is not None:
            wandb_run.log(values, step=step)

    if is_master(args):
        logger.info(f"{args}\n")

    if args.distributed:
        logger.info(
            f'Running in distributed mode with multiple processes. Device: {device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logger.info(f'Running with a single process. Device {args.device}.')
        
    if args.compute_neuron_head_importance: 
        head_importance, neuron_importance = compute_neuron_head_importance(args, model, tokenizer)
        reorder_neuron_head(args, model, head_importance, neuron_importance)
        logger.info(f"Reordering neurons and heads is done for task {args.task_name}. Exiting...")
        exit(0)

    if args.do_train:
        train_dataset = load_and_cache_examples(
            args,
            args.task_name,
            tokenizer,
            evaluate=False
        )

        if args.use_router:
            # which means we are training the network with router
            # Each sampler on a device will run with a different network width.
            train_sampler = SingleWidthBatchSampler(
                dataset=train_dataset,
                args=args,
                batch_size=args.per_gpu_train_batch_size,
                num_buckets=len(config.width_mult_list),
                rank=args.rank,
                shuffle=True,
                num_replicas=args.world_size
            )
            logger.info("Using SingleWidthBatchSampler for training the router with online bucketing.")
            train_dataloader = DataLoader(
                train_dataset, 
                shuffle=False,
                batch_sampler=train_sampler, 
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False)
        else:
            train_sampler = RandomSampler(train_dataset) if not args.distributed else \
                DistributedSampler(
                    train_dataset, 
                    num_replicas=args.world_size, 
                    drop_last=False)
            train_dataloader = DataLoader(
                train_dataset, 
                sampler=train_sampler, 
                batch_size=args.per_gpu_train_batch_size, 
                collate_fn=None)
         
        
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
        logger.info(f"Number of train samples: {len(train_dataset)}")

        # NOTE: if using batch sampler, correct t_total will be computed after calling set_bucket_and_batches of the sampler.
        # using t_total here does not consider the number of batches.
        logger.info(f"t_total in the beginning before any augmentation: {t_total}")

    if is_master(args):
        if args.name == "debug":
            shutil.rmtree(args.output_dir, ignore_errors=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.result_dir).mkdir(parents=True, exist_ok=True) 
        if args.checkpoint_path:
            Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
        wandbid_file = Path(args.output_dir) / "wandb-id.txt"

    args.idx_to_width_mult = {idx: width_mult for idx, width_mult in enumerate(config.width_mult_list)}

    if not args.use_router:  # in these two cases width_mult is not static and will be set in the train or val loop
        if args.adaptive_layer_idx is not None: 
            assert len(args.width_mult_list) == 1
            set_width_mult(model, config, width_mult=args.width_mult_list[0])
        else:
            if not any([x in args.model_type for x in {"shallow_deep", "pabee", "branchy", "dee", "fast", "berxit"}]):
                set_width_mult(model, config, width_mult=args.width_mult_list[0])

    if args.model_type == 'roberta':
        args.warmup_steps = int(t_total*0.06)

    if args.distributed:
        # find_unused_params is true, as there are width-specific layer norms, some of the parameters might not be used in forward passes.
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        dist.barrier()

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm_025', 'LayerNorm_05', 'LayerNorm_075', 'LayerNorm_10']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'name': 'weight_decay'},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'name': 'no_weight_decay'
            }]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
   
    start_epoch = 0 
    global_step = 0
    epochs = args.epochs 
    top_val_metric = 0.0
    top_val_metric_with_router = 0.0
    best_epoch_with_router = -1
    
    if args.checkpoint_path:
        args.resume = list(Path(args.checkpoint_path).glob("checkpoint_ep*"))
        if args.resume:
            checkpoint = torch.load(args.resume[0], map_location={'cuda:%d' % 0: 'cuda:%d' % args.local_rank})
            sd = checkpoint["state_dict"]
            start_epoch = checkpoint["epoch"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=True)
            global_step = checkpoint["global_step"]
            logger.info(f"=> Resuming from checkpoint {args.resume} (global_step {global_step})")
            results = checkpoint["results"]
            if results is not None:
                top_val_metric = results["top_val_metric"]
                top_val_metric_with_router = results["top_val_metric_with_router"] 
            if args.use_router:
                best_epoch_with_router = checkpoint["best_epoch_with_router"]
        else:
            logger.info(f"=> No checkpoint found at {args.checkpoint_path}. Training from scratch.")
      
    if args.distributed:
        dist.barrier()
     
    if args.name == "debug":
        results, corrects = evaluate(args, model, tokenizer)

    if args.do_train:
        e_bar = tqdm(range(start_epoch, epochs), disable=not is_master(args))
        train_history = defaultdict(dict)  # all of the processes should have this dict
        if not args.use_entropy_hardness:
            confidence_scores_epoch = {}
        else:
            entropies_epoch = {}
                    
        for epoch in e_bar:
            if is_master(args):
                logger.info(f"============Epoch {epoch}============")
            model.train()
            args.current_epoch = epoch
             
            if isinstance(train_sampler, SingleWidthBatchSampler):
                train_sampler.set_buckets_and_batches(epoch, config)
                if epoch == 0:
                    pass
                    # because we are repeating the samples. For now the number of warmup steps
                    # is calculated based on the initial number of samples.
                    # t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
                    # logger.info(f"t_total in the beginning: {t_total}")
                    # if args.model_type == 'roberta':
                    #     args.warmup_steps = int(t_total*0.06)
                    # scheduler = WarmupLinearSchedule(
                    #     optimizer, 
                    #     warmup_steps=args.warmup_steps, 
                    #     t_total=len(train_dataloader))
                dist_buckets_log = {}
                for width_, num_samples in train_sampler.bucket_lengths_before_repeat_or_aug.items():
                    dist_buckets_log[f"num train samples bucket {width_}"] = num_samples
                wandb_log(dist_buckets_log, step=global_step)
                
            p_bar = tqdm(train_dataloader, disable=not is_master(args))
            all_corrects = {}

            if args.use_router:
                model.apply(lambda m: setattr(m, 'mode', 'train'))
            
            num_samples_per_width = {width_: [] for width_ in args.width_mult_list} 
              
            if args.model_type in {"distilbert", "albert"}:
                grad_acc_step = 0

            for step, batch in enumerate(p_bar):
                args.cur_epoch = epoch
                if args.model_type in {"distilbert", "albert"}:
                    if grad_acc_step == 0:
                        optimizer.zero_grad()
                else:
                    optimizer.zero_grad()

                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                unique_ids = batch[0]
                inputs = {'input_ids': batch[1], 'attention_mask': batch[2], 'labels': batch[4],\
                    'token_type_ids': batch[3] if args.model_type in ["bert", "deebert", "fastbert",
                                                                      "sharcstranskimer", "berxitbert",
                                                                      "shallow_deep_bert", "pabee_bert",
                                                                      "albert", "deedynabert",
                                                                      "berxitdynabert", "fastdynabert",
                                                                      "pabee_dynabert", "branchy_dynabert",
                                                                      "shallow_deep_dynabert"] else None}
                 
                if any([x in args.model_type for x in {"shallow_deep", "pabee", "branchy"}]):
                    # baseline
                    outputs = model(**inputs)
                    loss, loss_internal = outputs["loss"], outputs["loss_internal"]
                    losses_logs = {"loss": loss.item(), "loss_internal": loss_internal.item()}
                    loss = args.lambda_loss_task * loss + args.lambda_loss_internal * loss_internal
                    loss.backward()
                elif "dee" in args.model_type:
                    outputs = model(**inputs)
                    loss, loss_internal = outputs["loss"], outputs["loss_internal"]
                    losses_logs = {"loss": loss.item(), "loss_internal": loss_internal.item()}
                    loss = loss_internal
                    loss.backward()
                elif "berxit" in args.model_type:
                    outputs = model(**inputs)
                    if step % 2 == 1:
                        loss = outputs["loss"]
                    else:
                        loss = outputs["loss_internal"] + outputs["loss"]
                    losses_logs = {"loss": loss.item()}
                    loss.backward()                    
                elif "fast" in args.model_type:
                    outputs = model(**inputs)
                    loss = outputs["loss"]
                    losses_logs = {"loss": loss.item()}
                    loss.backward()
                elif args.use_router:
                    width_mult_label, batch_width, per_iteration_bucket_loss_weighting = batch[-3:]
                    assert len(torch.unique(batch_width)) == 1
                    per_iteration_bucket_loss_weighting = per_iteration_bucket_loss_weighting[0].item()
                    this_batch_width = args.idx_to_width_mult[batch_width[0].item()]
                    set_width_mult(model, config, width_mult=this_batch_width)
                    inputs.update({'width_mult_label': width_mult_label, 'batch_width': this_batch_width})
                    outputs = model(**inputs)
                    loss_task = outputs["loss"]
                    loss_router = outputs["loss_router"]
                    loss = args.lambda_loss_task * loss_task + loss_router * args.lambda_loss_router
                    loss = loss * per_iteration_bucket_loss_weighting
                    if args.subnetwork_loss_weights is not None:
                        loss = loss * args.subnetwork_loss_weights[this_batch_width]

                    corrects = outputs["corrects"]
                    if not args.use_entropy_hardness:
                        confidence_scores_ground_truth = outputs["probs_ground_truth_class"]
                    else:
                        entropies = outputs["entropy"]
                            
                    losses_logs = {'loss': loss.item(), 'loss_task': loss_task.item(), 'loss_router': loss_router.item()}
                    loss.backward()

                    if args.model_type in {"distilbert", "albert"}:
                        grad_acc_step += 1

                    # tracking the number of samples in each batch
                    for width_ in args.width_mult_list: 
                        num_samples_per_width[width_].append(len(width_mult_label) if width_ == this_batch_width else 0)
                else:
                    # running dynaBERT or single individual network
                    outputs = model(**inputs)
                    loss = outputs["loss"]
                    corrects = outputs["corrects"]
                    losses_logs = {"loss": loss.item()}
                    loss.backward()

                if args.model_type in {"distilbert", "albert"}:
                    if grad_acc_step == args.gradient_accumulation_steps:
                        optimizer.step()
                        scheduler.step()
                        grad_acc_step = 0     
                else:
                    optimizer.step()

                 
                if args.model_type != "distilbert" and args.model_type != "albert": 
                    scheduler.step()

                global_step += 1
                 
                if is_master(args):
                    wandb_log(losses_logs, step=global_step)
                    lr_logs = {}
                    for g in optimizer.param_groups:
                        lr_logs[f"lr/{g['name']}"] = g["lr"] 
                    wandb_log(lr_logs, step=global_step)

                if args.save_pred_results:
                    all_corrects.update({uid.item(): corr.item() for uid, corr in zip(unique_ids, corrects)})

                if args.use_router:
                    if not args.use_entropy_hardness:
                        confidence_scores_epoch.update({uid.item(): confidence.item() for uid, confidence in zip(unique_ids, confidence_scores_ground_truth)})
                    else:
                        entropies_epoch.update({uid.item(): entropy.item() for uid, entropy in zip(unique_ids, entropies)})
                
            if args.use_router:
                write_num_samples_per_width(args, num_samples_per_width, epoch, args.result_dir)

            if args.save_pred_results:
                save_pred_results_train(args, all_corrects, args.result_dir, f'train_pred_result_ep{epoch}')
            
            if is_master(args):
                acc = []
                if args.task_name == "mnli":   # for both MNLI-m and MNLI-mm
                    acc_both = []
                
                results, corrects = evaluate(args, model, tokenizer) 
                
                if results[TASK_METRICS[args.task_name]] > top_val_metric:
                    top_val_metric = results[TASK_METRICS[args.task_name]]
                    if args.save_pred_results: 
                        with open(os.path.join(args.result_dir, f"best_val_pred_results.json"), 'w') as f:
                            json.dump(corrects, f)

                    # save the checkpoint
                    if args.save_best_checkpoint:
                        logger.info(f"saving the best model with val acc {top_val_metric} from epoch {epoch}...") 
                        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"best_val_model.pt"))
                
                results['top_val_metric'] = top_val_metric  # just for the sake of logging to wandb

                if args.use_router:
                    # if epoch >= args.online_buckets_window_size and results[TASK_METRICS[args.task_name]] > top_val_metric_with_router:
                    if epoch >= args.train_router_window_size and results[TASK_METRICS[args.task_name]] > top_val_metric_with_router:
                        patience = torch.tensor(0, device=device).long()
                        logger.info("Patience set to 0.")
                        top_val_metric_with_router = results[TASK_METRICS[args.task_name]]
                        best_epoch_with_router = epoch
                        args.best_epoch_with_router = best_epoch_with_router
                        # if args.measure_latency:
                            # assert args.task_name == "qqp", ""
                            # measure_mnli_latency(args, model, device, tokenizer)

                        if args.save_pred_results:
                            with open(os.path.join(args.result_dir, f"best_val_pred_results_with_router.json"), 'w') as f:
                                json.dump(corrects, f)
                        if args.save_best_checkpoint: 
                            logger.info(f"Saving the best model+router with val acc {top_val_metric_with_router} from epoch {epoch}...")
                            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"best_val_model_with_router.pt"))
 
                    results['top_val_metric_with_router'] = top_val_metric_with_router  # just for the sake of logging to wandb
 
                # save the checkpoint after each epoch
                checkpoint_dict = {
                    "epoch": epoch,
                    "name": args.name,
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "results": results,
                }
                if args.use_router:
                    checkpoint_dict.update({"best_epoch_with_router": best_epoch_with_router})

                if args.save_epoch_checkpoints: 
                    torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"checkpoint_ep{epoch}.pt"))
                
                if os.path.exists(os.path.join(args.checkpoint_path, f"checkpoint_ep{epoch-1}.pt")):
                    os.remove(os.path.join(args.checkpoint_path, f"checkpoint_ep{epoch-1}.pt"))

                with open(wandbid_file, 'w') as f:
                    f.write(args.wandb_id)

                logger.info(f"results: {results} ")
                wandb_results_log = {k: v for k, v in results.items() if "acc_" not in k and "corrects_per_" not in k}
                wandb_log(wandb_results_log, step=global_step)
        
                acc.append(list(results.values())[0])
                if args.task_name == "mnli":
                    acc_both.append(list(results.values())[0:2])

            if args.use_router:
                # save confidence scores to files. Just the main process has the train history and for others is just an empty dict
                if not args.use_entropy_hardness: 
                    train_history = save_and_update_train_history(args, confidence_scores_epoch, args.result_dir, \
                        f'train_confidence_scores_ep{epoch}', epoch, train_history, best_epoch_with_router)
                else:
                    train_history = save_and_update_train_history(args, entropies_epoch, args.result_dir, \
                        f'train_confidence_scores_ep{epoch}', epoch, train_history, best_epoch_with_router)
             
            if args.distributed:
                dist.barrier()
            
        if is_master(args):
            acc = []
            if args.task_name == "mnli":
                acc_both = []
            
            results, corrects = evaluate(args, model, tokenizer)

            if is_master(args):
                logger.info(f"width_mult_list: {args.width_mult_list} ")
                logger.info(f"results: {results} ")
                wandb_log(results, step=global_step)
        
                acc.append(list(results.values())[0])
                if args.task_name == "mnli":
                    acc_both.append(list(results.values())[0:2])
    else:
        results, corrects = evaluate(args, model, tokenizer)
        if is_master(args):
            logger.info(f"width_mult_list: {args.width_mult_list} ")
            logger.info(f"results: {results} ")
        
        if args.save_pred_results:
            with open(os.path.join(args.result_dir, f"best_val_pred_results.json"), 'w') as f:
                json.dump(corrects, f)
    
    if args.distributed:
        dist.barrier()
 
 
if __name__ == "__main__": 
    args = get_args() 
    args.width_mult_list = [float(eval(width)) for width in args.width_mult_list.split(',')]
    
    if args.subnetwork_loss_weights is not None:
        weights_ = [float(weight) for weight in args.subnetwork_loss_weights.split(',')]
        assert len(weights_) == len(args.width_mult_list), "length of subnetwork_loss_weights should be the same as width_mult_list"
        args.subnetwork_loss_weights = {}
        for width_mult, weight in zip(args.width_mult_list, weights_):
            args.subnetwork_loss_weights[width_mult] = weight

    if args.use_router:
        with open(args.widths_config_file, 'r') as f:
            bucket_config = yaml.load(f, Loader=yaml.FullLoader)
        
        threshs = bucket_config['buckets']['confidence_threshs']
        for key in threshs.copy().keys():
            if isinstance(key, str):
                threshs[float(eval(key))] = threshs.pop(key)
            else:
                threshs[float(key)] = threshs.pop(key)
        
        for width_ in args.width_mult_list:
            assert width_ in threshs.keys(), f"width {width_} not found in the bucket config file {args.widths_config_file}"
        args.confidence_threshs = threshs

    train(args)