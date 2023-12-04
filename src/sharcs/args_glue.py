import argparse
from sharcs.custom_transformers import glue_processors as processors
from sharcs.custom_transformers import MODEL_CLASSES


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the GLUE task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--output_dir', default='/home/user/adaptive-width/output')
    parser.add_argument("--per_gpu_train_batch_size", default=192, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--name", default="debug", type=str)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training",)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--report_to_wandb", action="store_true", help="whether to report to wandb")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The student (and teacher) model dir.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--width_mult_list', type=str, default='1.',
                        help="the possible widths used for training, e.g., '1.' is for separate training "
                        "while '0.25,0.5,0.75,1.0' is for vanilla slimmable training")
    parser.add_argument('--adaptive_layer_idx', type=int, default=None, required=False)
    parser.add_argument("--save_pred_results", 
                        action="store_true", 
                        help="whether to save the correct or not for each sample in the train/val set."
                        "This can be used when you want to save the offline labels.")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--verbose", type=bool, default=False, help="verbosity")
    parser.add_argument("--hidden_size_router", type=int, default=1024, help="the hidden size of the router network")
    parser.add_argument("--do_train", action="store_true", help="whether to train the router")
    parser.add_argument("--use_router", action="store_true", help="whether to use a network that has mulitple subwidths with a router"
                        "(either val, train, or student for distillation)")
    parser.add_argument("--train_router_window_size", type=int, default=0, help="the window size for training history for online buckets.")
    parser.add_argument("--lambda_loss_task", type=float, default=1.0, help="the weight of the task loss")  # just for training router
    parser.add_argument("--lambda_loss_router", type=float, default=1.0, help="the weight of the router loss")  # just for training router
    parser.add_argument("--write_val_preds", action="store_true", help="whether to write the validation predictions to a file")
    parser.add_argument("--repeat_smaller_sized_buckets", action="store_true", help="whether to repeat the samples for the sub-networks"
                        "with less training samples so that they have the same number of samples as the sub-network with the most samples.")
    parser.add_argument("--subnetwork_loss_weights", type=str, default=None, help="weights for the loss of different sub-networks.")
    parser.add_argument(
        "--widths_config_file", 
        type=str, 
        default="/home/user/adaptive-width/configs/main.yaml", 
        help="the path to the bucket config file")
    parser.add_argument("--measure_latency", action="store_true", help="whether to measure the latency of the best model on the validation set")
    parser.add_argument("--measure_flops", action="store_true", help="whether to measure the total flops of validation")
    parser.add_argument("--enable_per_iteration_bucket_loss_weighting", action="store_true", help="whether to enable per iteration bucket loss weighting")
    parser.add_argument("--weighted_dim_reduction", action="store_true", help="whether to use weighted dimension reduction")
    parser.add_argument("--init_private_layernorms_from_scratch", action="store_true", help="whether to initialize the private layernorms from scratch."
                        "If it's false, will use the first dims of already existing layernorms.")
    parser.add_argument("--custom_dropout_rate", type=float, default=None, help="custom dropout used for the transformer network")
    parser.add_argument("--compute_neuron_head_importance", action="store_true", help="whether to compute the neuron head importance and reorder"
                        "heads before training SHARCS.")
    parser.add_argument("--remove_pads_in_eval", action="store_true", help="whether to remove the pads in the eval")
    parser.add_argument("--save_epoch_checkpoints", action="store_true", help="whether to save the checkpoints")
    parser.add_argument("--save_last_epoch_ckpt", action="store_true", help="whether to save the last epoch checkpoint")
    parser.add_argument("--save_best_checkpoint", action="store_true", help="whether to save the best checkpoint")
    
    # baseline arguments
    parser.add_argument("--lambda_loss_internal", type=float, default=1.0, help="the weight of the internal loss") 
    parser.add_argument("--internal_classifier_thresh", type=float, default=0.5, help="the threshold for the internal classifier")
    parser.add_argument("--internal_classifier_layer", type=int, default=1, help="the layer for the internal classifier")
    parser.add_argument("--internal_classifier_all_layers", action="store_true", help="whether to attach an internal classifier to all layers")
    
    # ablation baseline arguments
    parser.add_argument("--use_entropy_hardness", action="store_true", help="whether to use entropy to measure hardness of samples"
                        "(instead of score of predicted probability of ground truth class)")

    # pabee baseline arguments
    parser.add_argument("--patience", type=int, default=6, help="the patience for early exiting")
  
    return parser.parse_args()