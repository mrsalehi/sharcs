eval "$(/mmfs1/home/mrsalehi/miniconda3/bin/conda shell.bash hook)" # init conda
conda activate adaptive_width

export TRANSFORMERS_CACHE=/home/user/transformers_cache
export HUGGINGFACE_HUB_CACHE=/home/user/huggingface_hub_cache
export HF_DATASETS_CACHE=/home/user/huggingface_datasets_cache
export WANDB_CACHE_DIR=/home/user/wandb_cache

# if $WIDTH_MULT is empty set it to 1.
if [ -z "$WIDTH_MULT" ]
then
    WIDTH_MULT=1
fi

# if $TASK is empty spit out an error message and exit
if [ -z "$TASK" ]
then
    echo "TASK is not set"
    exit 1
fi

# if SAVE_PRED_RESULTS is set to True set it to --save_pred_results else set it to ""
if [ "$SAVE_PRED_RESULTS" = "True" ] ; then
    SAVE_PRED_RESULTS="--save_pred_results"
else
    SAVE_PRED_RESULTS=""
fi

# if EPOCHS is empty set it to 10
if [ -z "$EPOCHS" ]
then
    EPOCHS=10
fi

# if per_gpu_eval_batch_size in capital letters is empty set it to 128
if [ -z "$PER_GPU_EVAL_BATCH_SIZE" ]
then
    PER_GPU_EVAL_BATCH_SIZE=128
fi

# if per_gpu_train_batch_size in capital letters is empty set it to 128
if [ -z "$PER_GPU_TRAIN_BATCH_SIZE" ]
then
    PER_GPU_TRAIN_BATCH_SIZE=192
fi

# if use_router in capital letters is True set it to --use_router else set it to ""
if [ "$USE_ROUTER" = "True" ]
then
    USE_ROUTER="--use_router"
else
    USE_ROUTER=""
fi

# if lambda_loss_task in capital letters is empty set it to 1.
if [ -z "$LAMBDA_LOSS_TASK" ]
then
    LAMBDA_LOSS_TASK=1
fi

if [ -z "$LAMBDA_LOSS_ROUTER" ]
then
    LAMBDA_LOSS_ROUTER=1
fi

# if hidden_size_router in capital letters is empty set it to 1024
if [ -z "$HIDDEN_SIZE_ROUTER" ]
then
    HIDDEN_SIZE_ROUTER=1024
fi

# if write_val_preds in capital letters is True set it to --write_val_preds else set it to ""
if [ "$WRITE_VAL_PREDS" = "True" ]
then
    WRITE_VAL_PREDS="--write_val_preds"
else
    WRITE_VAL_PREDS=""
fi

# if learning_rate in capital letters is empty set it to 2e-5
if [ -z "$LEARNING_RATE" ]
then
    LEARNING_RATE=2e-5
fi

# if ADAPTIVE_LAYER_IDX is empty set it to " " else set it to --adaptive_layer_idx $ADAPTIVE_LAYER_IDX
if [ -z "$ADAPTIVE_LAYER_IDX" ]
then
    ADAPTIVE_LAYER_IDX=""
else
    ADAPTIVE_LAYER_IDX="--adaptive_layer_idx $ADAPTIVE_LAYER_IDX"
fi

# if REPEAT_SMALLER_SIZED_BUCKETS is True set it to --repeat_smaller_sized_buckets else set it to ""
if [ "$REPEAT_SMALLER_SIZED_BUCKETS" = "True" ]
then
    REPEAT_SMALLER_SIZED_BUCKETS="--repeat_smaller_sized_buckets"
else
    REPEAT_SMALLER_SIZED_BUCKETS=""
fi

if [ -z "$MODEL_DIR" ]
then
    MODEL_DIR=/home/user/adaptive-width/models/dynabert/${TASK}/best
fi

# if TRAIN_ROUTER_WINDOW_SIZE is empty set it to 100000
if [ -z "$TRAIN_ROUTER_WINDOW_SIZE" ]
then
    TRAIN_ROUTER_WINDOW_SIZE=100000
fi

#  if weighted_dim_reduction in capital letters is True set it to --weighted_dim_reduction else set it to ""
if [ "$WEIGHTED_DIM_REDUCTION" = "True" ]
then
    WEIGHTED_DIM_REDUCTION="--weighted_dim_reduction"
else
    WEIGHTED_DIM_REDUCTION=""
fi

if [ -z "$WIDTHS_CONFIG_FILE" ]
then
    WIDTHS_CONFIG_FILE=/home/user/adaptive-width/configs/main.yaml
fi

# if init_private_layernorms_from_scratch in capital letters is True set it to --init_private_layernorms_from_scratch else set it to ""
if [ "$INIT_PRIVATE_LAYERNORMS_FROM_SCRATCH" = "True" ]
then
    INIT_PRIVATE_LAYERNORMS_FROM_SCRATCH="--init_private_layernorms_from_scratch"
else
    INIT_PRIVATE_LAYERNORMS_FROM_SCRATCH=""
fi

# if model_type in capital letters is empty set it to "bert"
if [ -z "$MODEL_TYPE" ]
then
    MODEL_TYPE=bert
fi

if [ -z "$SUBNETWORK_LOSS_WEIGHTS" ]
then
    SUBNETWORK_LOSS_WEIGHTS=""
else
    SUBNETWORK_LOSS_WEIGHTS="--subnetwork_loss_weights $SUBNETWORK_LOSS_WEIGHTS"
fi

# if custom_dropout_rate in capital letters is not empty set it to "--custom_dropout_rate $CUSTOM_DROPOUT_RATE" else set it to ""
if [ -z "$CUSTOM_DROPOUT_RATE" ]
then
    CUSTOM_DROPOUT_RATE=""
else
    CUSTOM_DROPOUT_RATE="--custom_dropout_rate $CUSTOM_DROPOUT_RATE"
fi

# if measure_flops in capital letters is True set it to --measure_flops else set it to ""
if [ "$MEASURE_FLOPS" = "True" ]
then
    MEASURE_FLOPS="--measure_flops"
else
    MEASURE_FLOPS=""
fi

# if remove_pads_in_eval in capital letters is True set it to --remove_pads_in_eval else set it to ""
if [ "$REMOVE_PADS_IN_EVAL" = "True" ]
then
    REMOVE_PADS_IN_EVAL="--remove_pads_in_eval"
else
    REMOVE_PADS_IN_EVAL=""
fi

# if seed in capital letters is empty set it to 42
if [ -z "$SEED" ]
then
    SEED=42
fi

# if save_epoch_checkpoints in capital letters is True set it to --save_epoch_checkpoints else set it to ""
if [ "$SAVE_EPOCH_CHECKPOINTS" = "True" ]
then
    SAVE_EPOCH_CHECKPOINTS="--save_epoch_checkpoints"
else
    SAVE_EPOCH_CHECKPOINTS=""
fi

# if save_best_checkpoint in capital letters is True set it to --save_best_checkpoint else set it to ""
if [ "$SAVE_BEST_CHECKPOINT" = "True" ]
then
    SAVE_BEST_CHECKPOINT="--save_best_checkpoint"
else
    SAVE_BEST_CHECKPOINT=""
fi

# if internal_classifier_thresh in capital letters is empty set it to 0.5
if [ -z "$INTERNAL_CLASSIFIER_THRESH" ]
then
    INTERNAL_CLASSIFIER_THRESH=0.5
fi

# if lambda_loss_internal in capital letters is empty set it to 1.0
if [ -z "$LAMBDA_LOSS_INTERNAL" ]
then
    LAMBDA_LOSS_INTERNAL=1.0
fi

# if internal_classifier_layer in capital letters is empty set it to 1
if [ -z "$INTERNAL_CLASSIFIER_LAYER" ]
then
    INTERNAL_CLASSIFIER_LAYER=1
fi

# if dist_backend in capital letters is empty set it to "nccl"
if [ -z "$DIST_BACKEND" ]
then
    DIST_BACKEND=nccl
fi

# if distributed in capital letters is empty set it to "True"
if [ -z "$DISTRIBUTED" ]
then
    DISTRIBUTED=True
fi

# if gradient_accumulation_steps in capital letters is empty set it to 1
if [ -z "$GRADIENT_ACCUMULATION_STEPS" ]
then
    GRADIENT_ACCUMULATION_STEPS=1
fi

# if internal_classifier_all_layers in capital letters is True set it to --internal_classifier_all_layers else set it to ""
if [ "$INTERNAL_CLASSIFIER_ALL_LAYERS" = "True" ]
then
    INTERNAL_CLASSIFIER_ALL_LAYERS="--internal_classifier_all_layers"
else
    INTERNAL_CLASSIFIER_ALL_LAYERS=""
fi

# if patience in capital letters is empty set it to 0
if [ -z "$PATIENCE" ]
then
    PATIENCE=0
fi

TASK_LOWERCASE=$(echo $TASK | tr '[:upper:]' '[:lower:]')

if [ "$USE_ROUTER" = "--use_router" ]
then
    # apptainer exec -B $(pwd) --nv /home/user/adaptive-width cd /home/user/adaptive-width/src/sharcs && python run_glue.py \
    # cd /home/user/tmp/${SLURM_JOB_NAME}/src && python run_glue.py \
    cd /home/user/adaptive-width/src/sharcs && python run_glue.py \
        --model_type $MODEL_TYPE \
        --num_workers 0 \
        --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
        --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
        --name $SLURM_JOB_NAME \
        --seed $SEED \
        --epochs $EPOCHS \
        --distributed $DISTRIBUTED \
        --width_mult_list $WIDTH_MULT \
        --task_name $TASK_LOWERCASE \
        --data_dir /home/user/adaptive-width/glue/glue_data/$TASK \
        --model_dir $MODEL_DIR $ADAPTIVE_LAYER_IDX \
        --do_train $SUBNETWORK_LOSS_WEIGHTS \
        $REPEAT_SMALLER_SIZED_BUCKETS \
        $ONLINE_AUGMENTATION \
        --learning_rate $LEARNING_RATE \
        --output_dir /home/user/adaptive-width/output \
        $TRAIN_SLIMMABLE_NETWORK \
        --widths_config_file $WIDTHS_CONFIG_FILE $SAVE_BEST_CHECKPOINT $SAVE_EPOCH_CHECKPOINTS $WEIGHTED_DIM_REDUCTION \
        $SAVE_PRED_RESULTS \
        $USE_ROUTER \
        $REMAINDER_NETWORK_TRAINING \
        --hidden_size_router $HIDDEN_SIZE_ROUTER $CUSTOM_DROPOUT_RATE \
        --train_router_window_size $TRAIN_ROUTER_WINDOW_SIZE $MEASURE_FLOPS $REMOVE_PADS_IN_EVAL \
        --report_to_wandb \
        --internal_classifier_thresh $INTERNAL_CLASSIFIER_THRESH \
        --lambda_loss_internal $LAMBDA_LOSS_INTERNAL \
        --lambda_loss_task $LAMBDA_LOSS_TASK \
        --lambda_loss_router $LAMBDA_LOSS_ROUTER \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --dist-backend $DIST_BACKEND $INIT_PRIVATE_LAYERNORMS_FROM_SCRATCH \
        $WRITE_VAL_PREDS $DATA_AUG
else
    # cd /home/user/tmp/${SLURM_JOB_NAME}/src && python run_glue.py \
    # apptainer exec -B $(pwd) --nv /home/user/adaptive-width cd /home/user/adaptive-width/src/sharcs && python run_glue.py \
    cd /home/user/adaptive-width/src/sharcs && python run_glue.py \
        --model_type $MODEL_TYPE \
        --num_workers 0 \
        --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
        --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
        --name $SLURM_JOB_NAME \
        --epochs $EPOCHS \
        --distributed $DISTRIBUTED \
        --seed $SEED \
        --width_mult_list $WIDTH_MULT \
        --task_name $TASK_LOWERCASE \
        --learning_rate $LEARNING_RATE \
        $WEIGHTED_DIM_REDUCTION \
        $REMAINDER_NETWORK_TRAINING \
        --model_dir $MODEL_DIR $ADAPTIVE_LAYER_IDX $MEASURE_FLOPS $REMOVE_PADS_IN_EVAL \
        --data_dir /home/user/adaptive-width/glue/glue_data/$TASK \
        --do_train $INTERNAL_CLASSIFIER_ALL_LAYERS \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --internal_classifier_thresh $INTERNAL_CLASSIFIER_THRESH \
        --lambda_loss_internal $LAMBDA_LOSS_INTERNAL \
        --output_dir /home/user/adaptive-width/output $SAVE_BEST_CHECKPOINT $SAVE_EPOCH_CHECKPOINTS \
        --patience $PATIENCE \
        $TRAIN_SLIMMABLE_NETWORK \
        $USE_ROUTER \
        --internal_classifier_layer $INTERNAL_CLASSIFIER_LAYER \
        --dist-backend $DIST_BACKEND \
        $SAVE_PRED_RESULTS \
        $INIT_PRIVATE_LAYERNORMS_FROM_SCRATCH \
        --hidden_size_router $HIDDEN_SIZE_ROUTER \
        --report_to_wandb \
        --lambda_loss_task $LAMBDA_LOSS_TASK \
        --lambda_loss_router $LAMBDA_LOSS_ROUTER \
        $WRITE_VAL_PREDS \
        $DATA_AUG
fi