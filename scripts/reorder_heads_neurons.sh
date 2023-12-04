# a list of task name
export TRANSFORMERS_CACHE=/home/user/transformers_cache
export HUGGINGFACE_HUB_CACHE=/home/user/huggingface_hub_cache
export HF_DATASETS_CACHE=/home/user/huggingface_datasets_cache

# TASKS=("MNLI QNLI RTE QQP STS-B CoLA SST-2 MRPC")
TASKS=("WNLI")
# TASKS=("STS-B CoLA SST-2 MRPC")
# TASKS=("CoLA SST-2 MRPC")

# for loop over tasks and for each one run run_glue.py
for TASK in $TASKS
do
cd /home/user/adaptive-width/nlp/src && python run_glue.py \
  --model_type roberta \
  --num_workers 1 \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 1 \
  --epochs 1 \
  --name debug \
  --distributed False \
  --width_mult_list 0.25,1.0 \
  --task_name $TASK \
  --do_train \
  --data_dir /home/user/adaptive-width/nlp/glue/glue_data/$TASK \
  --model_dir /home/user/roberta-base \
  --compute_neuron_head_importance 
done

# --model_dir /home/user/bert-base-uncased \