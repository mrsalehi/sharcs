# SHARCS: Efficient transformers through routing with dynamic width sub-networks
The official PyTorch implementation of the paper [SHARCS: Efficient transformers through routing with dynamic width sub-networks](https://arxiv.org/abs/2310.12126) (Findings of EMNLP 2023).

# Installation
Install the dependencies and the sharcs package from source with:
```
pip install -e .
```


# Usage
1. Download model checkpionts and place them under `models/` directory. You can download BERT and RoBERTa checkpoints from huggingface model hub. Make sure to include vocab and config and tokenizer files if there is any.

2. (Optional) To get better performance, before training the transformer encoder with SHARCS, reorder the heads so that more important heads are placed before less important ones in Multi-head attention modules. To find the importance of a head we follow DynaBERT's approach (https://arxiv.org/abs/2004.04037). To reorder the heads you can use `scripts/reorder_heads_neurons.sh` script.

3. To train a model on a GLUE task run `run_glue.py` python script with the necessary options. Note that when training multiple sub-networks, each one of them runs on a single GPU. For instance, for training with two sub-networks with 0.25 and full width, you need two GPUs. To enable such training, custom distributed samplers have been implemented (see `src/sharcs/sampler.py`).

4. To run experiments on an SLURM managed clusters, use the provided `scripts/slurm.sh` script. Note that you need to modify the script to match your cluster's configuration. The sbatch script will in turn run `scripts/run.sh` which will run the `run_glue.py` script. Here is an example command for running using SLURM with two sub-networks with 0.25 and full width on two GPUs (Note that env variables are passed to the sbatch script):
```
TASK=MNLI MODEL_TYPE=roberta INIT_PRIVATE_LAYERNORMS_FROM_SCRATCH=True WEIGHTED_DIM_REDUCTION=False PER_GPU_TRAIN_BATCH_SIZE=32 DISTRIBUTED=True REMOVE_PADS_IN_EVAL=True MEASURE_FLOPS=True SAVE_EPOCH_CHECKPOINTS=False SAVE_BEST_CHECKPOINT=False PER_GPU_EVAL_BATCH_SIZE=1 WIDTH_MULT=0.25,1.0 USE_ROUTER=True TRAIN_ROUTER_WINDOW_SIZE=3 ADAPTIVE_LAYER_IDX=3 REPEAT_SMALLER_SIZED_BUCKETS=True MODEL_DIR=models/roberta_reordered_heads/mnli_reordered_heads LAMBDA_LOSS_TASK=1.0 LAMBDA_LOSS_ROUTER=1.0 WIDTHS_CONFIG_FILE=configs/two_bucket/two_bucket_no_overlap_02.yaml EPOCHS=10 LEARNING_RATE=2e-5 sbatch --job-name MNLI_TRAIN --gres=gpu:<GPU_TYPE>:2 --ntasks-per-node=2 --nodes=1 --mem=64G --cpus-per-task=4 --account=<ACCOUNT> --time=72:00:00 --partition=<GPU_TYPE> scripts/slurm.sh
```

5. Evaluation code is included in `run_glue.py` script. You can measure the latency and FLOPS of the models with utilities provided in `src/sharcs/bench.py`.

(TODO: Better config system, e.g. reading the whole config from `yaml` file or using more robust and user-friendly CLI parsers such as absl flags).

# Apptainer
To make running the experiments easier (especially for HPC and cluster users), an Apptainer definition file is provided. Copy the `sharcs.def` file to your `/tmp` directory (or any directory where Apptainer has required permissions) and build the image from the definition file with the following command:
```
apptainer build --fakeroot /tmp/sharcs.sif sharcs.def
```

This results in a `sharcs.sif` image file. To run experiments within this image prepend the following command to all your commands:
```
apptainer exec -B $(pwd) --nv sharcs.sif
```


# NOTES
1. In the code the words "bucket" and "sub-network" are used interchangeably.