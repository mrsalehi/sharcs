# make an array called TASK_NAMES with 1 and 2 as the elements
# TASK_NAMES=(cola mnli mrpc qnli qqp rte sst-2 sts-b wnli)
# TASK_NAMES=("CoLA" "MNLI" "MRPC" "QNLI" "RTE" "SST-2" "STS-B" "WNLI")
TASK_NAMES=("MNLI") 
ADAPTIVE_LAYER_IDXS=(3 6 9)
WIDTH_MULTS=(0.25 0.5 0.75)

# iterate over the elements of TASK_NAMES
for TASK_NAME in "${TASK_NAMES[@]}"; do
    for ADAPTIVE_LAYER_IDX in "${ADAPTIVE_LAYER_IDXS[@]}"; do
        for WIDTH_MULT in "${WIDTH_MULTS[@]}"; do
            #set $WIDTH_MULT_STR to $WIDTH_MULT without any decimal points
            WIDTH_MULT_STR=$(echo $WIDTH_MULT | tr -d '.')
            JOB_NAME="${TASK_NAME}_adaptive_layer_${ADAPTIVE_LAYER_IDX}_${WIDTH_MULT_STR}_width_second_run"
            TASK=$TASK_NAME COPY_CODE_BASE_TO_TMP=True ADAPTIVE_LAYER_IDX=$ADAPTIVE_LAYER_IDX WIDTH_MULT=$WIDTH_MULT sbatch --job-name $JOB_NAME --requeue slurm.sh
        done
    done
done