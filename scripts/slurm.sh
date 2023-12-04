#!/bin/bash
#SBATCH --job-name=SLURM_RUN
#SBATCH --output=/home/user/adaptive-width/slurm_outputs/%x.out
#SBATCH --error=/home/user/adaptive-width/slurm_outputs/%x.err
#SBATCH --time=96:00:00
#SBATCH --chdir=/home/user/adaptive-width/
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --mem=300G
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:<GPU_TYPE>:2
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=<EMAIL>
#SBATCH --signal=B:TERM@120
#SBATCH --exclude=<NODE>

echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# if HPARAM_FILE is not empty then read the hyperparameters from the hparam file instead of CLI
if [ ! -z "$HPARAM_FILE" ] ; then 
    # read the hyperparameters from the hparam file line hparam_id
    echo "Reading hyperparameters from hparam file ${HPARAM_FILE}."
    INPUT=$HPARAM_FILE

    while read -r line; do
        # Split the line into two values using space as the delimiter
        read -r name value <<< "$line"
        # Create a variable with the name and assign the value to it
        export "$name=$value"
        echo "$name=$value"
    done < "${INPUT}"
fi

seed=$RANDOM
offset=12867
export MASTER_PORT=$((seed+offset))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

srun --cpu_bind=v --accel-bind=gn /usr/bin/bash /home/user/adaptive-width/scripts/run.sh