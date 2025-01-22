#!/bin/bash

#SBATCH --account=a-a06
#SBATCH --time=00:19:59
#SBATCH --job-name=Marlin-dataset-cache
#SBATCH --output=/capstor/scratch/cscs/%u/Marlin/logs/slurm/training/R-%x-%j.out  # ⚠️ WARNING ⚠️ Make sure this path exists!
#SBATCH --error=/capstor/scratch/cscs/%u/Marlin/logs/slurm/training/R-%x-%j.err   # ⚠️ WARNING ⚠️ Make sure this path exists!
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/NeMo/nemo-latest.toml
#SBATCH --no-requeue            # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

################ Configs ################
MARLIN_DIR=/capstor/scratch/cscs/$USER/Marlin
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache # Path to store cache from datasets ⚠️ WARNING ⚠️ MUST be in /iopsstor/scratch ⚠️ WARNING ⚠️

MODEL=Llama3.2-3B # Supported models: `Llama3-70B`, `Llama3-8B`, `Llama3.2-3B`, `Llama3.2-1B`. Not relevant for cache preparation, but recommended to set the same exact config than for training
DATASETS=/capstor/store/cscs/swissai/a06/datasets_tokenized/nemo/Llama-3.1-70B/fineweb-2

GBS=1024
SEQ_LEN=8192
TRAINING_STEPS=25000
#########################################

##### Setup ENV #####
cd $MARLIN_DIR
GPUS_PER_NODE=1

MASTER_ADDR=$(hostname)
MASTER_PORT=25678
#####################

# Training Args: Number of steps, global batch size, sequence length
TRAINING_ARGS=(
    ++trainer.max_steps=$TRAINING_STEPS
    ++model.global_batch_size=$GBS
    ++model.encoder_seq_length=$SEQ_LEN
)

DATA_ARGS=(
    ++model.data.index_mapping_dir=$DATASET_CACHE_DIR
    ++model.data.data_prefix=[$(python3 $MARLIN_DIR/scripts/tools/create_data_config.py -p $DATASETS)]
)

LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

TRAINING_CMD="$LAUNCHER $MARLIN_DIR/scripts/dataset_cache/build_dataset_caches.py \
    --config-path=$MARLIN_DIR/configs \
    --config-name=$MODEL.yaml \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]}"

CMD_PREFIX="numactl --membind=0-3"

srun -lu --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "FINISH TIME: $(date)"
