#!/bin/bash

#SBATCH --account=a-a06
#SBATCH --time=05:59:59
#SBATCH --job-name=Marlin
#SBATCH --output=/capstor/scratch/cscs/%u/Marlin/logs/slurm/training/R-%x-%j.out  # ⚠️ WARNING ⚠️ Make sure this path exists!
#SBATCH --error=/capstor/scratch/cscs/%u/Marlin/logs/slurm/training/R-%x-%j.err   # ⚠️ WARNING ⚠️ Make sure this path exists!
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/NeMo/nemo-latest.toml
#SBATCH --signal=SIGUSR2@600    # Send SIGUSR2 (Marlin Auto-checkpoint saver signal) 600 seconds before hitting the time limit
#SBATCH --no-requeue            # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

echo "[$(date)] $(sbatch --dependency=singleton $0)"

################ Configs ################
MODEL=Llama3.2-3B # Supported models: `Llama3-70B`, `Llama3-8B`, `Llama3.2-3B`, `Llama3.2-1B`
DATASETS="/capstor/store/cscs/swissai/a06/.datasets_tj/Llama-3.1-70B/fineweb-edu-full"

GBS=256
SEQ_LEN=8192
TRAINING_STEPS=5000
CHECKPOINT_STEPS=1000

#### Debugging ####
LOG_NCCL=false # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false # Turn on the NSYS profiler. Check `model.nsys_profile` options
MOCK_DATA=false # Set to `true` to use mock data
###################

# Directories, Logging & Artifacts
PROJECT_NAME=Clariden-Marlin
EXP_NAME=Marlin-$MODEL-NODES-$SLURM_NNODES
MARLIN_DIR=/capstor/scratch/cscs/$USER/Marlin
MARLIN_RUNS_DIR=$MARLIN_DIR/logs/Marlin-Runs # Path to store training logging artefacts
CKPT_DIR=/iopsstor/scratch/cscs/$USER/Marlin-Checkpoints/$PROJECT_NAME/$EXP_NAME # Path to store checkpoints ⚠️ WARNING ⚠️ MUST be in /iopsstor/scratch ⚠️ WARNING ⚠️
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache # Path to store cache from datasets ⚠️ WARNING ⚠️ MUST be in /iopsstor/scratch ⚠️ WARNING ⚠️
WANDB_KEY_DIR=/capstor/scratch/cscs/asolergi/.keys/wand_token.txt # Path to a .txt file containing a WANDB key. If not set WANDB will be disabled
#########################################

PROJECT_DIR=$MARLIN_RUNS_DIR/$PROJECT_NAME
LOG_EXP_DIR=$PROJECT_DIR/$EXP_NAME
TRIGGER_DIR=$LOG_EXP_DIR/triggers
DEBUG_DIR=$LOG_EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt

# Setup directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $LOG_EXP_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
ln -sfn $CKPT_DIR $LOG_EXP_DIR/checkpoint-dir-link

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

# Setup ENV
cd $MARLIN_DIR
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=8 # https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/5730fac6f97795931325cab0ac5dce1924cdcb3f/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py#L59
export NVTE_BWD_LAYERNORM_SM_MARGIN=8 # https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/5730fac6f97795931325cab0ac5dce1924cdcb3f/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py#L59

srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\\s*[0-9]*MiB")' > $GPU_MEM_LOGGING
ulimit -c 0

# General Args
GENERAL_ARGS=(
    ++run.name=$EXP_NAME
    ++trainer.num_nodes=$SLURM_NNODES
    ++exp_manager.explicit_log_dir=$LOG_EXP_DIR
    ++exp_manager.checkpoint_callback_params.dirpath=$CKPT_DIR
    ++exp_manager.checkpoint_callback_params.every_n_train_steps=$CHECKPOINT_STEPS
)

# Training Args: Number of steps, global batch size, sequence length
TRAINING_ARGS=(
    ++trainer.max_steps=$TRAINING_STEPS
    ++model.global_batch_size=$GBS
    ++model.encoder_seq_length=$SEQ_LEN
)

# Data Args
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS=(
    ++model.data.data_impl=mock
    ++model.data.data_prefix=/dont/raise/a/missing/arg/error
  )
else
  DATA_ARGS=(
      ++model.data.index_mapping_dir=$DATASET_CACHE_DIR
      ++model.data.data_prefix=[$(python3 $MARLIN_DIR/scripts/tools/create_data_config.py -p $DATASETS)]
  )
fi

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 -u $MARLIN_DIR/megatron_gpt_pretraining.py \
    --config-path=$MARLIN_DIR/configs \
    --config-name=$MODEL.yaml \
    ${GENERAL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]}"

# Resume from checkpoint logic
echo "[$(date)] Looking to remove unfinished checkpoints in $CKPT_DIR..."
python3 $MARLIN_DIR/scripts/tools/remove_unfinished_checkpoints.py $CKPT_DIR
echo "[$(date)] Removed all unfinished checkpoints in $CKPT_DIR"
# If CKPT_DIR is not empty a.k.a we have checkpoints
if [ -n "$( ls -A $CKPT_DIR )" ]; then
   RESUME_CKPT_DIR=$(python3 $MARLIN_DIR/scripts/tools/setup_resume.py $CKPT_DIR)
   echo "[$(date)] Resuming training from $RESUME_CKPT_DIR"
   TRAINING_CMD="$TRAINING_CMD '++exp_manager.resume_from_checkpoint=$RESUME_CKPT_DIR'"
else
    echo "[$(date)] Starting training from scratch (Any checkpoints in $CKPT_DIR)"
fi

if [ -n "$WANDB_KEY_DIR" ]; then
  echo "[$(date)] Setting WANDB logging"
  export WANDB_API_KEY=$(cat $WANDB_KEY_DIR)
  if [ -d "$LOG_EXP_DIR/wandb/latest-run" ]; then
    echo "[$(date)] Syncing WANDB from previous run"
    wandb sync $LOG_EXP_DIR/wandb/latest-run
  fi
  TRAINING_CMD="$TRAINING_CMD ++exp_manager.create_wandb_logger=True ++exp_manager.wandb_logger_kwargs.name=$EXP_NAME-$SLURM_JOB_ID ++exp_manager.wandb_logger_kwargs.project=$PROJECT_NAME"
else
  export WANDB_MODE=disabled
  echo "[$(date)] WANDB Logging disabled"
fi

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile --trace='nvtx,cudnn,cublas,cuda' --cuda-memory-usage='true' --output=$DEBUG_DIR/nsys-trace.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD ++model.nsys_profile.enabled=True"
fi

# Checkpoint Compute Environment
echo -e "$(date)" > $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nCMD: $CMD_PREFIX $TRAINING_CMD" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nSlurm file: $0\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $0 >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nTOML file: $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nNeMo: /opt/NeMo, $(git -C /opt/NeMo config --get remote.origin.url) ($(git -C /opt/NeMo rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nMegatron: /opt/Megatron-LM, $(git -C /opt/Megatron-LM config --get remote.origin.url) ($(git -C /opt/Megatron-LM rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\n$(pip list)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\n$(nvidia-smi)" >> $COMPUTE_ENVIRONMENT_DIR # CUDA Version & Driver
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nEnvironment Variables:\n\n$(printenv)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR

srun -lu --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "FINISH TIME: $(date)"
