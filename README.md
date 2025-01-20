<h1 align="center">
<p>Marlin
</h1>

<p align="center">
 <img src="images/MarlinNeMo.png" width="25%"/>
<!-- TOC -->

- [Introduction](#introduction)
- [Submitting jobs with Marlin](#submitting-jobs-with-marlin)
- [The Configuration file](#the-configuration-file)
    - [Experimenting with New Models and Hyperparameters](#experimenting-with-new-models-and-hyperparameters)
- [Directory structure](#directory-structure)
- [Fault Tolerance](#fault-tolerance)
- [Quality of Life Improvements](#quality-of-life-improvements)
- [Experiment tracking & debugging](#experiment-tracking--debugging)
- [Checkpointing](#checkpointing)
    - [Resuming from a checkpoint](#resuming-from-a-checkpoint)
- [Data](#data)
    - [Tokenization](#tokenization)
    - [Set the Datasets in NeMo](#set-the-datasets-in-nemo)
- [Environment](#environment)

<!-- /TOC -->

<!-- /TOC -->t](#environment)

<!-- /TOC -->

# Introduction
Marlin is a launcher designed to train large language models (LLMs) at scale using NeMo, which leverages Megatron-LM, specifically tailored for Slurm clusters. It's tailored for large-scale training runs and include mechanism for improved fault tolerance, automatic resumption after crashes or interruptions, automated evaluation submission, and enhanced WANDB logging. Additionally, it provides configurations optimized for peak performance on the Alps Supercomputer.

Marlin also features scripts to tokenize data for NeMo & Megatron using datatrove across multiple nodes and tools to convert model weights to the HuggingFace format.

# Submitting jobs with Marlin
Before submitting your first experiment with Marlin, review the `submit_Marlin.sh` script and the configuration files in the `configs/` directory. Currently, Marlin supports a variety of Llama3 models, and we plan to add support for more models soon.

>[!CAUTION]
> 🚨 On the Alps Supercomputer, ensure you **DO NOT WRITE**  under **`/capstor/store`**. Use `/iopsstor/scratch` instead 🚨

To use the `submit_Marlin.sh` script, ensure the following configurations are set up correctly:
- Ensure the paths specified for `#SBATCH --output` and `#SBATCH --error` exist.
- `MODEL`: Currently supported models: `Llama3-70B`, `Llama3-8B`, `Llama3.2-3B` & `Llama3.2-1B`
- `DATASETS`: Comma-separated list of paths containing the tokenized documents.
- `PROJECT_NAME` & `EXP_NAME`: These follow the same style as WANDB, allowing you to group multiple experiments under the same project.
- `MARLIN_DIR`: The absolute path to this codebase (e.g. `/capstor/scratch/cscs/$USER/Marlin`).
- `MARLIN_RUNS_DIR`: Directory to store all logging artifacts.
- `CKPT_DIR`: Directory for storing model checkpoints. A symlink to this directory will be created in `MARLIN_RUNS_DIR`.
- `DATASET_CACHE_DIR`: Directory to store dataset indexes. This is crucial for reusing dataset indexes across runs, especially when working with large datasets.
- `WANDB_KEY_DIR`: Specify the path to a file containing your WANDB key. If you do not wish to use WANDB logging, disable it by setting `export WANDB_MODE=disabled`.

Once all configurations are in place, submit the job to Slurm with `sbatch submit_Marlin.sh` and voilà!

# The Configuration file

The launcher uses Hydra to manage all configurations. For the Llama3 family of models, we start with `configs/Llama3-base.yaml`, which contains all available configuration options. Each specific model then overrides some of these values as needed.

This base configuration file includes comments about the most important settings and follows a naming convention similar to other projects like `transformers`, `nanotron`, or `Megatron-LM`.

For the Llama3 models, we have adopted the default configurations provided by NVIDIA in NeMo 2.0. You can check them [here](https://github.com/NVIDIA/NeMo/blob/7167e5e8176c2651114546e088e8fc78e2888213/nemo/collections/llm/gpt/model/llama.py#L93-L213).

## Experimenting with New Models and Hyperparameters

To experiment with new models, we recommend creating a new *.yaml* file for the model configuration. However, you can also override specific parameters directly in the `submit_Marlin.sh` script by using Hydra's CLI syntax (e.g. `++model.optim.shed.warmup_steps=500`)

# Directory structure
```
📦Marlin-Llama3.2-3B-NODES-1
 ┣ 📂checkpoint-dir-link
 ┣ 📂debug
 ┃ ┗ 📂87167
 ┃ ┃ ┣ 📜compute_environment.txt
 ┃ ┃ ┗ 📜memory_logging.txt
 ┣ 📂triggers
 ┗ 📂wandb
```
The launcher creates a project and an experiment folder under `MARLIN_RUNS_DIR`, where all logging and debugging artifacts for the run are stored. Similar to WANDB, a project can group multiple experiments.
Within each experiment folder, you will find:
- `debug`: For each job, this folder contains:
    - A file storing the compute environment of the job (e.g., Pip packages, nodes, CUDA version, etc.).
    - A snapshot of the GPU memory usage before execution begins. This is useful for detecting GPUs with memory leaks and mapping the Slurm job step ID to the different nodes.
    - NCCL logs and NSYS traces (if these configurations are enabled).
- `wandb`: Contains all WANDB-related artifacts.
- `triggers`: In this folder we will create the save and exit triggers. (More details in the next section.)
- `checkpoint-dir-link`: A symlink to the `CKPT_DIR`, where model checkpoints are stored.

Using the same project and experiment naming convention, the launcher creates a folder under `CKPT_DIR` to store the model checkpoints. Unlike `MARLIN_RUNS_DIR`, it is **crucial** that the `CKPT_DIR` supports high read/write speeds to minimize the time required to pause training while saving a checkpoint and the time it takes to load the checkpoint when resuming training.

# Fault Tolerance

Training of LLMs often spans several weeks or even months. However, compute clusters typically enforce job time limits of 12 to 24 hours. This results in frequent training interruptions, either due to these time limits or hardware crashes.

To avoid constantly monitoring the training process, we have designed a fault-tolerance system that ensures seamless continuity of training. The system handles:
- Maintaining Continuous Job Execution. Marlin ensures there is always one active job running and another queued. This is achieved using the `sbatch --dependency=singleton` flag, which guarantees that only one job with the specified name (owned by the user) can be running or suspended at any given time.
- Graceful Exit Before Time Limit. Marlin saves a checkpoint and exits gracefully a few minutes before the job time limit is reached. This allows checkpoint frequency to remain independent of the Slurm time limit while utilizing the full time window for training. This is achieved with `#SBATCH --signal=SIGUSR2@600`, which sends the `SIGUSR2` signal 600 seconds before the time limit. The signal is captured during the run to trigger the checkpoint save.
- Automatic Recovery: Automatically resumes training from the most recent checkpoint, recovering model weights, optimizer states, Dataloader state and WANDB logging.

These mechanisms are implemented across `submit_Marlin.sh`, `marlin/marlin_callback.py` and `marlin/signal_handler.py`.

# Quality of Life Improvements
We have also incorporated several quality-of-life features to streamline the training process:
- Save & Exit Triggers. By default, Slurm only allows the job owner to interact with a running job, which creates challenges during long runs when multiple team members are responsible for monitoring the process. To address this limitation, we have implemented a mechanism based on the presence of specific files (triggers):
  - Save Trigger: When the save trigger file is detected, the system will schedule a checkpoint save.
  - Exit Trigger: When the exit trigger file is detected, the system will gracefully exit the run and cancel all remaining jobs.

  This allows team members (not just the job owner) to intervene when necessary. The verification of these files will be carried out every `trainer.log_every_n_steps` steps.
- Automatic Evaluation Submission. After every checkpoint save, the system automatically submits another script to the queue that runs `lm-evaluation-harness` benchmarks.
- Automatic Checkpoint Security Copy. Another job is automatically submitted to create a security copy of the checkpoint after each checkpoint save.
- Logging `consumed_tokens`. Marlin logs the `consumed_tokens` metric to WANDB. This is especially helpful when conducting ablations.

# Experiment tracking & debugging
During training, every run will be tracked and logged using the following tools:
- WANDB. To enable WANDB tracking, set `WANDB_KEY_DIR` in `submit_Marlin.sh` to a path containing a file with your WANDB key. If not set, WANDB logging is **disabled** by default. As previously mentioned, WANDB logging will automatically resume correctly after recovering from a checkpoint. Example [run](https://wandb.ai/tj-solergibert/Marlin-Clariden-24H/workspace).
>[!WARNING]
> Ensure you plot your WANDB charts with the X-Axis set to one of the following: `global_step`, `consumed_tokens` or `consumed_samples`. Do not use `trainer/global_step` as it does **not** recover properly after an interruption.
- Tensorboard. To enable TensorBoard logging, set `exp_manager.create_tensorboard_logger=True` in your configuration. (It's **disabled** by default).
- Compute environment. At the start of every run, critical environment information is logged, such as the submitted job script, the installed pip packages, the CUDA driver and even all the environment variables. Refer to `submit_Marlin.sh` for the complete list of logged items.
- NCCL Logging. To debug networking issues, set `LOG_NCCL=true` in `submit_Marlin.sh` to enable NCCL info logging. To prevent cluttering the `#SBATCH --output` and `#SBATCH --error` files, each process logs to a separate file specified by `$NCCL_DEBUG_FILE` in `submit_Marlin.sh`.
- Nsys profiling. Set `NSYS_PROFILER=true` in `submit_Marlin.sh` to enable Nsys profiling and extract traces from a run. For details check `submit_Marlin.sh` &  the `Llama3-base.yaml` configuration file.

# Checkpointing
>[!CAUTION]
> 🚨 On the Alps Supercomputer, ensure you **DO NOT WRITE**  under **`/capstor/store`**. Use `/iopsstor/scratch` instead 🚨

Checkpointing is a critical component of LLM training. It must be fast to minimize disruption to training and complete, meaning it captures not only the model weights but also the optimizer states, DataLoader states and RNG states.

We use the PyTorch distributed checkpointing backend (`model.dist_ckpt_format: torch_dist`) leveraging the asynchronous checkpointing option and parallelizing both storing and loading checkpoints within all the devices. The checkpoints are topology-agnostic, allowing them to be loaded with a different topology from the one used to store them. Each process will store `dist_ckpt_torch_dist_multiproc` (Default: 2) files containing the state of the run.

In Alps, writing a Llama3-70B checkpoint to `/iopsstor/scratch` blocks training for approximately 40 seconds. Be aware that these checkpoints are huge: A Llama3-70B model checkpoint takes **1.4 TB**, the Llama3-8B version takes **147 GB** and the Llama3.2-3B takes **58 GB**.

In `submit_Marlin.sh` set `CKPT_STEPS` as the frequency every how many steps you want to store a checkpoint. Finally, keep in mind that in order to fully recover the DataLoader states we need to store the number of `consumed_samples` in the checkpoint folder name.
## Resuming from a checkpoint
There are 3 ways to resume from a checkpoint:
- Setting `exp_manager.resume_from_checkpoint` to recover the model weights, optimizer states and DataLoader states. This is the recommended approach for resuming training after an interruption during a long run.
- Setting `model.restore_from_path` to recover just the model weights. Useful for fine-tuning a pretrained model, as it resets the optimizer and sets the training iteration to 0.
- Setting `model.restore_from_ckpt` to recover the model weights and the optimizer states, but **not** the DataLoader states. We never recommend using this third option.

# Data
## Tokenization
While NeMo and Megatron provide data tokenization tools, we use `datatrove`, which enables us reading data in multiple formats (`json`, `parquet`, `csv`...), easy parallelization across multiple nodes and efficiently filter and deduplicate our data.

We have extended `datatrove` ([PR](https://github.com/huggingface/datatrove/pull/304)) to include the [`MegatronDocumentTokenizer`](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/pipeline/tokens/megatron_tokenizer.py#L144) pipeline stage, allowing the generation of files containing tokenized documents compatible with NeMo/Megatron.

All NeMo/Megatron tokenized files, also referred as *file prefixes*, consist of two file types:
- The `.bin` files contain raw tokens, where each token is either 4 bytes (for larger vocabularies) or 2 bytes.
- The `.idx` files contain metadata about the corresponding `.bin` files. For more details on creating these files, refer to [this example](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/pipeline/tokens/megatron_tokenizer.py#L59-L72).
>[!CAUTION]
> 🚨 On the Alps Supercomputer, ensure you **DO NOT WRITE**  under **`/capstor/store`**. Use `/iopsstor/scratch` instead 🚨

We include a launcher to tokenize data at scale using multiple nodes in `scripts/tokenization`. Start by preparing your workspace using the `scripts/tokenization/prepare_dumps.py` script, which identifies parquet files in the specified directory, filters them based on criteria (check `--filter-in` and `--filter-out`), and splits the workload evenly across `--n-dumps`.  This process generates *.txt* files for each dump, specifying the files to process.

Once the workspace is ready, configure the tokenization job by setting the tokenizer, the number of datatrove parallel workers per node, and the directory we prepared with `scripts/tokenization/prepare_dumps.py` in `scripts/tokenization/submit_tokenization.sh`. Running this script will submit multiple Slurm jobs, with each job responsible for processing one dump on a single node.
>[!CAUTION]
> 🚨 Ensure the `SBATCH --environment` flag in `scripts/tokenization/tokenize.sh` is correctly configured for your environment 🚨

Before running large-scale jobs, it is recommended to optimize the number of datatrove parallel workers  (`datatrove`'s [`LocalPipelineExecutor`](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/executor/local.py#L15) `tasks` & `workers`) and the input file size. You can easily modify the later with `datatrove` using the [`max_file_size`](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/pipeline/writers/parquet.py#L21) configuration of the `ParquetWriter`.

For example, on the Alps supercomputer, the best configuration involved processing parquet files of 500 MB with Snappy compression and using 28 datatrove workers per node, achieving a throughput of ~70 million tokens per second per node. More details [here](https://docs.google.com/presentation/d/1t12axPhvjpuxGQWr1xJIioazKeVZ212ewuyil5uWMnQ/edit#slide=id.p).

## Set the Datasets in NeMo
In NeMo, we will specify datasets using the `model.data.data_prefix `configuration. This field expects a list of tokenized file prefixes along with their respective weights. Example:
```yaml
model:
    data:
        data_prefix:
            - 0.2
            - tokenized-datasets/fineweb-2/dump_0/00000_tokens
            - 0.5
            - tokenized-datasets/fineweb-2/dump_1/00000_tokens
            - 0.3
            - tokenized-datasets/fineweb-2/dump_2/00000_tokens
```
For large runs consisting of hundreds or thousands of files, we will use the `scripts/create_data_config.py` script to automatically generate the list of file prefixes and their weights. Simply specify a comma-separated list of directories containing file prefixes in the `DATASETS` variable in `submit_Marlin.sh`.

# Environment
Although this project should be compatible with virtually any version of NeMo, the most relevant package versions are listed below.
- PyTorch NGC Container `nvcr.io/nvidia/pytorch:24.11-py3`

| Framework             | GitHub Repo                                                                                             | Commit                                     | Date       | Comments                                                                          |
|-----------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------|------------|-----------------------------------------------------------------------------------|
| Megatron              | [Branch](https://github.com/TJ-Solergibert/Megatron-LM/tree/goldfish)                                   | `1bb0513d4ac071b69c24ee394688e3810b623056` | 18/12/2024 | NVIDIA/Megatron `8d2bc4332c1aea92c7d69af5b220a37b1ba3cab3` commit + Goldfish loss |
| NeMo                  | [Branch](https://github.com/TJ-Solergibert/NeMo/tree/goldfish)                                          | `912d6a472a5b1c752401cc72f3b699c8fbf9d78d` | 18/12/2024 | NVIDIA/NeMo `97129886a712ae176cc8dd94a0597547206cabf7` commit + Goldfish loss     |
| nvidia-resiliency-ext | [Branch](https://github.com/NVIDIA/nvidia-resiliency-ext/tree/eb6cbc6f8004074640906f61accfcb7b7eab39cc) | `eb6cbc6f8004074640906f61accfcb7b7eab39cc` | 17/12/2024 | NVIDIA/nvidia-resiliency-ext main branch                                          |
| TransformerEngine     | [Branch](https://github.com/NVIDIA/TransformerEngine/tree/7f2afaaac23c10e37516fc8ff8f53103b9730c78)     | `7f2afaaac23c10e37516fc8ff8f53103b9730c78`   | 31/10/2024 | NVIDIA/TransformerEngine from the container                                       |
| PyTorch               | [Branch](https://github.com/pytorch/pytorch/tree/df5bbc09d191fff3bdb592c184176e84669a7157)              | `df5bbc09d191fff3bdb592c184176e84669a7157`   |  1/10/2024 | PyTorch from the container. Not 100% sure of the commit, pip list reports `torch 2.6.0a0+df5bbc09d1.nv24.11`                           |
