# Redteaming Challenge

[News] **Second Place** in both base & large model subtracks of Red Teaming LLM@**NeurIPS 2023, Torjan Detection Challenge**

This is the submitted version of Trojan Detection Challenge 2023 - Red Teaming Track (Base/Large Model Subtrack)

## Table of Contents

- [Installation](#installation)
- [Models](#models)
- [Preparations](#Preparations)
- [Experiments](#experiments)
- [Select](#Select)
- [Reproducibility](#reproducibility)

## Installation

The enviroment all we need can be installed by running the following command at the root of this repository:

```bash
pip install -e .
```

## Models

To modify the paths to your models and tokenizers, please add the following lines in `experiments/configs/xxx.py`. For base subtrack, please modify `experiments/configs/llama2_chat_7b.py`, for large subtrack, please modify `experiments/configs/llama2_chat_13b.py`. An example is given as follows.

```python
config.model_paths = [
    "meta-llama/Llama-2-7b-chat-hf"
]
config.tokenizer_paths = [
    "meta-llama/Llama-2-7b-chat-hf"
]
```

## Preparations
First, place the behaviors file `behaviors.json`, which contains a list of undesirable behaviors you want to elicit, under the specific folder `data/behavior/`

Other preparations can be finished by running the following command at the folder `prepare`. If you want to attack llama2-caht-7b, then `subtrack` is `base`. If you want to attack llama2-caht-13b, then `subtrack` is `large`.

**Note**: We need `Llama-2-7b-hf` to automatically generate targers used in our method, so you need to prepare your huggingface token.
```bash
cd prepare
bash prepare.sh $subtrack
```

## Experiments 

The `experiments` folder contains code and bash scripts to reproduce our redteaming process. (Note: To comply with competition restrictions, our code is designed to run on an 80GB A100 GPU by default. So if running for the competition, please do not modify the efficiency-related config to reproduce the best results within the time limit. Otherwise if there is not enough GPU memory (80GB) and an 'Out of Memory' error occurs, you can turn off the switch that accelerates code execution. In the `experiments/configs/template.py` file, set `config.enable_prefix_sharing=False`, which will reduce the memory usage to below 40GB for 13b model and below 50GB for 7b model but slow down the code execution. And you can rewrite the bash scripts to cancel the parallel running on one GPU for 7b model, which will further cut half mem off for GPU.)

**Note:** All bash or sbatch scripts default tasks number is set to exceed the time restriction (for the concern that different hardware may cause a shift in running efficiency), so if it exceeds, please terminate the task manually and use the testcases generated within the time restriction.

### Running the Experiment

In order to reproduce the competition requirements of running the experiment in 2/4 A100 GPU days, we provide bash scripts `experiments/bash_scripts` that can be used out of the box, and all running settings and hyperparameters are included in bash scripts and `experiments/configs/`. 
- `GPU_NUM` is the number of parallel GPUs, `bash_llama2_chat_7b.sh` is for base subtrack(llama2-chat-7b), `bash_llama2_chat_13b.sh` is for large subtrack(llama2-chat-13b)
- Since the competition does not require multi-GPU parallelism, we have not implemented optimizations for multi-GPU parallelism. Therefore, the optimal results are achieved with GPU_NUM=1. **It is recommended to set `GPU_NUM`=1**. If multi-GPU parallelism is needed, there may be instances of idle waiting on some GPUs. **To achieve more accurate timing in a multi-GPU scenario, please use timing through log files, and the timing tool is provided in `experiments/cal_time.ipynb`**. This tool calculates the total time used for the experiment from the .out file. This may help confirm whether the experiment was completed within the time limit. 
- **Note**: For base subtrack we default to running two subtasks in parallel on a single GPU. Timing through the log file involves measuring the total runtime for each subtask. Therefore, for the base subtrack, the displayed time when timing through the log file is equivalent to half of the actual GPU usage time. **Hence, for both the base and large subtracks, the time limit as shown by the tool in `experiments/cal_time.ipynb` is 96 hours.**

```bash
cd experiments/bash_scripts
bash bash_llama2_chat_7b.sh $GPU_NUM
bash bash_llama2_chat_13b.sh $GPU_NUM
```


## Select
Select operations can be finished by running the following command at folder `select`. If you want to attack llama2-caht-7b, then `subtrack` is `base`. If you want to attack llama2-caht-13b, then `subtrack` is `large`. And modify the `model_path` to your models and tokenizers (as shown in [Models](#models))

**Note**: Please set your `OPENAI_API_KEY` in `select/classify.py`, since our method need to use GPT-4 as the auto-selector for our test case.

```bash
cd select
bash select.sh $subtrack $model_path
```

## Reproducibility

A note for hardware: all experiments we run use one or multiple NVIDIA A100 GPUs, which have 80G memory per chip. 

