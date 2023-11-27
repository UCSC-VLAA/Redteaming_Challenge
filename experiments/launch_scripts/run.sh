#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # llama2 or vicuna
export setup=$2 # behaviors or strings
export ntrain=$3
export folder="../results_${model}"
export data_offset=$4
export batch_size=$5
export warm_up=$6

# Create results folder if it doesn't exist
if [ ! -d $folder ]; then
    mkdir $folder
    echo "Folder ${folder} created."
else
    echo "Folder ${folder} already exists."
fi

python -u ../main.py \
    --config="../configs/${model}.py" \
    --config.attack=gcg \
    --config.train_data="../../data/auto_target/harmful_behaviors.csv" \
    --config.result_prefix="${folder}/${setup}_${model}" \
    --config.n_train_data=$ntrain \
    --config.data_offset=$data_offset \
    --config.n_steps=150 \
    --config.test_steps=10 \
    --config.batch_size=$batch_size \
    --config.model=$model  \
    --config.setup=$setup \
    --config.warm_up=$warm_up
