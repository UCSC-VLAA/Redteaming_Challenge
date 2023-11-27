#!/bin/bash

cd ../launch_scripts

export id=$1
export gpu=$2
export n_train=$3
export offset=$4

export folder=../bash_scripts/out
export i=$id
export warm_up_start=0
export warm_up_incre=$[$id/3]
export batch_size=128

if [ $[$warm_up_start+$warm_up_incre] -ge 10 ]; then
    export warm_up=1.0
else
    export warm_up=0.$[$warm_up_start+$warm_up_incre]
fi

if [ ! -d $folder ]; then
    mkdir $folder
    echo "Folder ${folder} created."
else
    echo "Folder ${folder} already exists."
fi

CUDA_VISIBLE_DEVICES=$gpu bash run.sh llama2_chat_7b behaviors $n_train $offset $batch_size $warm_up > ${folder}/7b_${offset}_${i}.out &

wait