#!/bin/bash

export subtrack=$1
choice=(base large)
if ! [[ ${choice[*]} =~ "$subtrack" ]] ; then
    echo "subtrack's name can only be base or large, please input the right subtrack"
    exit 0
fi

if [ ! -f '../data/behavior/behaviors.json' ]; then
    echo "Please place the behaviors file 'behaviors.json' under the specific folder '../data/behavior/'"
    exit 0
fi

if [ ! -d '../experiments/icl' ]; then
    mkdir '../experiments/icl'
    echo "Folder '../experiments/icl' created."
else
    echo "Folder '../experiments/icl' already exists."
fi

if [ ! -d '../experiments/testcase' ]; then
    mkdir '../experiments/testcase'
    echo "Folder '../experiments/testcase' created."
else
    echo "Folder '../experiments/testcase' already exists."
fi

if [ ! -d '../data/auto_target' ]; then
    mkdir '../data/auto_target'
    echo "Folder '../data/auto_target' created."
else
    echo "Folder '../data/auto_target' already exists."
fi

if [ ! -f '../data/auto_target/harmful_behaviors.csv' ]; then
    echo "Auto_target_generating......"
    CUDA_VISIBLE_DEVICES=0 python -u auto_target_generate.py  \
                                    --behaviors_path="../data/behavior/behaviors.json" \
                                    --csv_path="../data/auto_target/harmful_behaviors.csv"
else
    echo "'../data/auto_target/harmful_behaviors.csv' already exists."
fi

echo "Preparing some record file for subtrack ${subtrack} during experiment......"
python -u prepare.py \
    --source_path="../data/behavior/behaviors.json" \
    --output_testcase_path="../experiments/testcase/${subtrack}_our_testcase.json" \
    --output_icl_dataset_path="../experiments/icl/icl_dataset_${subtrack}.json" \
    --output_prefix_dataset_path="../experiments/icl/prefix_dataset_${subtrack}.json" \
    --output_surffix_dataset_path="../experiments/icl/surffix_dataset_${subtrack}.json"