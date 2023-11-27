#!/bin/bash

export subtrack=$1
export model_path=$2
choice=(base large)
if ! [[ ${choice[*]} =~ "$subtrack" ]] ; then
    echo "subtrack's name can only be base or large, please input the right subtrack"
    exit 0
fi

if [ ! -d 'process' ]; then
    mkdir process
    echo "Folder 'process' created."
else
    echo "Folder 'process' already exists."
fi

cp ../experiments/testcase/${subtrack}_our_testcase.json process/${subtrack}_our_testcase.json

if [ ! -f "process/${subtrack}_our_generation.json" ]; then
    echo "Generating the response of testcase for subtrack ${subtrack}......"
    CUDA_VISIBLE_DEVICES=0 python -u local_generate.py \
                                    --model_path=$model_path \
                                    --subtrack=$subtrack
else
    echo "process/${subtrack}_our_generation.json already exist"
fi

if [ ! -f "process/${subtrack}_our_testcase_purify.json" ]; then
    echo "Classifying the generation for subtrack ${subtrack}......"
    python -u classify.py \
            --subtrack=$subtrack
else 
    echo "process/${subtrack}_our_testcase_purify.json already exist"
fi

echo "Sampling the testcase for subtrack ${subtrack}......"
python -u sample.py \
        --subtrack=$subtrack

echo "The final testcase to submit is 'select/${subtrack}_our_testcase.json'"
