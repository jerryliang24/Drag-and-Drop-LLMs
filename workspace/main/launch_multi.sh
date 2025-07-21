#!/bin/bash
EXEC_FILE="$1"
NUM_GPUS=$2
RANDOM_NUM=$((RANDOM % 20000))

export NUM_PROCESSES=$NUM_GPUS




accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --main_process_port=$RANDOM_NUM \
    --num_machines=1 \
    --dynamo_backend=no \
    --mixed_precision=bf16 \
    $EXEC_FILE
