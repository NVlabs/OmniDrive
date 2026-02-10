#!/usr/bin/env bash

TRT_HOME=$1
CONFIG=$2
TOKENIZER=$3
VISION_ENGINE=$4
LLM_ENGINE=$5
SAVE_PATH=$6

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH PYTHONPATH="./":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=1 \
    --master_port=$PORT \
    ./deploy/test_trtllm.py \
    --engine_pth ${VISION_ENGINE} \
    --eval bbox \
    --config ${CONFIG} \
    --launcher pytorch \
    --llm_engine_pth ${LLM_ENGINE} \
    --tokenizer_pth ${TOKENIZER} \
    --qa_save_path ${SAVE_PATH}
