#!/usr/bin/env bash

TRT_HOME=$1
CONFIG=$2
CHECKPOINT=$3
VISION_ENGINE=$4
LLM_ENGINE=$5
LLM_ONNX_PATH=$6
PLUGIN_PATH=$7
SAVE_PATH=$8
DUMP_NUMBER=$9
DUMP_PATH=$10

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
    ./deploy/test_edgellm.py \
    --engine_pth ${VISION_ENGINE} \
    --eval bbox \
    --config ${CONFIG} \
    --launcher pytorch \
    --llm_engine_pth ${LLM_ENGINE} \
    --llm_onnx_path ${LLM_ONNX_PATH} \
    --plugin_path ${PLUGIN_PATH} \
    --tokenizer_pth ${CHECKPOINT} \
    --qa_save_path ${SAVE_PATH} \
    --llm_checkpoint ${CHECKPOINT} \
    --dump_number ${DUMP_NUMBER} \
    --dump_path ${DUMP_PATH}
