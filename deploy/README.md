# OmniDrive Deployment with TensorTR-LLM and TensorRT-Edge-LLM
This document demonstrates the deployment of the [OmniDrive](https://arxiv.org/abs/2405.01533) utilizing [TensorRT](https://github.com/NVIDIA/TensorRT), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM). In this deployment demo, we will use [EVA](https://arxiv.org/abs/2303.11331)-base as the vision backbone and [TinyLlama](https://arxiv.org/abs/2401.02385) as the LLM head. We will provide a overview of the overall deployment strategy and result analysis.

## Table of Contents

1. [Deployment strategy](#strategy)
2. [Deployment Guideline](#guide)
3. [Results analysis](#result)
    - [Accuracy performance](#acc)
    - [Inference latencies](#latency)
4. [References](#ref)

## Deployment strategy <a name="strategy"></a>
The OmniDrive employs [EVA](https://arxiv.org/abs/2303.11331) as the vision backbone, [StreamPETR](https://arxiv.org/abs/2303.11926) for both the bounding box detection head and the map head, and a LLM model as the planning head. For deployment, we utilize `EVA-base` as the backbone and [TinyLlama](https://arxiv.org/abs/2401.02385) as the LLM head.

To enhence inference efficiency, engines are built seperately for the vision component (EVA backbone and StreamPETR necks) and the LLM component (TinyLlama). Below are the pipelines for deploying the two components:
 - The vision component: 
    1) export ONNX model
    2) build engines with TensorRT
 - The LLM component:
    - If using TensorRT-LLM
        1) convert the checkpoints to Hugging Face safetensor with TensorRT-LLM
        2) build engines with `trtllm-build`
    - Or if using TensorRT-Edge-LLM
        1) export ONNX model
        2) build engines with TensorRT scripts

<img src="../assets/deployment_strategy.png" width="1000">

For TensorRT-LLM, we use TensorRT 10.4, and TensorRT-LLM 0.13 to deploy the OmniDrive on A100 GPU, X86_64 Linux platforms; for TensorRT-Edge-LLM, we use TensorRT 10.14 and the TensorRT-Edge-LLM 0.4.0 (latest) to deploy the model on NVIDIA DRIVE AGX Thor platform. Notice that TensorRT-Edge-LLM also support X86_64 deployment. 

Please refer to this [config](../projects/configs/OmniDrive/eva_base_tinyllama.py) for model details. 

## Deployment Guideline <a name="guide"></a>
Please refer to the [TensorRT-LLM deployment guideline](./docs/trtllm_guide.md) and [TensorRT-Edge-LLM deployment guideline](./docs/edgellm_guide.md) for the detailed instructions on how to set up environment, build engines, and run engine inference with TensorRT-LLM and TensorRT-Edge-LLM.

## Results analysis <a name="result"></a>
### Accuracy performance <a name="acc"></a>
Here are the performance comparisons between the PyTorch model and engines.
Vision    | LLM  | BBOX mAP |  Planning L2 1s | Planning L2 2s | Planning L2 3s 
--------------------- | ---- | -------- | --- | --------------------------------| ------- 
PyTorch | PyTorch  | 0.354 |  0.151 | 0.314 | 0.585
FP32 engine | FP16 engine  | 0.354 | 0.150 | 0.312 | 0.581
FP32 engine | FP16 activation INT4 weight  |0.354|0.157|0.323|0.604
Mixed precision engine | FP16 engine  | 0.306 |0.166|0.337|0.615
Mixed precision engine | FP16 activation INT4 weight  | 0.306|0.171|0.349|0.634

### Inference latencies <a name="latency"></a>
Here is the runtime latency analysis for the engines. The units of the reference latency numbers in the following tables are `ms`.

**On an A100 with TensorRT-LLM**:
Vision | PyTorch | FP32 vision engine | Mixed-precision vision engine
--------| ------- | ------- | ------
Latency (ms) | 280.02 | 75.66   | 25.92

LLM                      | PyTorch  | FP16 LLM engine | FP16 activation INT4 weight
--------------------------   | -------  | -------- | ---
Time To First Token (TTFT) (ms)   | 107.82  |  10.02   | 11.30
Time Per Output Token (TPOT) (ms) |   27.49  |  2.80   |  2.52
Overall Latency (ms)              | 2256.50  | 256.20   |  232.60

**On NVIDIA DRIVE AGX Thor with TensorRT-Edge-LLM**:
Vision  | FP32 vision engine | Mixed-precision vision engine
-------- | ------- | -------
Latency (ms)  | 316.90 | 177.68

LLM                       | FP16 LLM engine | NVFP4 LLM engine
--------------------------     | -------- | -------- 
Time To First Token (TTFT) (ms)     |  16.55 |  9.85
Time Per Output Token (TPOT) (ms)   |  9.04   | 4.45
Overall Latency (ms)                 | 794.25  | 392.55

## References <a name="ref"></a>
1. [EVA paper](https://arxiv.org/abs/2303.11331)
2. [StreamPETR paper](https://arxiv.org/abs/2303.11926)
3. [TinyLlama paper](https://arxiv.org/abs/2401.02385)
4. [TensorRT repo](https://github.com/NVIDIA/TensorRT)
5. [TensorRT-LLM repo](https://github.com/NVIDIA/TensorRT-LLM)
