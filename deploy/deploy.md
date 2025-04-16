# OmniDrive Deployment with TensorRT and TensorTR-LLM
This document demonstrates the deployment of the [OmniDrive](https://arxiv.org/abs/2405.01533) utilizing [TensorRT](https://github.com/NVIDIA/TensorRT) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). In this deployment demo, we will use [EVA](https://arxiv.org/abs/2303.11331)-base as the vision backbone and [TinyLlama](https://arxiv.org/abs/2401.02385) as the LLM head. We will provide a step-by-step overview of the deployment process, including the overall strategy, environment setup, engine build, engine benchmarking and result analysis.

## Table of Contents

1. [Deployment strategy](#strategy)
2. [Environment setup](#env)
3. [Vision engine build](#vision)
4. [LLM engine build](#llm)
5. [Benchmark](#bench)
6. [Results analysis](#result)
    - [Accuracy performance](#acc)
    - [Inference latencies](#latency)
7. [Future works](#future)
8. [References](#ref)

## Deployment strategy <a name="strategy"></a>
The OmniDrive employs [EVA](https://arxiv.org/abs/2303.11331) as the vision backbone, [StreamPETR](https://arxiv.org/abs/2303.11926) for both the bounding box detection head and the map head, and a LLM model as the planning head. For deployment, we utilize `EVA-base` as the backbone and [TinyLlama](https://arxiv.org/abs/2401.02385) as the LLM head.

To enhence inference efficiency, engines are built seperately for the vision component (EVA backbone and StreamPETR necks) and the LLM component (TinyLlama). Below are the pipelines for deploying the two components:
 - The vision component: 
    1) export ONNX model
    2) build engines with TensorRT
 - The LLM component:
    1) convert the checkpoints to Hugging Face safetensor with TensorRT-LLM
    2) build engines with `trtllm-build`

<img src="../assets/deployment_strategy.png" width="1000">

We use TensorRT 10.4, and TensorRT-LLM 0.13 to deploy the OmniDrive on A100 GPU, X86_64 Linux platforms. (see this [config](../projects/configs/OmniDrive/eva_base_tinyllama.py) for model details.) Please notice that to run TensorRT-LLM within our environment, a patch must be applied to the TensorRT-LLM. You may refer to [Environment setup](#env) section for more details.

## Environment setup <a name="env"></a>
Please ensure that the folder structure complies with the requirements outlined in this [README](../README.md). You will need to prepare the following:
- The ```mmdetection3d``` folder (branch `v1.0.0rc6`)
- The ```data``` folder
- The pretrained LLM weights
- The model checkpoints

We recommend starting with the [Dockerfile](./omnidrive-deploy.dockerfile):
```bash
cd ./deploy/
docker build -t omnidrive-deploy:v0 --file ./omnidrive-deploy.dockerfile .
docker run -it --name omnidrive-deploy --gpus all --shm-size=8g -v <workspace>:<workspace> omnidrive-deploy:v0 /bin/bash
```

To setup TensorRT library, you may refer to [TensorRT github](https://github.com/NVIDIA/TensorRT). You should choose the correct wheel according to the python version in your environment. In this demo, the python version inside the docker is 3.8. So you should install the wheel that has `cp38` in its name.

```bash
cd <TensorRT_PATH>/python/
pip3 install ./tensorrt-*-cp38-none-linux_x86_64.whl
```

We also need to build the TensorRT-LLM wheels within the Docker environment. Please note that a [patch]() must be applied to the official TensorRT-LLM repo to ensure compatibility between TensorRT-LLM and the environment needed for the vision component.
Please follow the commands to build and install the TensorRT-LLM wheels in the docker.

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout d335ef9797c6116019ed8191450834e438fdca16
git submodule update --init --recursive
apt update && apt install git-lfs
git lfs install && git lfs pull
# NOTE that since 11.7 is not fully supported by tensorrt-llm officially, we provide a patch for a quick experiment.
# This patch is experimental, and targeting on omnidrive llm inference only. If you need further support please contact NVIDIA.
git apply <Omnidrive_PATH>/deploy/tensorrt_llm.patch

# Please change CUDA_ARCH and TensorRT_PATH according to your own hardware spec
export CUDA_ARCH=<Choose according to your hardware>
export TensorRT_PATH=<path to tensorrt>
python3 ./scripts/build_wheel.py --cuda_architectures=$CUDA_ARCH --trt_root=$TensorRT_PATH -D ENABLE_MULTI_DEVICE=0 --job_count=8 --clean

# Then install tensorrt_llm alone with some other dependencies
pip3 install pynvml==11.5.3
pip3 install ./build/tensorrt_llm_*.whl --force-reinstall --no-deps
pip3 install transformers==4.31.0
```

## Vision engine build <a name="vision"></a>
The vision component of the OmniDrive includes the vision backbone, positional embedding, bounding box detection head and map head. We will export a unified ONNX model for the vision component and build engines based on the ONNX models subsequently.

To export the [OmniDrive](../projects/configs/OmniDrive/eva_base_tinyllama.py) from PyTorch to ONNX
```bash
PYTHONPATH="./":$PYTHONPATH python3 ./deploy/export_vision.py ./projects/configs/OmniDrive/eva_base_tinyllama.py <checkpoint_path>
```
The exported ONNX model for the vision component will be at the `./onnxs/eva_base_tinyllama.onnx`.

> **NOTE**:
> We observe higher numerical sensitivity nature in omnidrive. Using FP16 precision for the whole network may cause significant performance degradation. When using FP16 precision, we recommend that you convert only the backbone as FP16 and keep the remaining operators in FP32. To set the precision for specific parts of the vision network, we mark the operations and generate a seperate ONNX model for FP16 engine building, which has the filename ending with ```_mixed_precision.onnx```.

We can then build the TensorRT engines for the vision component using ```trtexec```. Be sure to use the ONNX file ending with ```_mixed_precision.onnx``` when building engines with FP16 enabled:
```bash
export TRT_HOME=<TensorRT_PATH>
export LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH
# FP32 enigne
${TRT_HOME}/bin/trtexec --onnx=./onnxs/eva_base_tinyllama.onnx --skipInference --saveEngine=./engines/eva_base_tinyllama.engine --useCudaGraph

# FP16 enigne
${TRT_HOME}/bin/trtexec --onnx=./onnxs/eva_base_tinyllama_mixed_precision.onnx --skipInference --saveEngine=./engines/eva_base_tinyllama_mixed_precision.engine --fp16 --precisionConstraints=obey --layerPrecisions=*_FORCEFP32:fp32 --useCudaGraph
```

## LLM engine build <a name="llm"></a>
The LLM head is fine-tuned based on a pretrained Hugging Face [model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), therefore, the engine requires the weights from both the pretrained model and OmniDrive's checkpoint. 
The first step is to update the `llm_path` in the [config](../projects/configs/OmniDrive/eva_base_tinyllama.py) line 25: 
```llm_path = <pretrained_LLM_path>``` and then to combine and save the weights for LLM component.
```bash
PYTHONPATH="./":$PYTHONPATH  python3 ./deploy/save_llm_checkpoint.py --config ./projects/configs/OmniDrive/eva_base_tinyllama.py --checkpoint <checkpoint_path> --llm_checkpoint <pretrained_LLM_path> --save_checkpoint_pth <LLM_checkpoint_path>
```

Next we convert the checkpoint to Hugging Face safetensors format using apis from TensorRT-LLM.
```bash
# FP16 engine
LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH python3 ./deploy/convert_llm_checkpoint.py --model_dir <LLM_checkpoint_path> --output_dir <LLM_safetensor_path>/x86_1gpu_fp16/ --dtype float16

# FP16 activation, INT4 weight (w4a16 weight-only quantization)
LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH python3 ./deploy/convert_llm_checkpoint.py --model_dir <LLM_checkpoint_path> --output_dir <LLM_safetensor_path>/x86_1gpu_w4a16/ --dtype float16 --use_weight_only --weight_only_precision int4
```
After this step, the `.safetensors` file should be generated.

Since OmniDrive utilizes a modified version of TinyLlama, we need to change the `architecture` field in the safetensor's `config.json` from the default value `"architecture": "LlavaLlamaForCausalLM"` to `"architecture": "LlamaForCausalLM"`. We can run `trtllm-build` to build the TensorRT-LLM engine after this modification.

```bash
# FP16 engine and FP16 activation, INT4 weight
LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH trtllm-build --checkpoint_dir <LLM_safetensor_path>/x86_1gpu_<fp16_or_w4a16>/ --output_dir <LLM_engine_path>/x86_1gpu_<fp16_or_w4a16>/ --max_prompt_embedding_table_size 1024 --max_batch_size 1 --max_multimodal_len 2048 --gemm_plugin float16
```
At this point, we should have both the vision engine and the LLM engine ready.

## Benchmark <a name="bench"></a>
We provide a shell script to run the full benchmark for the OmniDrive engine.
```bash
bash ./deploy/dist_test.sh <TensorRT_PATH> <config_path> <tokenizer_path> <vision_engine_path> <LLM_engine_path>/x86_1gpu_<fp16_or_w4a16>/ <QA_save_path>
```
Similar to the PyTorch benchmark, the script will display performance evaluation for detection task, and will save all generated planning trajectories into `<QA_save_path>` folder. Please evaluate the planning results using [`eval_planning.py`](../evaluation/eval_planning.py).
```bash
python3 ./evaluation/eval_planning.py --base_path ./data/nuscenes/ --pred_path <QA_save_path>
```

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
Here is the runtime latency analysis for the engines. The data was collected on an A100. And the unit in the following table is `ms`.

Metrics | PyTorch | FP32 Vision engine | FP16 Vision engine
--------| ------- | ------- | ------
Latency | 280.024 | 75.66   | 25.92

Metrics                      | PyTorch  | FP16 LLM engine | FP16 activation INT4 weight
--------------------------   | -------  | -------- | ---
Time To First Token (TTFT)   | 107.824  |  10.02   | 11.298
Time Per Output Token (TPOT) |   27.49  |  2.798   |  2.515
Time Per Frame               | 2256.50  | 256.20   |  232.6

## Future works <a name="future"></a>
- [ ] Better quantization (accuracy and latency)
- [ ] DriveOS deployment
- [ ] Better execution schedule for lower latency and better hardware utilization

## References <a name="ref"></a>
1. [EVA paper](https://arxiv.org/abs/2303.11331)
2. [StreamPETR paper](https://arxiv.org/abs/2303.11926)
3. [TinyLlama paper](https://arxiv.org/abs/2401.02385)
4. [TensorRT repo](https://github.com/NVIDIA/TensorRT)
5. [TensorRT-LLM repo](https://github.com/NVIDIA/TensorRT-LLM)
