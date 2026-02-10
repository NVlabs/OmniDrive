# OmniDrive Deployment with TensorRT-LLM
In this document, we will provide a step-by-step overview of the deployment process on X86_64 using TensorRT-LLM, including the environment setup, engine build, and engine benchmarking.

## Table of Contents

1. [Environment setup](#env)
2. [Vision engine build](#vision)
3. [LLM engine build](#llm)
4. [Benchmark](#bench)

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
