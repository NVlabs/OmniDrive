# OmniDrive Deployment with TensorRT-Edge-LLM
In this document, we will provide a step-by-step overview of the deployment process on X86_64 and NVIDIA DRIVE AGX Thor using TensorRT-Edge-LLM, including the environment setup, engine build, and engine benchmarking.

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

### X86_64 environment setup
To setup the environment for LLM ONNX export. please refer to the [official installation guide](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/developer_guide/01.3_Installation.md) to install `tensorrt-edge-llm` Python package. Please use any environments that can work with `torch>=2.9` for the LLM ONNX export step.

For vision ONNX export, vision and LLM engine build and model benchmarking, we recommend using the [Dockerfile](./omnidrive-deploy.dockerfile):
```bash
cd ./deploy/
docker build -t omnidrive-deploy:v0 --file ./omnidrive-deploy.dockerfile .
docker run -it --name omnidrive-deploy --gpus all --shm-size=8g -v <workspace>:<workspace> omnidrive-deploy:v0 /bin/bash
```

To setup TensorRT library within the container, you may refer to [TensorRT github](https://github.com/NVIDIA/TensorRT). You should choose the correct wheel according to the python version and the platform. In this demo, the python version inside the docker is 3.8. So you should install the wheel that has `cp38` in its name.

```bash
cd <TensorRT_PATH>/python/
pip3 install ./tensorrt-*-cp38-none-linux_x86_64.whl
```

Within the container, pLease follow the official installation guide to build the TensorRT-Edge-LLM project and compile the plugins. Notice that since `cuda11.7` does not support compilation for `SM89`, we need to remove that from the [CMakeList.txt](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/CMakeLists.txt#L52):
```bash
if(NOT DEFINED AARCH64_BUILD)
  # set(CMAKE_CUDA_ARCHITECTURES 80;86;89)
  set(CMAKE_CUDA_ARCHITECTURES 80;86)
  if(CUDA_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100 120)
  endif()
endif()
```
### NVIDIA DRIVE AGX Thor environment setup
On NVIDIA DRIVE AGX Thor board, simply following the official installation guide to build the TensorRT-Edge-LLM project, compile the plugin, and install TensorRT Python package is enough to secure the deploy environment.

## Vision engine build <a name="vision"></a>
The vision component of the OmniDrive includes the vision backbone, positional embedding, bounding box detection head and map head. We will export a unified ONNX model for the vision component and build engines based on the ONNX models subsequently.

To export the [OmniDrive](../projects/configs/OmniDrive/eva_base_tinyllama.py) from PyTorch to ONNX
```bash
# in the container
PYTHONPATH="./":$PYTHONPATH python3 ./deploy/export_vision.py ./projects/configs/OmniDrive/eva_base_tinyllama.py <checkpoint_path>
```
The exported ONNX model for the vision component will be at the `./onnxs/eva_base_tinyllama.onnx`.

> **NOTE**:
> We observe higher numerical sensitivity nature in omnidrive. Using FP16 precision for the whole network may cause significant performance degradation. When using FP16 precision, we recommend that you convert only the backbone as FP16 and keep the remaining operators in FP32. To set the precision for specific parts of the vision network, we mark the operations and generate a seperate ONNX model for FP16 engine building, which has the filename ending with ```_mixed_precision.onnx```.

We can then build the TensorRT engines for the vision component using ```trtexec```. Be sure to use the ONNX file ending with ```_mixed_precision.onnx``` when building engines with FP16 enabled:
```bash
# in the container
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
# in the container
PYTHONPATH="./":$PYTHONPATH  python3 ./deploy/save_llm_checkpoint.py --config ./projects/configs/OmniDrive/eva_base_tinyllama.py --checkpoint <checkpoint_path> --llm_checkpoint <pretrained_LLM_path> --save_checkpoint_pth <LLM_checkpoint_path>
```

Then, we need to modify the `model architecture` field in the `config.json` file from `llava_llama` to `llama`, and run the model export command:
```bash
# in an environment with torch>=2.9
tensorrt-edgellm-export-llm --model_dir <LLM_checkpoint_path> --output_dir <ONNX_save_path>
```
To apply quantization on the LLM model, use the `tensorrt-edgellm-quantize-llm` first:
```bash
# in an environment with torch>=2.9
tensorrt-edgellm-quantize-llm --model_dir <LLM_checkpoint_path> --quantization <dtype> --output_dir <QUANT_LLM_checkpoint_path>
tensorrt-edgellm-export-llm --model_dir <QUANT_LLM_checkpoint_path> --output_dir <QUANT_ONNX_save_path>
```

To build the LLM engine and to run the inference, we will need the plugins compilied by TensorRT-Edge-LLM, please find `libNvInfer_edgellm_plugin.so` under `build` folder. Please refer to the [Environment setup](#env) section to build and compile the project if not already.

The LLM engine will be automatically built on the first time you run the benchmark script.

## Benchmark <a name="bench"></a>
We provide a shell script to run the full benchmark with TensorRT-Edge-LLM for the OmniDrive engine on X86_64, and it will also generate dumped inputs for NVIDIA DRIVE AGX Thor inference.
```bash
# in the container
bash ./deploy/test_edgellm.sh <TensorRT_PATH> <config_path> <LLM_checkpoint_path> <vision_engine_path> <LLM_engine_path> <LLM_onnx_path> <plugin_path> <QA_save_path> <num_dumped_input> <dumped_input_path>
```
Evaluate the planning results using [`eval_planning.py`](../evaluation/eval_planning.py).
```bash
# in the container
python3 ./evaluation/eval_planning.py --base_path ./data/nuscenes/ --pred_path <QA_save_path>
```
On NVIDIA DRIVE AGX Thor, use the dumped input data to run engine inference:
```bash
# On NVIDIA DRIVE AGX Thor
LD_LIBRARY_PATH=<TensorRT_PATH>/lib/:$LD_LIBRARY_PATH python3 ./deploy/test_edgellm_agx.py --vision_engine_pth <vision_engine_path> --llm_engine_pth <LLM_engine_path> --llm_onnx_path <LLM_onnx_path> --plugin_path <plugin_path> --load_input_path <dumped_input_path> --save_output_path <dumped_output_path>
```
Note that on NVIDIA DRIVE AGX Thor we cannot use the engines and plugin built on X86_64, they need to be explicitly built on NVIDIA DRIVE AGX Thor.

To verify the generated output ids, run the script to compare with the output ids generated on X86_64:
```bash
# in the container
python3 ./deploy/decode_dumped_outputIds.py --tokenizer_pth <tokenizer_path> --dumped_output_ids_path <dumped_output_path>
```
