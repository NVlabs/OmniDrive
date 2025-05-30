# OmniDrive: LLM-Agent for Autonomous Driving with 3D Perception, Reasoning and Planning

<!-- ## Introduction -->

https://github.com/NVlabs/OmniDrive/assets/74858581/f64987a0-b890-416d-90c1-e0daaeb542d6

We present OmniDrive, a holistic Drive LLM-Agent framework for end-to-end autonomous driving. Our main contributions involve novel solutions in both model (OmniDrive-Agent) and benchmark (OmniDrive-nuScenes). The former features a novel 3D multimodal LLM design that uses sparse queries to lift and compress visual representations into 3D. The latter is constituted of comprehensive VQA tasks for reasoning and planning, including scene description, traffic regulation, 3D grounding, counterfactual reasoning, decision making and planning.

<div align="center">
<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/teaser.png" width="1000">
</div>

## News
- `[2025/04/16]` Adding TensorRT support. [[Link](https://github.com/NVlabs/OmniDrive/blob/main/deploy/deploy.md)]
- `[2025/02/26]` OmniDrive is accepted to CVPR 2025.
- `[2024/07/18]` OmniDrive-nuScenes model release. [[HF](https://huggingface.co/exiawsh/OmniDrive/tree/main)]
- `[2024/05/02]` OmniDrive-nuScenes dataset release. [[Data](https://github.com/NVlabs/OmniDrive/releases/tag/v1.0)]
- `[2024/05/02]` Technical report release. [[arXiv](https://arxiv.org/abs/2405.01533)]

## Getting Started

Please follow [**Environment Setup**](./docs/setup.md) step by step.

## Currently Supported Features
- [x] OmnDrive Training Framework
- [x] OmnDrive Dataset
- [x] OmnDrive Checkpoint
- [x] Evaluation
- [x] Data Generation
- [x] TensorRT Inference
- [x] Tiny LLM

## Visual Results

Joint End-to-end Planning and Reasoning

<div align="center">
<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo1.gif" width="1000">

<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo2.gif" width="1000">
</div>
<br>

Interactive Conversation with Ego Vehicle

<div align="center">
<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo3.gif" width="1000">
</div>
<br>

Counterfactual Reasoning of Planning Behaviors

<div align="center">
<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo4.png" width="1000">

<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo5.png" width="1000">
</div>

## Citation
If this work is helpful for your research, please consider citing:

```
@inproceedings{wang2025omnidrive,
  title={{OmniDrive}: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning},
  author={Shihao Wang and Zhiding Yu and Xiaohui Jiang and Shiyi Lan and Min Shi and Nadine Chang and Jan Kautz and Ying Li and Jose M. Alvarez},
  booktitle={CVPR},
  year={2025}
}
```

## Acknowledgement
The team would like to give special thanks to the NVIDIA TSE Team, including Le An, Chengzhe Xu, Yuchao Jin, and Josh Park, for their exceptional work on the TensorRT deployment of OmniDrive.
