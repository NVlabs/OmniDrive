
# OmniDrive: LLM-Agent for Autonomous Driving with 3D Perception, Reasoning and Planning

<!-- ## Introduction -->

https://github.com/NVlabs/OmniDrive/assets/74858581/f64987a0-b890-416d-90c1-e0daaeb542d6

We present OmniDrive, a holistic Drive LLM-Agent framework for end-to-end autonomous driving. Our main contributions involve novel solutions in both model (OmniDrive-Agent) and benchmark (OmniDrive-nuScenes). The former features a novel 3D multimodal LLM design that uses sparse queries to lift and compress visual representations into 3D. The latter is constituted of comprehensive VQA tasks for reasoning and planning, including scene description, traffic regulation, 3D grounding, counterfactual reasoning, decision making and planning.

## News
- `[2024/05/02]` OmniDrive-nuScenes dataset release. [[Download](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/omnidrive_data.zip)]
- `[2024/05/02]` [ArXiv](https://arxiv.org/abs/2405.01533) technical report release.

## Visual Results

Joint End-to-end Planning and Reasoning

<div align="center">
<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo1.gif" width="1000">

<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo2.gif" width="1000">
</div>
<br><br>

Interactive Conversation with Ego Vehicle

<div align="center">
<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo3.gif" width="1000">
</div>
<br><br>

Counterfactual Reasoning of Planning Behaviors

<div align="center">
<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo4.png" width="1000">

<img src="https://github.com/NVlabs/OmniDrive/releases/download/v1.0/demo5.png" width="1000">
</div>
<br><br>

## Citation
If this work is helpful for your research, please consider citing:

```
@article{wang2024omnidrive,
  title={{OmniDrive}: A Holistic LLM-Agent Framework for Autonomous Driving with 3D Perception, Reasoning and Planning},
  author={Shihao Wang and Zhiding Yu and Xiaohui Jiang and Shiyi Lan and Min Shi and Nadine Chang and Jan Kautz and Ying Li and Jose M. Alvarez},
  journal={arXiv:2405.01533},
  year={2024}
}
```

