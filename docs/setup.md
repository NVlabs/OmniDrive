# Environment Setup

**1. Download nuScenes**

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

**2. Download infos files**
Download the [Info Files](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/data_nusc.zip).

Unzip the data_nusc.tar.gz and move the files into data/nuscenes/.

The pkl info is generated using StreamPETR's data converter. We converted the GT in the lidar coordinate to an ego coordinate system using update_coords.py (to align with OpenLanev2) and added canbus/command information.

**3. Install Packages**
```shell
cd /path/to/OmniDrive
conda create -n omnidrive python=3.9
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install flash-attn==2.5.6
pip install transformers==4.31.0 
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
git clone https://github.com/OpenDriveLab/OpenLane-V2.git
cd OpenLane-V2
pip install -e .
cd ..
pip install -r requirements.txt
```

After preparation, you will be able to see the following directory structure:  

**4. Folder structure**
```
OmniDrive
├── projects/
├── mmdetection3d/
├── OpenLane-V2/
├── tools/
├── configs/
├── ckpts/
│   ├── pretrain_qformer/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── conv/
│   │   ├── desc/
│   │   ├── keywords/
│   │   ├── vqa/
│   │   ├── nuscenes2d_ego_temporal_infos_train.pkl
│   │   ├── nuscenes2d_ego_temporal_infos_val.pkl
│   │   ├── data_dict_sample.pkl
│   │   ├── data_dict_subset_B.json
│   │   ├── data_dict_subset_B_val.pkl
│   │   ├── lane_obj_train.pkl
```

## Pretrained Weights
```shell
cd /path/to/OmniDrive
mkdir ckpts
```
Please download the pretrained [2D llm weights](https://huggingface.co/exiawsh/pretrain_qformer/tree/main) and [vision encoder + projector weights](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/eva02_petr_proj.pth) to ./ckpts.

The [vision encoder + projector weights](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/eva02_petr_proj.pth) are extracted from ckpts/pretrain_qformer/, which is pretrained by using llava data.

The pkl files for counterfactual evaluation: https://github.com/NVlabs/OmniDrive/releases/download/v1.0/eval_cf.zip
