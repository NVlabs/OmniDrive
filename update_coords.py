# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import os
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import tqdm
from tools.data_converter import nuscenes_converter as nuscenes_converter
from mmdet3d.datasets import NuScenesDataset
import mmcv
from data_utils.trajectory_api import NuScenesTraj
from data_utils.nuscmap_extractor import NuscMapExtractor
from mmdet3d.core.bbox import Box3DMode
from openlanev2.centerline.io import io
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
roi_size=(100, 50) # meters
cat2id_map={
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}

lane_json_path = './data/nuscenes/data_dict_subset_B.json'

can_bus_root_path = "./data/nuscenes/"
data_root = "./data/nuscenes/"
info_prefix = 'val'
key_infos = pickle.load(open(os.path.join(data_root,'nuscenes2d_temporal_infos_{}.pkl'.format(info_prefix)), 'rb'))
info_path = os.path.join(data_root,'nuscenes2d_ego_temporal_infos_{}.pkl'.format(info_prefix))

nuscenes_version = 'v1.0-trainval'
nuscenes = NuScenes(nuscenes_version, data_root)
nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
traj_api = NuScenesTraj(nuscenes, prediction_steps=12, planning_steps=6, CLASSES=CLASSES, box_mode_3d=Box3DMode.LIDAR)
map_api = NuscMapExtractor(data_root, roi_size)

def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(13)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    rotation = last_pose.pop('orientation')
    pos = last_pose.pop('pos')
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 13 elements
    return np.array(can_bus)

sample_dict = {}

for split, segments in io.json_load(lane_json_path).items():
        for segment_id, timestamps in segments.items():
            for timestamp in timestamps:
                sample_dict[timestamp.split(sep='.')[0]] = (split, segment_id, timestamp.split(sep='.')[0])
                
for current_id in tqdm.tqdm(range(len(key_infos['infos']))):
    sample = nuscenes.get('sample', key_infos['infos'][current_id]['token']) 
    info = key_infos['infos'][current_id]
    ego2global_rotation = info['ego2global_rotation']
    ego2global_translation = info['ego2global_translation']
    
    can_bus = _get_can_bus_info(nuscenes, nusc_can_bus, sample)
    
    key_infos['infos'][current_id]["can_bus"] = can_bus
    #openlane
    if str(info['cams']['CAM_FRONT']['timestamp']) not in sample_dict.keys():
        continue
    else:
        info['lane_info'] = sample_dict[str(info['cams']['CAM_FRONT']['timestamp'])]

    # motion prediction
    gt_fut_traj, gt_fut_yaw, gt_fut_traj_mask, gt_fut_idx = traj_api.get_traj_label(info['token'])
    info['gt_fut_traj'] = gt_fut_traj
    info['gt_fut_yaw'] = gt_fut_yaw
    info['gt_fut_traj_mask'] = gt_fut_traj_mask
    info['gt_fut_idx'] = gt_fut_idx

    # planning
    planning_all, planning_mask_all, command = traj_api.get_planning_label(info['token'])
    info['gt_planning'] = planning_all
    info['gt_planning_mask'] = planning_mask_all
    info['gt_planning_command'] = command
      
    # map
    scene_record = nuscenes.get('scene', sample['scene_token'])
    log_record = nuscenes.get('log', scene_record['log_token'])
    location = log_record['location']
    scene_name = scene_record['name']
    info['description'] = scene_record['description']
    info['location'] = location
    info['scene_name'] = scene_name
    map_geoms = map_api.get_map_geom(location, info['ego2global_translation'], info['ego2global_rotation'])

    map_label2geom = {}
    for k, v in map_geoms.items():
        if k in cat2id_map.keys():
            map_label2geom[cat2id_map[k]] = v
    info['map_geoms'] = map_label2geom

    # detection, use ego coordinate
    ann_infos = list()
    for ann in sample['anns']:
        ann_info = nuscenes.get('sample_annotation', ann)
        velocity = nuscenes.box_velocity(ann_info['token'])
        if np.any(np.isnan(velocity)):
            velocity = np.zeros(3)
        ann_info['velocity'] = velocity
        if len(ann_info['attribute_tokens']) == 0:
            ann_info['attr'] = ''
        else:
            ann_info['attr'] = nuscenes.get('attribute', ann_info['attribute_tokens'][0])['name']
        ann_infos.append(ann_info)

    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_velos = list()
    gt_names = list()
    gt_fullnames = list()
    num_lidar_pts = list()
    num_radar_pts = list()
    gt_valid_flags = list()
    gt_dict = dict()
    gt_attrs= list()
    for ann_info in ann_infos:
        # Use ego coordinate.
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw])

        full_name = ann_info['category_name']
        if full_name in NuScenesDataset.NameMapping:
            name = NuScenesDataset.NameMapping[full_name]

        valid_flag = ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] > 0

        gt_valid_flags.append(valid_flag)
        gt_names.append(name)
        gt_fullnames.append(full_name)
        gt_boxes.append(gt_box)
        gt_velos.append(box_velo)
        gt_attrs.append(ann_info['attr'])
        num_lidar_pts.append(ann_info['num_lidar_pts'])
        num_radar_pts.append(ann_info['num_radar_pts'])
    if len(gt_boxes) > 0:
        gt_valid_flags = np.array(gt_valid_flags, dtype=bool).reshape(-1)
        gt_boxes = np.concatenate(gt_boxes, axis=0).reshape(-1, 7)
        gt_velos = np.concatenate(gt_velos, axis=0).reshape(-1, 2)
        gt_names = np.array(gt_names)
        num_lidar_pts = np.array(num_lidar_pts)
        num_radar_pts = np.array(num_radar_pts)
    else:
        gt_valid_flags = np.array(gt_valid_flags, dtype=bool).reshape(-1)
        gt_boxes = np.array(gt_boxes).reshape(-1, 7)
        gt_velos = np.array(gt_velos).reshape(-1, 2)
        gt_names = np.array(gt_names)
        num_lidar_pts = np.array(num_lidar_pts)
        num_radar_pts = np.array(num_radar_pts)

    info['gt_boxes'] = gt_boxes
    info['gt_names'] = gt_names
    info['gt_velocity'] = gt_velos
    info['num_lidar_pts'] = num_lidar_pts
    info['num_radar_pts'] = num_radar_pts
    info['valid_flag'] = gt_valid_flags
    info['gt_fullnames'] = gt_fullnames
    info['gt_attrs'] = gt_attrs

    
    key_infos['infos'][current_id] = info

mmcv.dump(key_infos, info_path)
print(info_path)
