from random import sample
import numpy as np
import torch
from nuscenes.prediction import PredictHelper

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from nuscenes.utils.data_classes import Box
import math
class NuScenesTraj(object):
    def __init__(self,
                 nusc,
                 prediction_steps,
                 planning_steps,
                 CLASSES,
                 box_mode_3d,):
        super().__init__()
        self.nusc = nusc
        self.prediction_steps = prediction_steps
        self.planning_steps = planning_steps
        self.predict_helper = PredictHelper(self.nusc)
        self.CLASSES = CLASSES
        self.box_mode_3d = box_mode_3d
        self.cat2idx = {}
        for idx, dic in enumerate(nusc.category):
            self.cat2idx[dic['name']] = idx
        
    def generate_sdc_info(self, as_lidar_instance3d_box=False):
        # sdc dim from https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        psudo_sdc_bbox = np.array([0.0, 0.0, 0.0, 1.73, 4.08, 1.56, 0.0])

        gt_bboxes_3d = np.array([psudo_sdc_bbox]).astype(np.float32)
        gt_names_3d = ['car']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        if as_lidar_instance3d_box:
            # if we do not want the batch the box in to DataContrainer
            return gt_bboxes_3d

        gt_labels_3d = DC(to_tensor(gt_labels_3d))
        gt_bboxes_3d = DC(gt_bboxes_3d, cpu_only=True)

        return gt_bboxes_3d, gt_labels_3d
    
    def get_planning_label(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        sd_ref = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ref_pose_record = self.nusc.get('ego_pose', sd_ref['ego_pose_token'])
        ref_e2g_rot = Quaternion(ref_pose_record['rotation']).rotation_matrix
        ref_e2g_trans = np.array(ref_pose_record['translation'])
        planning = []
        for _ in range(self.planning_steps):
            next_annotation_token = sample['next']
            if next_annotation_token == "":
                break
            sample = self.nusc.get('sample', next_annotation_token)
            sd_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            rec_pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
            rec_e2g_rot = Quaternion(rec_pose_record['rotation']).rotation_matrix
            rec_e2g_trans = np.array(rec_pose_record['translation'])
            next_bbox3d = self.generate_sdc_info(as_lidar_instance3d_box=True)
            
            next_bbox3d.rotate(rec_e2g_rot.T)
            next_bbox3d.translate(rec_e2g_trans)
            
            next_bbox3d.translate(-ref_e2g_trans)
            next_bbox3d.rotate(ref_e2g_rot)
            planning.append(next_bbox3d)
            
        planning_all = np.zeros((1, self.planning_steps, 3))
        planning_mask_all = np.zeros((1, self.planning_steps, 2))
        n_valid_timestep = len(planning)
        if n_valid_timestep > 0:
            planning = [p.tensor.squeeze(0) for p in planning]
            planning = np.stack(planning, axis=0)  # (valid_t, 9)
            planning = planning[:, [0,1,6]]  # (x, y, yaw)
            planning_all[:,:n_valid_timestep,:] = planning
            planning_mask_all[:,:n_valid_timestep,:] = 1
        
        mask = planning_mask_all[0].any(axis=1)
        if mask.sum() == 0:
            command = 2 #'FORWARD'
        elif planning_all[0, mask][-1][1] >= 2:
            command = 0 #'LEFT' 
        elif planning_all[0, mask][-1][1] <= -2:
            command = 1 #'RIGHT'
        else:
            command = 2 #'FORWARD'
        
        return planning_all, planning_mask_all, command            
            
    def get_traj_label(self, sample_token):

        sample = self.nusc.get('sample', sample_token)
        sd_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        _, boxes, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        annotations = [
                        self.nusc.get('sample_annotation', token)
                        for token in sample['anns']
                    ]

        num_box = len(annotations)
        gt_fut_trajs = np.zeros((num_box, self.prediction_steps, 2))
        gt_fut_yaw = np.zeros((num_box, self.prediction_steps))
        gt_fut_masks = np.zeros((num_box, self.prediction_steps))
        gt_fut_idx = np.zeros((num_box, 1))
        for i, anno in enumerate(annotations):
            gt_fut_idx[i] = self.cat2idx[anno['category_name']] if anno['category_name'] in self.cat2idx.keys() else -1
            cur_box = boxes[i]
            cur_box.rotate(Quaternion(cs_record['rotation']))
            cur_box.translate(np.array(cs_record['translation']))
            cur_anno = anno
            for j in range(self.prediction_steps):
                if cur_anno['next'] != '':
                    anno_next = self.nusc.get('sample_annotation', cur_anno['next'])
                    box_next = Box(
                        anno_next['translation'], anno_next['size'], Quaternion(anno_next['rotation'])
                    )
                    # Move box to ego vehicle coord system.
                    box_next.translate(-np.array(pose_record['translation']))
                    box_next.rotate(Quaternion(pose_record['rotation']).inverse)

                    gt_fut_trajs[i, j] = box_next.center[:2] - cur_box.center[:2]
                    gt_fut_masks[i, j] = 1
                    # add yaw diff
                    box_yaw = cur_box.orientation.yaw_pitch_roll[0]
                    box_yaw_next = box_next.orientation.yaw_pitch_roll[0]
                    gt_fut_yaw[i, j] = box_yaw_next - box_yaw
                    cur_anno = anno_next
                    cur_box = box_next
                else:
                    gt_fut_trajs[i, j:] = 0
                    break	
        return gt_fut_trajs, gt_fut_yaw, gt_fut_masks, gt_fut_idx

def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw