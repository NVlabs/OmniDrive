# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
'''
calculate planner metric same as stp3
'''
import numpy as np
import torch
import cv2
import copy
import matplotlib.pyplot as plt
from map_api import NuScenesMap
from skimage.draw import polygon
import math
import pickle
from nuscenes.eval.common.utils import Quaternion
import random
ego_width, ego_length = 1.85, 4.084


class PlanningMetric():
    def __init__(self, base_path, step=6):
        super().__init__()
        self.X_BOUND = [-50.0, 50.0, 0.1]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.1]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        self.step = step
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]

        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()
        self.nusc_maps = {
        'boston-seaport': NuScenesMap(dataroot=base_path, map_name='boston-seaport'),
        'singapore-hollandvillage': NuScenesMap(dataroot=base_path, map_name='singapore-hollandvillage'),
        'singapore-onenorth': NuScenesMap(dataroot=base_path, map_name='singapore-onenorth'),
        'singapore-queenstown': NuScenesMap(dataroot=base_path, map_name='singapore-queenstown'),
    }

        self.W = ego_width
        self.H = ego_length

        # self.category_index = [i for i in range(23)]
        # self.category_index = [0,1,2,3,4,5,6]+ [8,9,10,11,12,13,14,15,16,17]
        self.category_index = [8,9,10,11,12,13,14,15,16,17] # vehicle only

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx
    
    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        """
        Parameters
        ----------
            x_bounds: Forward direction in the ego-car.
            y_bounds: Sides
            z_bounds: Height

        Returns
        -------
            bev_resolution: Bird's-eye view bev_resolution
            bev_start_position Bird's-eye view first element
            bev_dimension Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                    dtype=torch.long)

        return bev_resolution, bev_start_position, bev_dimension
    
    def get_label(
            self,
            gt_agent_boxes,
            gt_agent_feats
        ):
        segmentation_np = self.get_birds_eye_view_label(gt_agent_boxes,gt_agent_feats)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0)

        return segmentation
    
    def get_birds_eye_view_label(
            self,
            gt_agent_boxes,
            gt_agent_feats,
            add_rec=False,
        ):
        '''
        gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
            dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)
        gt_agent_feats: (B, A, 34)
            dim 34 = fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
            lcf_feat (x, y, yaw, vx, vy, width, length, height, type)
        ego_lcf_feats: (B, 9) 
            dim 8 = (vx, vy, ax, ay, w, length, width, vel, steer)
        '''
        T = self.step
        agent_num = gt_agent_feats.shape[0]


        gt_agent_fut_trajs = gt_agent_feats[..., :T*2].reshape(-1, T, 2)
        gt_agent_fut_mask = gt_agent_feats[..., T*2:T*3].reshape(-1, T)
        gt_agent_fut_yaw = gt_agent_feats[..., T*3:T*4].reshape(-1, T, 1)
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]
        if add_rec:
            gt_agent_fut_trajs = np.concatenate([gt_agent_boxes[:, np.newaxis, 0:2], gt_agent_fut_trajs], 1)
            gt_agent_fut_yaw = np.concatenate([gt_agent_boxes[:, np.newaxis, 6:7], gt_agent_fut_yaw], 1)
            gt_agent_fut_mask = np.concatenate([np.ones_like(gt_agent_fut_mask[:, :1]), gt_agent_fut_mask], 1)

        if add_rec:
            T += 1
        segmentation = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))
        
        for t in range(T):
            for i in range(agent_num):
                if gt_agent_fut_mask[i][t] == 1:
                    # Filter out all non vehicle instances
                    category_index = int(gt_agent_feats[i][-1])
                    agent_length, agent_width = gt_agent_boxes[i][4], gt_agent_boxes[i][3]
                    x_a = gt_agent_fut_trajs[i, t, 0]
                    y_a = gt_agent_fut_trajs[i, t, 1]
                    yaw_a = gt_agent_fut_yaw[i, t, 0]
                    param = [x_a,y_a,yaw_a,agent_length, agent_width]
                    if (category_index in self.category_index):
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(segmentation[t], [poly_region], 1)
        # segmentation -= 1
        
        return segmentation

    def get_ego_seg(
            self,
            gt_agent_boxes,
            gt_agent_feats,
            add_rec=False,
        ):
        '''
        gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
            dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)
        gt_agent_feats: (B, A, 34)
            dim 34 = fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
            lcf_feat (x, y, yaw, vx, vy, width, length, height, type)
        ego_lcf_feats: (B, 9) 
            dim 8 = (vx, vy, ax, ay, w, length, width, vel, steer)
        '''
        T = self.step
        agent_num = gt_agent_feats.shape[0]
        gt_agent_fut_trajs = gt_agent_feats[..., :T*2].reshape(-1, T, 2)
        gt_agent_fut_mask = gt_agent_feats[..., T*2:T*3].reshape(-1, T)
        gt_agent_fut_yaw = gt_agent_feats[..., T*3:T*4].reshape(-1, T, 1)

        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]
        if add_rec:
            gt_agent_fut_trajs = np.concatenate([gt_agent_boxes[:, np.newaxis, 0:2], gt_agent_fut_trajs], 1)
            gt_agent_fut_yaw = np.concatenate([gt_agent_boxes[:, np.newaxis, 6:7], gt_agent_fut_yaw], 1)
            gt_agent_fut_mask = np.concatenate([np.ones_like(gt_agent_fut_mask[:, :1]), gt_agent_fut_mask], 1)

        if add_rec:
            T += 1
        segmentation = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))
        
        for t in range(T):
            for i in range(agent_num):
                if gt_agent_fut_mask[i][t] == 1:
                    agent_length, agent_width = gt_agent_boxes[i][4], gt_agent_boxes[i][3]
                    x_a = gt_agent_fut_trajs[i, t, 0]
                    y_a = gt_agent_fut_trajs[i, t, 1]
                    yaw_a = gt_agent_fut_yaw[i, t, 0]
                    param = [x_a,y_a,yaw_a,agent_length, agent_width]
                    poly_region = self._get_poly_region_in_image(param)
                    cv2.fillPoly(segmentation[t], [poly_region], 1.0)
        
        return segmentation

    def _get_poly_region_in_image(self,param):
        lidar2cv_rot = np.array([[1,0], [0,1]])
        x_a,y_a,yaw_a,agent_length, agent_width = param
        trans_a = np.array([[x_a,y_a]]).T
        # rot_mat_a = np.array([[np.cos(yaw_a), -np.sin(yaw_a)],
        #                         [np.sin(yaw_a), np.cos(yaw_a)]])
        rot_mat_a = np.array([[-np.sin(yaw_a), np.cos(yaw_a)],
                            [np.cos(yaw_a), np.sin(yaw_a)]])
        agent_corner = np.array([
            [agent_length/2, -agent_length/2, -agent_length/2, agent_length/2],
            [agent_width/2, agent_width/2, -agent_width/2, -agent_width/2]]) #(2,4)
        agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a #(2,4)
        # convert to cv frame
        agent_corner_cv2 = (np.matmul(lidar2cv_rot, agent_corner_lidar) \
            - self.bev_start_position[:2,None] + self.bev_resolution[:2,None] / 2.0).T / self.bev_resolution[:2] #(4,2)
        agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)

        return agent_corner_cv2

    def traj_check(self, ego_seg, bev_seg, light_seg, drivable_seg, start_step=1, end_step=7):
        coll_index = (ego_seg[start_step:end_step] == 1.0) & (bev_seg != -1)
        coll_index = np.unique(bev_seg[coll_index]).astype(np.int64)
        right_light = ((np.expand_dims(light_seg, 0) == 1) & (ego_seg == 0)).sum() > 0
        out_of_drivable = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg == 1)).sum() > 0
        
        return coll_index, right_light, out_of_drivable

    def red_light_area(self, curves):
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        res = self.bev_resolution[:2]
        dim = self.bev_dimension[:2]

        for curve in curves:
            left_lane, right_lane = expand_lane(curve[..., :2], width=2, res=res, dim=dim)
            cv2.fillPoly(segmentation, [np.vstack((left_lane, right_lane[::-1]))], 1.0)

        return segmentation        

    def compute_L2(self, trajs, gt_trajs):
        '''
        trajs: torch.Tensor (n_future, 2)
        gt_trajs: torch.Tensor (n_future, 2)
        '''
        pred_len = trajs.shape[0]
        ade = float(
            sum(
                torch.sqrt(
                    (trajs[i, 0] - gt_trajs[i, 0]) ** 2
                    + (trajs[i, 1] - gt_trajs[i, 1]) ** 2
                )
                for i in range(pred_len)
            )
            / pred_len
        )
        
        return ade

    def get_drivable_area(self, e2g_t, e2g_r_mat, data):
        location = data['location']
        nusc_map = self.nusc_maps[location]
        patch_size=(100, 100)
        canvas_size=(self.bev_dimension[0], self.bev_dimension[1])
        patch_box = (e2g_t[0], e2g_t[1], patch_size[0], patch_size[1])
        patch_angle = math.degrees(Quaternion(matrix=e2g_r_mat).yaw_pitch_roll[0])
        drivable_area = nusc_map.get_map_mask(patch_box, patch_angle, ['drivable_area'], canvas_size=canvas_size)
        drivable_area = drivable_area.squeeze(0)

        return drivable_area

    def evaluate_single_coll(self, traj, segmentation, input_gt=None, gt_traj=None, index=None):
        '''
        traj: torch.Tensor (n_future, 2)
            自车IMU系为轨迹参考系

                0------->
                |        x
                |
                |y
                
        segmentation: torch.Tensor (n_future, 200, 200)
        '''
        # 0.985793 is the distance betweem the LiDAR and the IMU(ego).

        import mmcv
        pts = np.array([
            [-self.H / 2. + 0.5 + 0.985793, self.W / 2.],
            [self.H / 2. + 0.5 + 0.985793, self.W / 2.],
            [self.H / 2. + 0.5 + 0.985793, -self.W / 2.],
            [-self.H / 2. + 0.5 + 0.985793, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy() ) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)
        rc_ori = rc + (self.bx.cpu().numpy() / self.dx.cpu().numpy())


        traj_with_ego = torch.cat([traj.new_zeros(1, 2), traj], 0)
        rc_yaw = []
        rotate_angle = 0
        for i in range(traj.size(0)):
            delta = traj_with_ego[i+1] - traj_with_ego[i]
            cur_rotate_angle = torch.atan2(*delta[[1, 0]])
            if delta.norm()<1: cur_rotate_angle = 0
            rotate_angle = cur_rotate_angle
            rotate_angle = -torch.tensor(rotate_angle)
            rot_sin = torch.sin(rotate_angle)
            rot_cos = torch.cos(rotate_angle)
            rot_mat = torch.Tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
            tmp = rc_ori @ rot_mat.cpu().numpy() -  (self.bx.cpu().numpy() / self.dx.cpu().numpy())
            tmp = tmp.round().astype(np.int64)
            rc_yaw.append(tmp)
        rc_yaw = np.stack(rc_yaw)


        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)

        trajs_ = copy.deepcopy(trajs)
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc_yaw # (n_future, 32, 2)

        r = trajs_[:,:,0].astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:,:,1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision2 = np.full(n_future, False)
        # obs_occ = copy.deepcopy(segmentation).cpu().numpy() * 0
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision2[t] = np.any(segmentation[t,  cc[I], rr[I]].cpu().numpy())

        return torch.from_numpy(collision2).to(device=traj.device)

    def evaluate_coll(
            self, 
            trajs, 
            gt_trajs, 
            segmentation,
            index=None,
            ignore_gt=False,
        ):
        '''
        trajs: torch.Tensor (B, n_future, 2)
        自车IMU系为轨迹参考系

                0------->
                |        x
                |
                |y
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)

        '''
        B, n_future, _ = trajs.shape
        # trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        # gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i], input_gt=True)

            xx, yy = trajs[i,:,0], trajs[i, :, 1]

            xi = ((-self.bx[0] + xx) / self.dx[0]).long()
            yi = ((-self.bx[1] + yy) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            ).to(gt_box_coll.device)
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future)
            # segmentation: B, T, H, W
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()

            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(trajs[i],
                    segmentation[i],
                    gt_traj=gt_trajs[i],
                    input_gt=False,
                    index=None,
                    ).to(segmentation.device)
            if ignore_gt:
                obj_box_coll_sum += (gt_box_coll).long()                
            else:
                obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()
        return obj_coll_sum, obj_box_coll_sum
    