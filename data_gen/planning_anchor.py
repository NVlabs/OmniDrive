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
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
import tqdm
def k_means_anchors(k: int, future_traj_all: np.ndarray) -> np.ndarray:
    """
    Extracts anchors for multipath/covernet using k-means on train set
    trajectories.
    
    Args:
        k (int): The number of clusters for k-means algorithm.
        future_traj_all (np.ndarray): The array containing all future trajectories.

    Returns:
        np.ndarray: The k anchor trajectories.
    """
    prototype_traj = future_traj_all
    traj_len = prototype_traj.shape[1]
    traj_dim = prototype_traj.shape[2]
    ds_size = future_traj_all.shape[0]
    trajectories = future_traj_all
    clustering = KMeans(n_clusters=k).fit(trajectories.reshape((ds_size, -1)))
    anchors = np.zeros((k, traj_len, traj_dim))
    for i in range(k):
        anchors[i] = np.mean(trajectories[clustering.labels_ == i], axis=0)
    return anchors

data_root = "./data/nuscenes/"
info_prefix = 'train'
key_infos = pickle.load(open(os.path.join(data_root,'nuscenes2d_ego_temporal_infos_{}.pkl'.format(info_prefix)), 'rb'))
step = 6

planning_trajs = []
for current_id in tqdm.tqdm(range(len(key_infos['infos']))):
    if 'gt_planning' in key_infos['infos'][current_id].keys():
        mask = key_infos['infos'][current_id]['gt_planning_mask'][0].any(axis=1)
        traj = key_infos['infos'][current_id]['gt_planning'][0][mask]
        if traj.shape[0] == step:
            planning_trajs.append(traj)
planning_trajs = np.stack(planning_trajs)
planning_anchor = k_means_anchors(4096, planning_trajs)
pickle.dump(planning_anchor, open('planning_anchor_infos.pkl', 'wb'))