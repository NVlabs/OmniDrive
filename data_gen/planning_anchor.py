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