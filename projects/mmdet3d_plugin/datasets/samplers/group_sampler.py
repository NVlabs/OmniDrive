# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import math
import itertools
import copy
import torch.distributed as dist
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from .sampler import SAMPLER
import random


@SAMPLER.register_module()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, num_replicas = get_dist_info()

    if num_replicas == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

@SAMPLER.register_module()
class InfiniteGroupEachSampleInBatchSampler(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.
    Shuffling is only done for group order, not done within groups.
    Arguments:
        dataset: Dataset used for sampling.
        min_len: Minimum sequence sampling length
        max_len: Maximum sequence sampling length
        num_iters_to_seq: After `num_iters_to_seq` iterations, 
            start sequential sampling. Default: 0
        samples_per_gpu (optional): Per gpu batchsize. Default: 1
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self, 
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 seq_split_num=2,
                 warmup_split_num=10,
                 num_iters_to_seq=4000,):

        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.seq_split_num = seq_split_num
        self.warmup_split_num = warmup_split_num
        self.sub_seq_generator = torch.Generator()
        self.sub_seq_generator.manual_seed(self.rank + seed)
        self.seed = sync_random_seed(seed)

        self.size = len(self.dataset)
        self._iters = 0
        self.num_iters_to_seq = num_iters_to_seq
        
        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = samples_per_gpu * num_replicas
        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict {group_idx: List[dataset sample_idxs]}
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)} 

        self.group_idx_to_sample_idxs_generator = {
            group_idx: self._sample_sub_sequence(group_idx)
            for group_idx in range(self.groups_num)
        }

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator 
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) 
            for local_sample_idx in range(self.batch_size)]
        
        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(), 
                                    global_sample_idx, 
                                    None,
                                    self.global_batch_size)

    def _sample_sub_sequence(self, group_idx):
        '''randomly split sub-sequences in a whole sequence'''

        sample_ids = self.group_idx_to_sample_idxs[group_idx]
        while True:
            if self._iters < self.num_iters_to_seq:
                idx = torch.randperm(len(sample_ids), generator=self.sub_seq_generator).tolist()
                idx.remove(0)
                idx = sorted(idx[:self.warmup_split_num]) # choose n-1 split position
                split_idx = [0] + idx + [len(sample_ids)]
                sub_seq_idx = [sample_ids[split_idx[i]: split_idx[i + 1]] 
                            for i in range(len(split_idx) - 1)] # [[1,2,3], [4,5], ...]
                shuffled = torch.randperm(len(sub_seq_idx), generator=self.sub_seq_generator).tolist()
                yield from [sub_seq_idx[i] for i in shuffled]
            
            else:
                # split the sequence into parts
                idx = torch.randperm(len(sample_ids), generator=self.sub_seq_generator).tolist()
                idx.remove(0)
                idx = sorted(idx[:self.seq_split_num - 1]) # choose n-1 split position
                split_idx = [0] + idx + [len(sample_ids)]
                sub_seq_idx = [sample_ids[split_idx[i]: split_idx[i + 1]] 
                            for i in range(len(split_idx) - 1)] # [[1,2,3], [4,5], ...]
                shuffled = torch.randperm(len(sub_seq_idx), generator=self.sub_seq_generator).tolist()
                yield from [sub_seq_idx[i] for i in shuffled]
        

    def __iter__(self):
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = \
                        copy.deepcopy(next(self.group_idx_to_sample_idxs_generator[new_group_idx]))

                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))
            
            self._iters += 1
            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return self.size
        
    def set_epoch(self, epoch):
        self.epoch = epoch