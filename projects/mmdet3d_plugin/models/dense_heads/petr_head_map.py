# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from math import factorial
from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb1d, pos2posemb3d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points_lane, memory_refresh, SELayer_Linear
import numpy as np
from mmcv.cnn import xavier_init

@HEADS.register_module()
class PETRHeadM(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_lane (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 out_dims=4096,
                 embed_dims=256,
                 num_lane=100,
                 memory_len=1000,
                 topk_proposals=500,
                 num_lanes_one2one=0,
                 k_one2many=0,
                 lambda_one2many=1.0,
                 num_extra=256,
                 with_ego_pos=True,
                 with_mask=False,
                 pc_range=None,
                 num_reg_fcs=2,
                 n_control=4,
                 num_pts_vector=20, #16
                 dir_interval=1,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 match_costs=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights
            
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is PETRHeadM):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.with_ego_pos = with_ego_pos
        self.with_mask = with_mask
        self.output_dims = out_dims
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.dir_interval = dir_interval
        self.num_pts_vector = num_pts_vector
        self.n_control = n_control
        self.num_lane = num_lane
        self.num_lanes_one2one = num_lanes_one2one
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_extra = num_extra 

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        super(PETRHeadM, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.pc_range = nn.Parameter(torch.tensor(
            pc_range), requires_grad=False)

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.n_control*3))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.input_projection = nn.Linear(self.in_channels, self.embed_dims)
        if self.output_dims is not None:
            self.output_projection = nn.Linear(self.embed_dims, self.output_dims)

        self.reference_points_lane = nn.Linear(self.embed_dims, 3)
        
        self.points_embedding_lane = nn.Embedding(self.n_control, self.embed_dims)
        self.instance_embedding_lane = nn.Embedding(self.num_lane, self.embed_dims)
        
        self.query_embedding = nn.Embedding(self.num_extra, self.embed_dims)
        self.query_pos = None

        self.time_embedding = None

        self.ego_pose_pe = None

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        xavier_init(self.reference_points_lane, distribution='uniform', bias=0.)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)


    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.sample_time = None
        self.memory_mask = None

    def pre_update_memory(self, data):
        B = data['img_feats'].size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = data['img_feats'].new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = data['img_feats'].new_zeros(B, self.memory_len, self.n_control, 3)
            self.memory_timestamp = data['img_feats'].new_zeros(B, self.memory_len, 1)
            self.memory_egopose = data['img_feats'].new_zeros(B, self.memory_len, 4, 4)
            self.sample_time = data['timestamp'].new_zeros(B)
            self.memory_mask = data['img_feats'].new_zeros(B, self.memory_len, 1)
            x = self.sample_time.to(data['img_feats'].dtype)
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.sample_time += data['timestamp']
            x = (torch.abs(self.sample_time) < 2.0).to(data['img_feats'].dtype)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points_lane(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_mask = memory_refresh(self.memory_mask[:, :self.memory_len], x)
            self.sample_time = data['timestamp'].new_zeros(B)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec):
        rec_reference_points = all_bbox_preds[-1].reshape(outs_dec.shape[1], -1, self.n_control, 3)
        out_memory = outs_dec[-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_mask = torch.cat([torch.ones_like(rec_timestamp), self.memory_mask], dim=1)
        self.memory_reference_point = transform_reference_points_lane(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.sample_time -= data['timestamp']
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
        
        return out_memory
    
    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_pos(nerf_positional_encoding(temp_reference_point.flatten(-2))) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(tgt[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(tgt[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())
            
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is PETRHeadM:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    

    def forward(self, img_metas, pos_embed, **data):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_lane, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_lane, 9].
        """
        self.pre_update_memory(data)

        x = data['img_feats']
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)

        memory = self.input_projection(memory)

        #1800, 256; 11, 256
        lane_embedding = self.instance_embedding_lane.weight.unsqueeze(-2) + self.points_embedding_lane.weight.unsqueeze(0) 
        reference_points_lane = self.reference_points_lane(lane_embedding).sigmoid().flatten(-2).unsqueeze(0).repeat(B, 1, 1)
        query_pos = self.query_pos(nerf_positional_encoding(reference_points_lane))
        tgt = self.instance_embedding_lane.weight.unsqueeze(0).repeat(B, 1, 1)
        query_embedding = self.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)

        # attn mask for Hy
        self_attn_mask = (
            torch.zeros([self.num_lane+self.num_extra, self.num_lane+self.num_extra]).bool().to(x.device)
        )
        self_attn_mask[self.num_lanes_one2one+self.num_extra:, 0: self.num_lanes_one2one+self.num_extra] = True
        self_attn_mask[0: self.num_lanes_one2one+self.num_extra, self.num_lanes_one2one+self.num_extra:] = True
        temporal_attn_mask = (
            torch.zeros([self.num_lane+self.num_extra, self.num_lane+self.num_extra+self.memory_len]).bool().to(x.device)
        )
        temporal_attn_mask[:self_attn_mask.size(0), :self_attn_mask.size(1)] = self_attn_mask
        if self.with_mask:
            temporal_attn_mask[self.num_extra:, :self.num_extra] = True

        tgt, query_pos, reference_points_lane, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points_lane)

        tgt = torch.cat([query_embedding, tgt], dim=1)
        query_pos = torch.cat([torch.zeros_like(query_embedding), query_pos], dim=1)
        
        outs_dec = self.transformer(tgt, memory, query_pos, pos_embed, temporal_attn_mask, temp_memory, temp_pos)

        vlm_memory = outs_dec[-1, :, :self.num_extra, :]
        outs_dec = outs_dec[:, :, self.num_extra:, :]
        
        outs_dec = torch.nan_to_num(outs_dec)
        
        lane_queries = outs_dec
        outputs_lane_preds = []
        outputs_lane_clses = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points_lane.clone())
            reference = reference.view(B, self.num_lane, self.n_control*3)
            tmp = self.reg_branches[lvl](lane_queries[lvl])
            outputs_lanecls = self.cls_branches[lvl](lane_queries[lvl])

            tmp = tmp.reshape(B, self.num_lane, self.n_control*3)
            tmp += reference
            tmp = tmp.sigmoid()

            outputs_coord = tmp
            outputs_coord = outputs_coord.reshape(B, self.num_lane, self.n_control, 3)
            outputs_lane_preds.append(outputs_coord)
            outputs_lane_clses.append(outputs_lanecls)

        all_lane_preds = torch.stack(outputs_lane_preds)  # torch.Size([6, 1, 600, 33])
        all_lane_clses = torch.stack(outputs_lane_clses)  # torch.Size([6, 1, 600, 1])

        all_lane_preds[..., 0:3] = (all_lane_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        all_lane_preds = all_lane_preds.flatten(-2)

        all_lane_cls_one2one = all_lane_clses[:, :, 0: self.num_lanes_one2one, :]
        all_lane_preds_one2one = all_lane_preds[:, :, 0: self.num_lanes_one2one, :]
        all_lane_cls_one2many = all_lane_clses[:, :, self.num_lanes_one2one:, :]
        all_lane_preds_one2many = all_lane_preds[:, :, self.num_lanes_one2one:, :]
        outs_dec_one2one = outs_dec[:, :, 0: self.num_lanes_one2one, :]
        outs_dec_one2many = outs_dec[:, :, self.num_lanes_one2one:, :]
        out_memory = self.post_update_memory(data, rec_ego_pose, all_lane_cls_one2one, all_lane_preds_one2one, outs_dec_one2one)
        outs = {
            'all_lane_cls_one2one': all_lane_cls_one2one,
            'all_lane_preds_one2one': all_lane_preds_one2one,
            'all_lane_cls_one2many': all_lane_cls_one2many,
            'all_lane_preds_one2many': all_lane_preds_one2many,
            'outs_dec_one2one': outs_dec_one2one,
            'outs_dec_one2many':outs_dec_one2many,
        }
        if self.output_dims is not None:
            vlm_memory = self.output_projection(vlm_memory)
        return outs, vlm_memory


    def loss(self,
             gt_lanes,
             preds_dicts,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_lane, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_lane, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_lane_cls_one2one']
        all_bbox_preds = preds_dicts['all_lane_preds_one2one']
        all_cls_scores_one2many_list = preds_dicts['all_lane_cls_one2many']
        all_bbox_preds_one2many_list = preds_dicts['all_lane_preds_one2many']
        gt_lanes = [lane.reshape(-1, self.n_control*3) for lane in gt_lanes]
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_lanes for _ in range(num_dec_layers)]
        gt_labels = [torch.zeros(gt_lane.shape[0], dtype=torch.long, device=gt_lanes[0].device) for gt_lane in gt_lanes]
        all_gt_labels_list = [gt_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        
        # for one2many
        one2many_gt_bboxes_list = []
        one2many_gt_labels_list = []
        
        for gt_bboxes in gt_lanes:
            one2many_gt_bboxes_list.append(gt_bboxes.repeat(self.k_one2many, 1))
        for gt_labels in gt_labels:
            one2many_gt_labels_list.append(gt_labels.repeat(self.k_one2many))
        all_gt_bboxes_list_one2many = [one2many_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list_one2many = [one2many_gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)
        
        img_metas_list_one2many = img_metas_list
        all_gt_bboxes_ignore_list_one2many = all_gt_bboxes_ignore_list
        losses_cls_one2many, losses_bbox_one2many, losses_dir_one2many = multi_apply(
            self.loss_single, all_cls_scores_one2many_list, all_bbox_preds_one2many_list,
            all_gt_bboxes_list_one2many, all_gt_labels_list_one2many, img_metas_list_one2many,
            all_gt_bboxes_ignore_list_one2many)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls_lane'] = losses_cls[-1]
        loss_dict['loss_cls_H'] = losses_cls_one2many[-1] * self.lambda_one2many
        loss_dict['loss_bbox_lane'] = losses_bbox[-1]
        loss_dict['loss_bbox_H'] = losses_bbox_one2many[-1] * self.lambda_one2many
        # loss_dict['loss_dir'] = losses_dir[-1]
        # loss_dict['loss_dir_H'] = losses_dir_one2many[-1] * self.lambda_one2many
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_dir_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_lane'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_cls_H'] = losses_cls_one2many[num_dec_layer] * self.lambda_one2many
            loss_dict[f'd{num_dec_layer}.loss_bbox_lane'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_H'] = losses_bbox_one2many[num_dec_layer] * self.lambda_one2many
            # loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            # loss_dict[f'd{num_dec_layer}.loss_dir_H'] = losses_dir_one2many[num_dec_layer] * self.lambda_one2many
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()


        dir_weights = bbox_weights.reshape(-1, self.n_control, 3)[:, :-1,0]
        pts_preds_dir = bbox_preds.reshape(-1, self.n_control, 3)[:,1:,:] - bbox_preds.reshape(-1, self.n_control, 3)[:,:-1,:]
        pts_targets_dir = bbox_targets.reshape(-1, self.n_control, 3)[:, 1:,:] - bbox_targets.reshape(-1, self.n_control, 3)[:,:-1,:]
        loss_dir = self.loss_dir(
            pts_preds_dir, pts_targets_dir,
            dir_weights,
            avg_factor=num_total_pos)
        bbox_preds = bbox_preds.reshape(-1, self.n_control * 3)
        
        # bbox_preds = self.control_points_to_lane_points(bbox_preds)
        # bbox_targets = self.control_points_to_lane_points(bbox_targets)
        bbox_weights = bbox_weights.mean(-1).unsqueeze(-1).repeat(1, bbox_preds.shape[-1])
        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        

        
        return loss_cls, loss_bbox, loss_dir


    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
    
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_preds = bbox_pred.size(0)
        # import pdb;pdb.set_trace()
        # bbox_pred = self.control_points_to_lane_points(bbox_pred)
        # gt_bboxes = self.control_points_to_lane_points(gt_bboxes)
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_preds,),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_preds)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        
        if sampling_result.num_gts > 0:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            bbox_weights[pos_inds] = 1.0
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights,
                bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        cls_scores = preds_dicts['all_lane_cls_one2one'][-1]
        bbox_preds = preds_dicts['all_lane_preds_one2one'][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            boxes, scores = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append([boxes, scores])
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):

        assert len(cls_score) == len(bbox_pred)
        cls_score = cls_score.sigmoid()
        det_bboxes = bbox_pred
        for p in range(self.n_control):
            det_bboxes[..., 3 * p].clamp_(min=self.pc_range[0], max=self.pc_range[3])
            det_bboxes[..., 3 * p + 1].clamp_(min=self.pc_range[1], max=self.pc_range[4])
            
        # det_bboxes = self.control_points_to_lane_points(det_bboxes)
        det_bboxes = det_bboxes.reshape(det_bboxes.shape[0], -1, 3)

        return det_bboxes.cpu().numpy(), cls_score.cpu().numpy()

    def onnx_export(self, **kwargs):
        raise NotImplementedError(f'TODO: replace 4 with self.n_control : {self.n_control}')

    def control_points_to_lane_points(self, lanes):
            if lanes.shape[-1] == 0:
                return lanes.reshape(-1, 33)
            lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)

            def comb(n, k):
                return factorial(n) // (factorial(k) * factorial(n - k))

            n_points = 11
            n_control = lanes.shape[1]
            A = np.zeros((n_points, n_control))
            t = np.arange(n_points) / (n_points - 1)
            for i in range(n_points):
                for j in range(n_control):
                    A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
            bezier_A = torch.tensor(A, dtype=torch.float32).to(lanes.device)
            lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
            lanes = lanes.reshape(lanes.shape[0], -1)

            return lanes