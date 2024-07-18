import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST

@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
    
@MATCH_COST.register_module()
class LaneL1Cost:
    r"""
    Notes
    -----
    Adapted from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/match_costs/match_cost.py#L11.

    """
    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, lane_pred, gt_lanes):
        lane_cost = torch.cdist(lane_pred, gt_lanes, p=1)
        return lane_cost * self.weight

