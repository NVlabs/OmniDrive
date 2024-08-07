import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import HungarianAssigner, AssignResult
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

@BBOX_ASSIGNERS.register_module()
class LaneHungarianAssigner(HungarianAssigner):

    def assign(self,
               lane_pred,
               cls_pred,
               gt_lanes,
               gt_labels,
               img_meta,
               gt_lanes_ignore=None,
               eps=1e-7):
        assert gt_lanes_ignore is None, \
            'Only case when gt_lanes_ignore is None is supported.'
        num_gts, num_lanes = gt_lanes.size(0), lane_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = lane_pred.new_full((num_lanes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = lane_pred.new_full((num_lanes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_lanes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and lanecost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        reg_cost = self.reg_cost(lane_pred, gt_lanes)
        # weighted sum of above three costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = torch.nan_to_num(cost)
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            lane_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            lane_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)