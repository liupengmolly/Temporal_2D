from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        #
        # heatmap_pred = output.reshape((batch_size, num_joints, -1))
        # heatmap_gt = target.reshape((batch_size, num_joints, -1))
        # pixel_nums = heatmap_gt.size(-1)
        # target_weight = target_weight.expand(target_weight.size(0), num_joints, pixel_nums)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                heatmap_pred = heatmap_pred.mul(target_weight[:, idx])
                heatmap_gt = heatmap_gt.mul(target_weight[:, idx])
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class MultiJointsMSELoss(JointsMSELoss):
    def __init__(self, use_target_weight, temporal):
        super(MultiJointsMSELoss, self).__init__(use_target_weight)
        self.temporal = temporal

    def forward(self, outputs, targets, target_weight=None):
        # 因为有个init_heatmap 所以第一张图会算两次
        # losses = []
        loss = 0.0
        # losses.append(super().forward(outputs[0],targets[:,0,:,:,:]))
        for i in range(self.temporal):
            loss += super().forward(outputs[i], targets[:,i,:,:,:])
        return loss/self.temporal

