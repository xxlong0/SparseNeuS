import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import icecream as ic
from loss.ncc import NCC
from termcolor import colored


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class PixelColorLoss(nn.Module):
    def __init__(self):
        super(PixelColorLoss, self).__init__()
        self.eps = 1e-4

    def forward(self, pred, gt, mask ):
        """

        :param pred: [N_pts, 3]
        :param gt: [N_pts, 3]
        :param mask: [N_pts]
        :return:
        """

        error = torch.abs(pred - gt).sum(dim=-1, keepdim=False)  # [N_pts]
        
        if torch.sum(mask)> 10:
            error = error[mask]
            loss = torch.mean(error)
        else:
            loss = 0
        
        return loss
    
    
class AccMaskLoss(nn.Module):
    def __init__(self):
        super(AccMaskLoss, self).__init__()
    
    def forward(self, pred, gt):
        """
        :param pred: [N_pts, 1]
        :param gt: [N_pts, 1]
        :return:
        """

        loss = F.binary_cross_entropy(pred.clip(1e-3, 1.0 - 1e-3), gt)
        
        return loss

class PatchColorLoss(nn.Module):
    def __init__(self, type='ncc', h_patch_size=3):
        super(PatchColorLoss, self).__init__()
        self.type = type  # 'l1' or 'ncc' loss
        self.ncc = NCC(h_patch_size=h_patch_size)
        self.eps = 1e-4

        print("Initialize patch color loss: type {} patch_size {}".format(type, h_patch_size))

    def forward(self, pred, gt, mask, penalize_ratio=0.7 ):
        """

        :param pred: [N_pts, Npx, 3]
        :param gt: [N_pts, Npx, 3]
        :param mask: [N_pts]
        :return:
        """

        pred = pred[mask]
        gt = gt[mask]
        if self.type == 'l1':
            error = torch.abs(pred - gt).mean(dim=-1, keepdim=False).sum(dim=-1, keepdim=False)  # [N_pts]
        elif self.type == 'ncc':
            error = 1 - self.ncc(pred[:, None, :, :], gt)[:, 0]  # ncc 1 positive, -1 negative
            error, indices = torch.sort(error)
            # only sum relatively small errors
            s_error = torch.index_select(error, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
            # mask[int(penalize_ratio * mask.shape[0]):] = False  # can help boundaries
        elif self.type == 'ssd':
            error = ((pred - gt) ** 2).mean(dim=-1, keepdim=False).sum(dim=-1, keepdims=False)

        # error = error[mask]

        return torch.mean(s_error)

