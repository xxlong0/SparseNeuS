"""
Trainer for fine-tuning
"""
import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import logging
import mcubes
import trimesh
from icecream import ic
from models.render_utils import sample_pdf
from utils.misc_utils import visualize_depth_numpy

from utils.training_utils import tocuda, numpy2tensor
from loss.depth_metric import compute_depth_errors
from loss.color_loss import OcclusionColorLoss, OcclusionColorPatchLoss
from loss.depth_loss import DepthLoss, DepthSmoothLoss

from models.projector import Projector

from models.rays import gen_rays_between

from models.sparse_neus_renderer import SparseNeuSRenderer

import pdb


class FinetuneTrainer(nn.Module):
    """
    Trainer used for fine-tuning
    """

    def __init__(self,
                 rendering_network_outside,
                 pyramid_feature_network_lod0,
                 pyramid_feature_network_lod1,
                 sdf_network_lod0,
                 sdf_network_lod1,
                 variance_network_lod0,
                 variance_network_lod1,
                 sdf_network_finetune,
                 finetune_lod,  # which lod fine-tuning use
                 n_samples,
                 n_importance,
                 n_outside,
                 perturb,
                 alpha_type='div',
                 conf=None
                 ):
        super(FinetuneTrainer, self).__init__()

        self.conf = conf
        self.base_exp_dir = conf['general.base_exp_dir']

        self.finetune_lod = finetune_lod

        self.anneal_start = self.conf.get_float('train.anneal_start', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.end_iter = self.conf.get_int('train.end_iter')

        # network setups
        self.rendering_network_outside = rendering_network_outside
        self.pyramid_feature_network_geometry_lod0 = pyramid_feature_network_lod0  # 2D pyramid feature network for geometry
        self.pyramid_feature_network_geometry_lod1 = pyramid_feature_network_lod1  # use differnet networks for the two lods

        self.sdf_network_lod0 = sdf_network_lod0  # the first lod is density_network
        self.sdf_network_lod1 = sdf_network_lod1

        # - warpped by ModuleList to support DataParallel
        self.variance_network_lod0 = variance_network_lod0
        self.variance_network_lod1 = variance_network_lod1
        self.variance_network_finetune = variance_network_lod0 if self.finetune_lod == 0 else variance_network_lod1

        self.sdf_network_finetune = sdf_network_finetune

        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.perturb = perturb
        self.alpha_type = alpha_type

        self.sdf_renderer_finetune = SparseNeuSRenderer(
            self.rendering_network_outside,
            self.sdf_network_finetune,
            self.variance_network_finetune,
            None,  # rendering_network
            self.n_samples,
            self.n_importance,
            self.n_outside,
            self.perturb,
            alpha_type='div',
            conf=self.conf)

        # sdf network weights
        self.sdf_igr_weight = self.conf.get_float('train.sdf_igr_weight')
        self.sdf_sparse_weight = self.conf.get_float('train.sdf_sparse_weight', default=0)

        self.sdf_decay_param = self.conf.get_float('train.sdf_decay_param', default=100)
        self.color_pixel_weight = self.conf.get_float('train.color_pixel_weight', default=1.0)
        self.color_patch_weight = self.conf.get_float('train.color_patch_weight', default=0.)
        self.tv_weight = self.conf.get_float('train.tv_weight', default=0.001)  # no use
        self.visibility_beta = self.conf.get_float('train.visibility_beta', default=0.025)
        self.visibility_gama = self.conf.get_float('train.visibility_gama', default=0.015)
        self.visibility_penalize_ratio = self.conf.get_float('train.visibility_penalize_ratio', default=0.8)
        self.visibility_weight_thred = self.conf.get_list('train.visibility_weight_thred', default=[0.7])
        self.if_visibility_aware = self.conf.get_bool('train.if_visibility_aware', default=True)
        self.train_from_scratch = self.conf.get_bool('train.train_from_scratch', default=False)

        self.depth_criterion = DepthLoss()
        self.depth_smooth_criterion = DepthSmoothLoss()
        self.occlusion_color_criterion = OcclusionColorLoss(beta=self.visibility_beta,
                                                            gama=self.visibility_gama,
                                                            weight_thred=self.visibility_weight_thred,
                                                            occlusion_aware=self.if_visibility_aware)
        self.occlusion_color_patch_criterion = OcclusionColorPatchLoss(
            type=self.conf.get_string('train.patch_loss_type', default='ncc'),
            h_patch_size=self.conf.get_int('model.h_patch_size', default=5),
            beta=self.visibility_beta, gama=self.visibility_gama,
            weight_thred=self.visibility_weight_thred,
            occlusion_aware=self.if_visibility_aware
        )

        # self.iter_step = 0
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')

        # - True if fine-tuning
        self.if_fitted_rendering = self.conf.get_bool('train.if_fitted_rendering', default=False)

    def get_trainable_params(self):
        # set trainable params

        params = []
        faster_params = []
        slower_params = []

        params += self.variance_network_finetune.parameters()
        slower_params += self.sdf_network_finetune.sparse_volume_lod0.parameters()
        params += self.sdf_network_finetune.sdf_layer.parameters()

        faster_params += self.sdf_network_finetune.renderer.parameters()

        self.params_to_train = {
            'slower_params': slower_params,
            'params': params,
            'faster_params': faster_params
        }

        return self.params_to_train

    @torch.no_grad()
    def prepare_con_volume(self, sample):
        # * only support batch_size==1
        sizeW = sample['img_wh'][0]
        sizeH = sample['img_wh'][1]
        partial_vol_origin = sample['partial_vol_origin'][None, :]  # [B, 3]
        near, far = sample['near_fars'][0, :1], sample['near_fars'][0, 1:]
        near = 0.8 * near
        far = 1.2 * far

        imgs = sample['images']
        intrinsics = sample['intrinsics']
        intrinsics_l_4x = intrinsics.clone()
        intrinsics_l_4x[:, :2] *= 0.25
        w2cs = sample['w2cs']
        c2ws = sample['c2ws']
        proj_matrices = sample['affine_mats'][None, :, :, :]

        # ***********************     Lod==0     ***********************

        with torch.no_grad():
            geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs)
            conditional_features_lod0 = self.sdf_network_lod0.get_conditional_volume(
                feature_maps=geometry_feature_maps[None, :, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                lod=0,
            )

        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']

        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']
        coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        if self.finetune_lod == 0:
            return con_volume_lod0, con_valid_mask_volume_lod0, coords_lod0

        # * extract depth maps for all the images for adaptive rendering_network
        depth_maps_lod0, depth_masks_lod0 = None, None
        if self.finetune_lod == 1:
            sdf_volume_lod0 = self.sdf_network_lod0.get_sdf_volume(
                con_volume_lod0, con_valid_mask_volume_lod0,
                coords_lod0, partial_vol_origin)  # [1, 1, dX, dY, dZ]

        if self.finetune_lod == 1:
            geometry_feature_maps_lod1 = self.obtain_pyramid_feature_maps(imgs, lod=1)

            pre_coords, pre_feats = self.sdf_renderer_finetune.get_valid_sparse_coords_by_sdf(
                sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0],
                maximum_pts=200000)

            pre_coords[:, 1:] = pre_coords[:, 1:] * 2

            conditional_features_lod1 = self.sdf_network_lod1.get_conditional_volume(
                feature_maps=geometry_feature_maps_lod1[None, :, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                pre_coords=pre_coords,
                pre_feats=pre_feats
            )

            con_volume_lod1 = conditional_features_lod1['dense_volume_scale1']
            con_valid_mask_volume_lod1 = conditional_features_lod1['valid_mask_volume_scale1']
            coords_lod1 = conditional_features_lod1['coords_scale1']  # [1,3,wX,wY,wZ]
            con_valid_mask_volume_lod0 = F.interpolate(con_valid_mask_volume_lod0, scale_factor=2)

        return con_volume_lod1, con_valid_mask_volume_lod1, coords_lod1

    def initialize_finetune_network(self, sample, sparse_con_volume=None, sparse_coords_volume=None,
                                    train_from_scratch=False):

        if not train_from_scratch:
            if sparse_con_volume is None:  # if the
                con_volume, con_mask_volume, _ = self.prepare_con_volume(sample)

                device = con_volume.device

                self.sdf_network_finetune.initialize_conditional_volumes(
                    con_volume,
                    con_mask_volume
                )
            else:
                self.sdf_network_finetune.initialize_conditional_volumes(
                    None,
                    None,
                    sparse_con_volume,
                    sparse_coords_volume
                )
        else:
            device = sample['images'].device
            vol_dims = self.sdf_network_finetune.vol_dims
            con_volume = torch.zeros(
                [1, self.sdf_network_finetune.regnet_d_out, vol_dims[0], vol_dims[1], vol_dims[2]]).to(device)
            con_mask_volume = torch.ones([1, 1, vol_dims[0], vol_dims[1], vol_dims[2]]).to(device)
            self.sdf_network_finetune.initialize_conditional_volumes(
                con_volume,
                con_mask_volume
            )

        self.sdf_network_lod0, self.sdf_network_lod1 = None, None
        self.pyramid_feature_network_geometry_lod0, self.pyramid_feature_network_geometry_lod1 = None, None

    def train_step(self, sample,
                   perturb_overwrite=-1,
                   background_rgb=None,
                   iter_step=0,
                   chunk_size=512,
                   save_vis=False,
                   ):

        # * finetune on one specific scene
        # * only support batch_size==1
        # ! attention: the list of string cannot be splited in DataParallel
        batch_idx = sample['batch_idx'][0]
        meta = sample['meta'][batch_idx]  # the scan lighting ref_view info

        sizeW = sample['img_wh'][0][0]
        sizeH = sample['img_wh'][0][1]
        partial_vol_origin = sample['partial_vol_origin']  # [B, 3]
        near, far = sample['query_near_far'][0, :1], sample['query_near_far'][0, 1:]

        img_index = sample['img_index'][0]  # [n]

        # the full-size ray variables
        sample_rays = sample['rays']
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]
        rays_ndc_uv = sample_rays['rays_ndc_uv'][0]

        imgs = sample['images'][0]
        intrinsics = sample['intrinsics'][0]
        w2cs = sample['w2cs'][0]
        proj_matrices = sample['affine_mats']
        scale_mat = sample['scale_mat']
        trans_mat = sample['trans_mat']

        query_c2w = sample['query_c2w']

        # ***********************     Lod==0     ***********************

        conditional_features_lod0 = self.sdf_network_finetune.get_conditional_volume()

        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']
        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']

        # coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        # # - extract mesh
        if iter_step % self.val_mesh_freq == 0:
            torch.cuda.empty_cache()
            self.validate_mesh(self.sdf_network_finetune,
                               self.sdf_renderer_finetune.extract_geometry,
                               conditional_volume=con_volume_lod0,
                               lod=0,
                               threshold=0.,
                               occupancy_mask=con_valid_mask_volume_lod0[0, 0],
                               mode='ft', meta=meta,
                               iter_step=iter_step, scale_mat=scale_mat, trans_mat=trans_mat)

            torch.cuda.empty_cache()

        render_out = self.sdf_renderer_finetune.render(
            rays_o, rays_d, near, far,
            self.sdf_network_finetune,
            None,  # rendering_network
            background_rgb=background_rgb,
            alpha_inter_ratio=1.0,
            # * related to conditional feature
            lod=0,
            conditional_volume=con_volume_lod0,
            conditional_valid_mask_volume=con_valid_mask_volume_lod0,
            # * 2d feature maps
            feature_maps=None,
            color_maps=imgs,
            w2cs=w2cs,
            intrinsics=intrinsics,
            img_wh=[sizeW, sizeH],
            query_c2w=query_c2w,
            if_general_rendering=False,
            img_index=img_index,
            rays_uv=rays_ndc_uv if self.color_patch_weight > 0 else None,
        )

        # * optional TV regularizer, we don't use in this paper
        if self.tv_weight > 0:
            tv = self.sdf_network_finetune.tv_regularizer()
        else:
            tv = 0.0
        render_out['tv'] = tv
        loss_lod0, losses_lod0, depth_statis_lod0 = self.cal_losses_sdf(render_out, sample_rays, iter_step)

        losses = {
            # - lod 0
            'loss_lod0': loss_lod0,
            'losses_lod0': losses_lod0,
            'depth_statis_lod0': depth_statis_lod0,
        }

        return losses

    def val_step(self, sample,
                 perturb_overwrite=-1,
                 background_rgb=None,
                 iter_step=0,
                 chunk_size=512,
                 save_vis=True,
                 ):
        # * only support batch_size==1
        # ! attention: the list of string cannot be splited in DataParallel
        batch_idx = sample['batch_idx'][0]
        meta = sample['meta'][batch_idx]  # the scan lighting ref_view info

        sizeW = sample['img_wh'][0][0]
        sizeH = sample['img_wh'][0][1]
        H, W = sizeH, sizeW

        partial_vol_origin = sample['partial_vol_origin']  # [B, 3]
        near, far = sample['query_near_far'][0, :1], sample['query_near_far'][0, 1:]

        img_index = sample['img_index'][0]  # [n]

        # the ray variables
        sample_rays = sample['rays']
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]
        rays_ndc_uv = sample_rays['rays_ndc_uv'][0]

        imgs = sample['images'][0]
        intrinsics = sample['intrinsics'][0]
        intrinsics_l_4x = intrinsics.clone()
        intrinsics_l_4x[:, :2] *= 0.25
        w2cs = sample['w2cs'][0]
        c2ws = sample['c2ws'][0]
        proj_matrices = sample['affine_mats']

        # - the image to render
        scale_mat = sample['scale_mat']  # [1,4,4]  used to convert mesh into true scale
        trans_mat = sample['trans_mat']
        query_c2w = sample['query_c2w']  # [1,4,4]
        query_w2c = sample['query_w2c']  # [1,4,4]
        true_img = sample['query_image'][0]
        true_img = np.uint8(true_img.permute(1, 2, 0).cpu().numpy() * 255)

        depth_min, depth_max = near.cpu().numpy(), far.cpu().numpy()

        true_depth = sample['query_depth'] if 'query_depth' in sample.keys() else None
        if true_depth is not None:
            true_depth = true_depth[0].cpu().numpy()
            true_depth_colored = visualize_depth_numpy(true_depth, [depth_min, depth_max])[0]
        else:
            true_depth_colored = None

        rays_o = rays_o.reshape(-1, 3).split(chunk_size)
        rays_d = rays_d.reshape(-1, 3).split(chunk_size)

        # - obtain conditional features
        with torch.no_grad():
            # - lod 0
            conditional_features_lod0 = self.sdf_network_finetune.get_conditional_volume()

        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']
        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']
        # coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        out_rgb_fine = []
        out_normal_fine = []
        out_depth_fine = []

        out_rgb_mlp = []

        if save_vis:
            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):

                # ****** lod 0 ****
                render_out = self.sdf_renderer_finetune.render(
                    rays_o_batch, rays_d_batch, near, far,
                    self.sdf_network_finetune,
                    None,
                    background_rgb=background_rgb,
                    alpha_inter_ratio=1.,
                    # * related to conditional feature
                    lod=0,
                    conditional_volume=con_volume_lod0,
                    conditional_valid_mask_volume=con_valid_mask_volume_lod0,
                    # * 2d feature maps
                    feature_maps=None,
                    color_maps=imgs,
                    w2cs=w2cs,
                    intrinsics=intrinsics,
                    img_wh=[sizeW, sizeH],
                    query_c2w=query_c2w,
                    if_general_rendering=False,
                    if_render_with_grad=False,
                    img_index=img_index,
                    # rays_uv=rays_ndc_uv
                )

                feasible = lambda key: ((key in render_out) and (render_out[key] is not None))

                if feasible('depth'):
                    out_depth_fine.append(render_out['depth'].detach().cpu().numpy())

                # if render_out['color_coarse'] is not None:
                if feasible('color_fine'):
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

                if feasible('color_mlp'):
                    out_rgb_mlp.append(render_out['color_mlp'].detach().cpu().numpy())

                if feasible('gradients') and feasible('weights'):
                    if render_out['inside_sphere'] is not None:
                        out_normal_fine.append((render_out['gradients'] * render_out['weights'][:,
                                                                          :self.n_samples + self.n_importance,
                                                                          None] * render_out['inside_sphere'][
                                                    ..., None]).sum(dim=1).detach().cpu().numpy())
                    else:
                        out_normal_fine.append((render_out['gradients'] * render_out['weights'][:,
                                                                          :self.n_samples + self.n_importance,
                                                                          None]).sum(dim=1).detach().cpu().numpy())
                del render_out

            # - save visualization of lod 0

            self.save_visualization(true_img, true_depth_colored, out_depth_fine, out_normal_fine,
                                    query_w2c[0], out_rgb_fine, H, W,
                                    depth_min, depth_max, iter_step, meta, "val_lod0",
                                    out_color_mlp=out_rgb_mlp)

        # - extract mesh
        if (iter_step % self.val_mesh_freq == 0):
            torch.cuda.empty_cache()
            self.validate_mesh(self.sdf_network_finetune,
                               self.sdf_renderer_finetune.extract_geometry,
                               conditional_volume=con_volume_lod0, lod=0,
                               threshold=0,
                               occupancy_mask=con_valid_mask_volume_lod0[0, 0],
                               mode='val', meta=meta,
                               iter_step=iter_step, scale_mat=scale_mat, trans_mat=trans_mat)

            torch.cuda.empty_cache()

    def save_visualization(self, true_img, true_colored_depth, out_depth, out_normal, w2cs, out_color, H, W,
                           depth_min, depth_max, iter_step, meta, comment, out_color_mlp=[]):
        if len(out_color) > 0:
            img_fine = (np.concatenate(out_color, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)

        if len(out_color_mlp) > 0:
            img_mlp = (np.concatenate(out_color_mlp, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)

        if len(out_normal) > 0:
            normal_img = np.concatenate(out_normal, axis=0)
            rot = w2cs[:3, :3].detach().cpu().numpy()
            # - convert normal from world space to camera space
            normal_img = (np.matmul(rot[None, :, :],
                                    normal_img[:, :, None]).reshape([H, W, 3]) * 128 + 128).clip(0, 255)
        if len(out_depth) > 0:
            pred_depth = np.concatenate(out_depth, axis=0).reshape([H, W])
            pred_depth_colored = visualize_depth_numpy(pred_depth, [depth_min, depth_max])[0]

        if len(out_depth) > 0:
            os.makedirs(os.path.join(self.base_exp_dir, 'depths_' + comment), exist_ok=True)
            if true_colored_depth is not None:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'depths_' + comment,
                                 '{:0>8d}_{}.png'.format(iter_step, meta)),
                    np.concatenate(
                        [true_colored_depth, pred_depth_colored, true_img])[:, :, ::-1])
            else:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'depths_' + comment,
                                 '{:0>8d}_{}.png'.format(iter_step, meta)),
                    np.concatenate(
                        [pred_depth_colored, true_img])[:, :, ::-1])
        if len(out_color) > 0:
            os.makedirs(os.path.join(self.base_exp_dir, 'synthesized_color_' + comment), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'synthesized_color_' + comment,
                                    '{:0>8d}_{}.png'.format(iter_step, meta)),
                       np.concatenate(
                           [img_fine, true_img])[:, :, ::-1])  # bgr2rgb

        if len(out_color_mlp) > 0:
            os.makedirs(os.path.join(self.base_exp_dir, 'synthesized_color_mlp_' + comment), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'synthesized_color_mlp_' + comment,
                                    '{:0>8d}_{}.png'.format(iter_step, meta)),
                       np.concatenate(
                           [img_mlp, true_img])[:, :, ::-1])  # bgr2rgb

        if len(out_normal) > 0:
            os.makedirs(os.path.join(self.base_exp_dir, 'normals_' + comment), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'normals_' + comment,
                                    '{:0>8d}_{}.png'.format(iter_step, meta)),
                       normal_img[:, :, ::-1])

    def forward(self, sample,
                perturb_overwrite=-1,
                background_rgb=None,
                iter_step=0,
                mode='train',
                save_vis=False,
                ):

        if mode == 'train':
            return self.train_step(sample,
                                   perturb_overwrite=perturb_overwrite,
                                   background_rgb=background_rgb,
                                   iter_step=iter_step,
                                   )
        elif mode == 'val':
            return self.val_step(sample,
                                 perturb_overwrite=perturb_overwrite,
                                 background_rgb=background_rgb,
                                 iter_step=iter_step, save_vis=save_vis,
                                 )

    def obtain_pyramid_feature_maps(self, imgs, lod=0):
        """
        get feature maps of all conditional images
        :param imgs:
        :return:
        """

        if lod == 0:
            extractor = self.pyramid_feature_network_geometry_lod0
        elif lod >= 1:
            extractor = self.pyramid_feature_network_geometry_lod1

        pyramid_feature_maps = extractor(imgs)

        # * the pyramid features are very important, if only use the coarst features, hard to optimize
        fused_feature_maps = torch.cat([
            F.interpolate(pyramid_feature_maps[0], scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(pyramid_feature_maps[1], scale_factor=2, mode='bilinear', align_corners=True),
            pyramid_feature_maps[2]
        ], dim=1)

        return fused_feature_maps

    def cal_losses_sdf(self, render_out, sample_rays, iter_step=-1):

        def get_weight(iter_step, weight):
            if iter_step < 0:
                return weight

            if self.anneal_end == 0.0:
                return weight
            elif iter_step < self.anneal_start:
                return 0.0
            else:
                return np.min(
                    [1.0,
                     (iter_step - self.anneal_start) / (self.anneal_end * 2 - self.anneal_start)]) * weight

        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]
        true_rgb = sample_rays['rays_color'][0]

        if 'rays_depth' in sample_rays.keys():
            true_depth = sample_rays['rays_depth'][0]
        else:
            true_depth = None
        mask = sample_rays['rays_mask'][0]

        color_fine = render_out['color_fine']
        color_fine_mask = render_out['color_fine_mask']
        depth_pred = render_out['depth']

        variance = render_out['variance']
        cdf_fine = render_out['cdf_fine']
        weight_sum = render_out['weights_sum']

        if self.train_from_scratch:
            occlusion_aware = False if iter_step < 5000 else True
        else:
            occlusion_aware = True

        gradient_error_fine = render_out['gradient_error_fine']

        sdf = render_out['sdf']

        # * color generated by mlp
        color_mlp = render_out['color_mlp']
        color_mlp_mask = render_out['color_mlp_mask']

        if color_mlp is not None:
            # Color loss
            color_mlp_mask = color_mlp_mask[..., 0]

            color_mlp_loss, color_mlp_error = self.occlusion_color_criterion(pred=color_mlp, gt=true_rgb,
                                                                             weight=weight_sum.squeeze(),
                                                                             mask=color_mlp_mask,
                                                                             occlusion_aware=occlusion_aware)

            psnr_mlp = 20.0 * torch.log10(
                1.0 / (((color_mlp[color_mlp_mask] - true_rgb[color_mlp_mask]) ** 2).mean() / (3.0)).sqrt())
        else:
            color_mlp_loss = 0.
            psnr_mlp = 0.

        # - blended patch loss
        blended_color_patch = render_out['blended_color_patch']  # [N_pts, Npx, 3]
        blended_color_patch_mask = render_out['blended_color_patch_mask']  # [N_pts, 1]
        color_patch_loss = 0.0
        color_patch_error = 0.0
        visibility_beta = 0.0
        if blended_color_patch is not None:
            rays_patch_color = sample_rays['rays_patch_color'][0]
            rays_patch_mask = sample_rays['rays_patch_mask'][0]
            patch_mask = (rays_patch_mask * blended_color_patch_mask).float()[:, 0] > 0  # [N_pts]

            color_patch_loss, color_patch_error, visibility_beta = self.occlusion_color_patch_criterion(
                blended_color_patch,
                rays_patch_color,
                weight=weight_sum.squeeze(),
                mask=patch_mask,
                penalize_ratio=self.visibility_penalize_ratio,
                occlusion_aware=occlusion_aware
            )

        if true_depth is not None:
            depth_loss = self.depth_criterion(depth_pred, true_depth, mask)

            # depth evaluation
            depth_statis = compute_depth_errors(depth_pred.detach().cpu().numpy(), true_depth.cpu().numpy(),
                                                mask.cpu().numpy() > 0)
            depth_statis = numpy2tensor(depth_statis, device=rays_o.device)
        else:
            depth_loss = 0.
            depth_statis = None

        # - if without sparse_loss, the mean sdf is 0.02.
        # - use sparse_loss to prevent occluded pts have 0 sdf
        sparse_loss_1 = torch.exp(-1 * torch.abs(render_out['sdf_random']) * self.sdf_decay_param * 10).mean()
        sparse_loss_2 = torch.exp(-1 * torch.abs(sdf) * self.sdf_decay_param).mean()
        sparse_loss = (sparse_loss_1 + sparse_loss_2) / 2

        sdf_mean = torch.abs(sdf).mean()
        sparseness_1 = (torch.abs(sdf) < 0.01).to(torch.float32).mean()
        sparseness_2 = (torch.abs(sdf) < 0.02).to(torch.float32).mean()

        # Eikonal loss
        gradient_error_loss = gradient_error_fine

        # * optional TV regularizer
        if 'tv' in render_out.keys():
            tv = render_out['tv']
        else:
            tv = 0.0

        loss = color_mlp_loss + \
               color_patch_loss * self.color_patch_weight + \
               sparse_loss * get_weight(iter_step, self.sdf_sparse_weight) + \
               gradient_error_loss * self.sdf_igr_weight

        losses = {
            "loss": loss,
            "depth_loss": depth_loss,
            "color_mlp_loss": color_mlp_error,
            "gradient_error_loss": gradient_error_loss,
            "sparse_loss": sparse_loss,
            "sparseness_1": sparseness_1,
            "sparseness_2": sparseness_2,
            "sdf_mean": sdf_mean,
            "psnr_mlp": psnr_mlp,
            "weights_sum": render_out['weights_sum'],
            "alpha_sum": render_out['alpha_sum'],
            "variance": render_out['variance'],
            "sparse_weight": get_weight(iter_step, self.sdf_sparse_weight),
            'color_patch_loss': color_patch_error,
            'visibility_beta': visibility_beta,
            'tv': tv,
        }

        losses = numpy2tensor(losses, device=rays_o.device)

        return loss, losses, depth_statis

    def validate_mesh(self, sdf_network, func_extract_geometry, world_space=True, resolution=512,
                      threshold=0.0, mode='val',
                      # * 3d feature volume
                      conditional_volume=None, lod=None, occupancy_mask=None,
                      bound_min=[-1, -1, -1], bound_max=[1, 1, 1], meta='', iter_step=0, scale_mat=None,
                      trans_mat=None
                      ):
        bound_min = torch.tensor(bound_min, dtype=torch.float32)
        bound_max = torch.tensor(bound_max, dtype=torch.float32)

        vertices, triangles, fields = func_extract_geometry(
            sdf_network,
            bound_min, bound_max, resolution=resolution,
            threshold=threshold, device=conditional_volume.device,
            # * 3d feature volume
            conditional_volume=conditional_volume, lod=lod,
            # occupancy_mask=occupancy_mask
        )

        if scale_mat is not None:
            scale_mat_np = scale_mat.cpu().numpy()
            vertices = vertices * scale_mat_np[0][0, 0] + scale_mat_np[0][:3, 3][None]

        if trans_mat is not None:
            trans_mat_np = trans_mat.cpu().numpy()
            vertices_homo = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=1)
            vertices = np.matmul(trans_mat_np, vertices_homo[:, :, None])[:, :3, 0]

        mesh = trimesh.Trimesh(vertices, triangles)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes_' + mode), exist_ok=True)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes_' + mode,
                                 'mesh_{:0>8d}_{}_lod{:0>1d}.ply'.format(iter_step, meta, lod)))

    def gen_video(self, sample,
                  perturb_overwrite=-1,
                  background_rgb=None,
                  iter_step=0,
                  chunk_size=1024,
                  ):
        # * only support batch_size==1
        batch_idx = sample['batch_idx'][0]
        meta = sample['meta'][batch_idx]  # the scan lighting ref_view info

        sizeW = sample['img_wh'][0][0]
        sizeH = sample['img_wh'][0][1]
        H, W = sizeH, sizeW

        partial_vol_origin = sample['partial_vol_origin']  # [B, 3]
        near, far = sample['query_near_far'][0, :1], sample['query_near_far'][0, 1:] * 0.8

        img_index = sample['img_index'][0]  # [n]

        # the ray variables
        sample_rays = sample['rays']
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]
        rays_ndc_uv = sample_rays['rays_ndc_uv'][0]

        imgs = sample['images'][0]
        intrinsics = sample['intrinsics'][0]
        intrinsics_l_4x = intrinsics.clone()
        intrinsics_l_4x[:, :2] *= 0.25
        w2cs = sample['w2cs'][0]
        c2ws = sample['c2ws'][0]
        proj_matrices = sample['affine_mats']

        # - the image to render
        scale_mat = sample['scale_mat']  # [1,4,4]  used to convert mesh into true scale
        trans_mat = sample['trans_mat']
        query_c2w = sample['query_c2w']  # [1,4,4]
        query_w2c = sample['query_w2c']  # [1,4,4]
        true_img = sample['query_image'][0]
        true_img = np.uint8(true_img.permute(1, 2, 0).cpu().numpy() * 255)
        rendering_c2ws = sample['rendering_c2ws'][0]  # [n, 4, 4]
        rendering_imgs_idx = sample['rendering_imgs_idx'][0]

        depth_min, depth_max = near.cpu().numpy(), far.cpu().numpy()

        true_depth = sample['query_depth'] if 'query_depth' in sample.keys() else None
        if true_depth is not None:
            true_depth = true_depth[0].cpu().numpy()
            true_depth_colored = visualize_depth_numpy(true_depth, [depth_min, depth_max])[0]
        else:
            true_depth_colored = None

        # - obtain conditional features
        with torch.no_grad():
            # - lod 0
            conditional_features_lod0 = self.sdf_network_finetune.get_conditional_volume()

        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']
        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']
        # coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        inter_views_num = 60
        resolution_level = 2
        for r_idx in range(rendering_c2ws.shape[0] - 1):
            for idx in range(inter_views_num):
                query_c2w, rays_o, rays_d = gen_rays_between(
                    rendering_c2ws[r_idx], rendering_c2ws[r_idx + 1], intrinsics[0],
                    np.sin(((idx / 60.0) - 0.5) * np.pi) * 0.5 + 0.5,
                    H, W, resolution_level=resolution_level)

                rays_o = rays_o.reshape(-1, 3).split(chunk_size)
                rays_d = rays_d.reshape(-1, 3).split(chunk_size)

                out_rgb_fine = []
                out_normal_fine = []
                out_depth_fine = []

                for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                    # ****** lod 0 ****
                    render_out = self.sdf_renderer_finetune.render(
                        rays_o_batch, rays_d_batch, near, far,
                        self.sdf_network_finetune,
                        None,
                        background_rgb=background_rgb,
                        alpha_inter_ratio=1.,
                        # * related to conditional feature
                        lod=0,
                        conditional_volume=con_volume_lod0,
                        conditional_valid_mask_volume=con_valid_mask_volume_lod0,
                        # * 2d feature maps
                        feature_maps=None,
                        color_maps=imgs,
                        w2cs=w2cs,
                        intrinsics=intrinsics,
                        img_wh=[sizeW, sizeH],
                        query_c2w=query_c2w,
                        if_general_rendering=False,
                        if_render_with_grad=False,
                        img_index=img_index,
                        # rays_uv=rays_ndc_uv
                    )
                    # pdb.set_trace()
                    feasible = lambda key: ((key in render_out) and (render_out[key] is not None))

                    if feasible('depth'):
                        out_depth_fine.append(render_out['depth'].detach().cpu().numpy())

                    # if render_out['color_coarse'] is not None:
                    if feasible('color_mlp'):
                        out_rgb_fine.append(render_out['color_mlp'].detach().cpu().numpy())
                    if feasible('gradients') and feasible('weights'):
                        if render_out['inside_sphere'] is not None:
                            out_normal_fine.append((render_out['gradients'] * render_out['weights'][:,
                                                                              :self.n_samples + self.n_importance,
                                                                              None] * render_out['inside_sphere'][
                                                        ..., None]).sum(dim=1).detach().cpu().numpy())
                        else:
                            out_normal_fine.append((render_out['gradients'] * render_out['weights'][:,
                                                                              :self.n_samples + self.n_importance,
                                                                              None]).sum(dim=1).detach().cpu().numpy())
                    del render_out

                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape(
                    [H // resolution_level, W // resolution_level, 3, -1]) * 256).clip(0, 255)
                save_dir = os.path.join(self.base_exp_dir, 'render_{}_{}'.format(rendering_imgs_idx[r_idx],
                                                                                 rendering_imgs_idx[r_idx + 1]))
                os.makedirs(save_dir, exist_ok=True)
                # ic(img_fine.shape)
                print(cv.imwrite(
                    os.path.join(save_dir, '{}.png'.format(idx + r_idx * inter_views_num)),
                    img_fine.squeeze()[:, :, ::-1]))
                print(os.path.join(save_dir, '{}.png'.format(idx + r_idx * inter_views_num)))
