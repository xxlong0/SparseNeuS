import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
import re
import os
import logging
from glob import glob

from models.rays import gen_rays_from_single_image, gen_random_rays_from_single_image

from data.scene import get_boundingbox


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # ? why need transpose here
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # ! return cam2world matrix here


class DtuFit:
    def __init__(self, root_dir, split, scan_id, n_views, train_img_idx=[], test_img_idx=[],
                 img_wh=[800, 600], clip_wh=[0, 0], original_img_wh=[1600, 1200],
                 N_rays=512, h_patch_size=5, near=425, far=900):
        super(DtuFit, self).__init__()
        logging.info('Load data: Begin')

        self.root_dir = root_dir
        self.split = split
        self.scan_id = scan_id
        self.n_views = n_views

        self.near = near
        self.far = far

        if self.scan_id is not None:
            self.data_dir = os.path.join(self.root_dir, self.scan_id)
        else:
            self.data_dir = self.root_dir

        self.img_wh = img_wh
        self.clip_wh = clip_wh

        if len(self.clip_wh) == 2:
            self.clip_wh = self.clip_wh + self.clip_wh

        self.original_img_wh = original_img_wh
        self.N_rays = N_rays
        self.h_patch_size = h_patch_size  # used to extract patch for supervision
        self.train_img_idx = train_img_idx
        self.test_img_idx = test_img_idx

        camera_dict = np.load(os.path.join(self.data_dir, 'cameras.npz'), allow_pickle=True)
        self.images_list = sorted(glob(os.path.join(self.data_dir, "image/*.png")))
        self.masks_list = sorted(glob(os.path.join(self.data_dir, "mask/*.png")))
        # world_mat: projection matrix: world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in
                              range(len(self.images_list))]

        self.raw_near_fars = np.stack([np.array([self.near, self.far]) for i in range(len(self.images_list))])

        # - reference image; transform the world system to the ref-camera system
        self.ref_img_idx = self.train_img_idx[0]
        ref_world_mat = self.world_mats_np[self.ref_img_idx]
        self.ref_w2c = np.linalg.inv(load_K_Rt_from_P(None, ref_world_mat[:3, :4])[1])

        self.all_images = []
        self.all_intrinsics = []
        self.all_w2cs = []
        self.all_masks = []

        self.load_scene()  # load the scene

        # ! estimate scale_mat
        self.scale_mat, self.scale_factor = self.cal_scale_mat(
            img_hw=[self.img_wh[1], self.img_wh[0]],
            intrinsics=self.all_intrinsics[self.train_img_idx],
            extrinsics=self.all_w2cs[self.train_img_idx],
            near_fars=self.raw_near_fars[self.train_img_idx],
            factor=1.1)

        # * after scaling and translation, unit bounding box
        self.scaled_intrinsics, self.scaled_w2cs, self.scaled_c2ws, \
        self.scaled_affine_mats, self.scaled_near_fars = self.scale_cam_info()

        self.bbox_min = np.array([-1.0, -1.0, -1.0])
        self.bbox_max = np.array([1.0, 1.0, 1.0])
        self.partial_vol_origin = torch.Tensor([-1., -1., -1.])

        logging.info('Load data: End')

    def load_scene(self):

        scale_x = self.img_wh[0] / self.original_img_wh[0]
        scale_y = self.img_wh[1] / self.original_img_wh[1]

        for idx in range(len(self.images_list)):
            image = cv.imread(self.images_list[idx])
            mask = cv.imread(self.masks_list[idx], -1)[:, :, 3]
            
            image = cv.resize(image, (self.img_wh[0], self.img_wh[1])) / 255.
            image = image[self.clip_wh[1]:self.img_wh[1] - self.clip_wh[3],
                    self.clip_wh[0]:self.img_wh[0] - self.clip_wh[2]]
            
            
            mask = cv.resize(mask, (self.img_wh[0], self.img_wh[1])) / 255.
            mask = mask[self.clip_wh[1]:self.img_wh[1] - self.clip_wh[3],
                    self.clip_wh[0]:self.img_wh[0] - self.clip_wh[2]]
            mask = mask > 0.5
            
            self.all_images.append(np.transpose(image[:, :, ::-1], (2, 0, 1)))
            self.all_masks.append(mask)

            P = self.world_mats_np[idx]
            P = P[:3, :4]
            intrinsics, c2w = load_K_Rt_from_P(None, P)
            w2c = np.linalg.inv(c2w)

            intrinsics[:1] *= scale_x
            intrinsics[1:2] *= scale_y

            intrinsics[0, 2] -= self.clip_wh[0]
            intrinsics[1, 2] -= self.clip_wh[1]

            self.all_intrinsics.append(intrinsics)
            # - transform from world system to ref-camera system
            self.all_w2cs.append(w2c @ np.linalg.inv(self.ref_w2c))

        self.all_images = torch.from_numpy(np.stack(self.all_images)).to(torch.float32)
        self.all_intrinsics = torch.from_numpy(np.stack(self.all_intrinsics)).to(torch.float32)
        self.all_w2cs = torch.from_numpy(np.stack(self.all_w2cs)).to(torch.float32)
        self.all_masks = torch.from_numpy(np.stack(self.all_masks)).to(torch.float32)
        self.img_wh = [self.img_wh[0] - self.clip_wh[0] - self.clip_wh[2],
                       self.img_wh[1] - self.clip_wh[1] - self.clip_wh[3]]

    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)
        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()

    def scale_cam_info(self):
        new_intrinsics = []
        new_near_fars = []
        new_w2cs = []
        new_c2ws = []
        new_affine_mats = []
        for idx in range(len(self.all_images)):
            intrinsics = self.all_intrinsics[idx]
            P = intrinsics @ self.all_w2cs[idx] @ self.scale_mat
            P = P.cpu().numpy()[:3, :4]

            # - should use load_K_Rt_from_P() to obtain c2w
            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)
            new_intrinsics.append(intrinsics)
            affine_mat = np.eye(4)
            affine_mat[:3, :4] = intrinsics[:3, :3] @ w2c[:3, :4]
            new_affine_mats.append(affine_mat)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1

            new_near_fars.append([0.95 * near, 1.05 * far])

        new_intrinsics, new_w2cs, new_c2ws, new_affine_mats, new_near_fars = \
            np.stack(new_intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), \
            np.stack(new_affine_mats), np.stack(new_near_fars)

        new_intrinsics = torch.from_numpy(np.float32(new_intrinsics))
        new_w2cs = torch.from_numpy(np.float32(new_w2cs))
        new_c2ws = torch.from_numpy(np.float32(new_c2ws))
        new_affine_mats = torch.from_numpy(np.float32(new_affine_mats))
        new_near_fars = torch.from_numpy(np.float32(new_near_fars))

        return new_intrinsics, new_w2cs, new_c2ws, new_affine_mats, new_near_fars


    def get_conditional_sample(self):
        sample = {}
        support_idxs = self.train_img_idx

        sample['images'] = self.all_images[support_idxs]  # (V, 3, H, W)
        sample['w2cs'] = self.scaled_w2cs[self.train_img_idx]  # (V, 4, 4)
        sample['c2ws'] = self.scaled_c2ws[self.train_img_idx]  # (V, 4, 4)
        sample['near_fars'] = self.scaled_near_fars[self.train_img_idx]  # (V, 2)
        sample['intrinsics'] = self.scaled_intrinsics[self.train_img_idx][:, :3, :3]  # (V, 3, 3)
        sample['affine_mats'] = self.scaled_affine_mats[self.train_img_idx]  # ! in world space

        sample['scan'] = self.scan_id
        sample['scale_factor'] = torch.tensor(self.scale_factor)
        sample['scale_mat'] = torch.from_numpy(self.scale_mat)
        sample['trans_mat'] = torch.from_numpy(np.linalg.inv(self.ref_w2c))
        sample['img_wh'] = torch.from_numpy(np.array(self.img_wh))
        sample['partial_vol_origin'] = torch.tensor(self.partial_vol_origin, dtype=torch.float32)

        return sample

    def __len__(self):
        if self.split == 'train':
            return self.n_views * 1000
        else:
            return len(self.test_img_idx) * 1000

    def __getitem__(self, idx):
        sample = {}

        if self.split == 'train':
            render_idx = self.train_img_idx[idx % self.n_views]
            support_idxs = [idx for idx in self.train_img_idx if idx != render_idx]
        else:
            # render_idx = idx % self.n_test_images + self.n_train_images
            render_idx = self.test_img_idx[idx % len(self.test_img_idx)]
            support_idxs = [render_idx]

        sample['images'] = self.all_images[support_idxs]  # (V, 3, H, W)
        sample['w2cs'] = self.scaled_w2cs[support_idxs]  # (V, 4, 4)
        sample['c2ws'] = self.scaled_c2ws[support_idxs]  # (V, 4, 4)
        sample['intrinsics'] = self.scaled_intrinsics[support_idxs][:, :3, :3]  # (V, 3, 3)
        sample['affine_mats'] = self.scaled_affine_mats[support_idxs]  # ! in world space
        sample['scan'] = self.scan_id
        sample['scale_factor'] = torch.tensor(self.scale_factor)
        sample['img_wh'] = torch.from_numpy(np.array(self.img_wh))
        sample['partial_vol_origin'] = torch.tensor(self.partial_vol_origin, dtype=torch.float32)
        sample['img_index'] = torch.tensor(render_idx)

        # - query image
        sample['query_image'] = self.all_images[render_idx]
        sample['query_mask'] = self.all_masks[render_idx]
        sample['query_c2w'] = self.scaled_c2ws[render_idx]
        sample['query_w2c'] = self.scaled_w2cs[render_idx]
        sample['query_intrinsic'] = self.scaled_intrinsics[render_idx]
        sample['query_near_far'] = self.scaled_near_fars[render_idx]
        sample['meta'] = str(self.scan_id) + "_" + os.path.basename(self.images_list[render_idx])
        sample['scale_mat'] = torch.from_numpy(self.scale_mat)
        sample['trans_mat'] = torch.from_numpy(np.linalg.inv(self.ref_w2c))
        sample['rendering_c2ws'] = self.scaled_c2ws[self.test_img_idx]
        sample['rendering_imgs_idx'] = torch.Tensor(np.array(self.test_img_idx).astype(np.int32))

        # - generate rays
        if self.split == 'val' or self.split == 'test':
            sample_rays = gen_rays_from_single_image(
                self.img_wh[1], self.img_wh[0],
                sample['query_image'],
                sample['query_intrinsic'],
                sample['query_c2w'],
                depth=None,
                mask=sample['query_mask'])
        else:
            sample_rays = gen_random_rays_from_single_image(
                self.img_wh[1], self.img_wh[0],
                self.N_rays,
                sample['query_image'],
                sample['query_intrinsic'],
                sample['query_c2w'],
                depth=None,
                mask=sample['query_mask'],
                dilated_mask=None,
                importance_sample=False,
                h_patch_size=self.h_patch_size
            )

        sample['rays'] = sample_rays

        return sample
