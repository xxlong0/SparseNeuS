from torch.utils.data import Dataset
from utils.misc_utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
from data.scene import get_boundingbox

from models.rays import gen_rays_from_single_image, gen_random_rays_from_single_image

from termcolor import colored
import pdb
import random


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
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


# ! load one ref-image with multiple src-images in camera coordinate system
class MVSDatasetDtuPerView(Dataset):
    def __init__(self, root_dir, split, n_views=3, img_wh=(640, 512), downSample=1.0,
                 split_filepath=None, pair_filepath=None,
                 N_rays=512,
                 vol_dims=[128, 128, 128], batch_size=1,
                 clean_image=False, importance_sample=False, test_ref_views=[]):

        self.root_dir = root_dir
        self.split = split

        self.img_wh = img_wh
        self.downSample = downSample
        self.num_all_imgs = 49  # this preprocessed DTU dataset has 49 images
        self.n_views = n_views
        self.N_rays = N_rays
        self.batch_size = batch_size  # - used for construct new metas for gru fusion training

        self.clean_image = clean_image
        self.importance_sample = importance_sample
        self.test_ref_views = test_ref_views  # used for testing
        self.scale_factor = 1.0
        self.scale_mat = np.float32(np.diag([1, 1, 1, 1.0]))

        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'

        self.split_filepath = f'data/dtu/lists/{self.split}.txt' if split_filepath is None else split_filepath
        self.pair_filepath = f'data/dtu/dtu_pairs.txt' if pair_filepath is None else pair_filepath

        print(colored("loading all scenes together", 'red'))
        with open(self.split_filepath) as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        self.all_intrinsics = []  # the cam info of the whole scene
        self.all_extrinsics = []
        self.all_near_fars = []

        self.metas, self.ref_src_pairs = self.build_metas()  # load ref-srcs view pairs info of the scene

        self.allview_ids = [i for i in range(self.num_all_imgs)]

        self.load_cam_info()  # load camera info of DTU, and estimate scale_mat

        self.build_remap()
        self.define_transforms()
        print(f'==> image down scale: {self.downSample}')

        # * bounding box for rendering
        self.bbox_min = np.array([-1.0, -1.0, -1.0])
        self.bbox_max = np.array([1.0, 1.0, 1.0])

        # - used for cost volume regularization
        self.voxel_dims = torch.tensor(vol_dims, dtype=torch.float32)
        self.partial_vol_origin = torch.Tensor([-1., -1., -1.])

    def build_remap(self):
        self.remap = np.zeros(np.max(self.allview_ids) + 1).astype('int')
        for i, item in enumerate(self.allview_ids):
            self.remap[item] = i

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor()])

    def build_metas(self):
        metas = []
        ref_src_pairs = {}
        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3] if 'train' not in self.split else range(7)

        with open(self.pair_filepath) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                ref_src_pairs[ref_view] = src_views

        for light_idx in light_idxs:
            for scan in self.scans:
                with open(self.pair_filepath) as f:
                    num_viewpoint = int(f.readline())
                    # viewpoints (49)
                    for _ in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                        # ! only for validation
                        if len(self.test_ref_views) > 0 and ref_view not in self.test_ref_views:
                            continue

                        metas += [(scan, light_idx, ref_view, src_views)]

        return metas, ref_src_pairs

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = depth_min + float(lines[11].split()[1]) * 192
        self.depth_interval = float(lines[11].split()[1])
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics
        return intrinsics_, extrinsics, [depth_min, depth_max]

    def load_cam_info(self):
        for vid in range(self.num_all_imgs):
            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4  # * the provided intrinsics is 4x downsampled, now keep the same scale with image
            self.all_intrinsics.append(intrinsic)
            self.all_extrinsics.append(extrinsic)
            self.all_near_fars.append(near_far)

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (1200, 1600)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=self.downSample, fy=self.downSample,
                             interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4,
                           interpolation=cv2.INTER_NEAREST)

        return depth, depth_h

    def read_mask(self, filename):
        mask_h = cv2.imread(filename, 0)
        mask_h = cv2.resize(mask_h, None, fx=self.downSample, fy=self.downSample,
                            interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask_h, None, fx=0.25, fy=0.25,
                          interpolation=cv2.INTER_NEAREST)

        mask[mask > 0] = 1  # the masks stored in png are not binary
        mask_h[mask_h > 0] = 1

        return mask, mask_h

    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)
        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, ref_view, src_views = self.metas[idx % len(self.metas)]

        # generalized, load some images at once
        view_ids = [ref_view] + src_views[:self.n_views]
        # * transform from world system to camera system
        w2c_ref = self.all_extrinsics[self.remap[ref_view]]
        w2c_ref_inv = np.linalg.inv(w2c_ref)

        image_perm = 0  # only supervised on reference view

        imgs, depths_h, masks_h = [], [], []  # full size (640, 512)
        intrinsics, w2cs, near_fars = [], [], []  # record proj mats between views
        mask_dilated = None
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                        f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
            depth_filename = os.path.join(self.root_dir,
                                          f'Depths_raw/{scan}/depth_map_{vid:04d}.pfm')
            mask_filename = os.path.join(self.root_dir,
                                         f'Masks/{scan}_train/mask_{vid:04d}.png')

            img = Image.open(img_filename)
            img_wh = np.round(np.array(img.size) * self.downSample).astype('int')
            img = img.resize(img_wh, Image.BILINEAR)

            if os.path.exists(mask_filename) and self.clean_image:
                mask_l, mask_h = self.read_mask(mask_filename)
            else:
                # print(self.split, "don't find mask file", mask_filename)
                mask_h = np.ones([img_wh[1], img_wh[0]])
            masks_h.append(mask_h)

            if i == 0:
                kernel_size = 101  # default 101
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask_dilated = np.float32(cv2.dilate(np.uint8(mask_h * 255), kernel, iterations=1) > 128)

            if self.clean_image:
                img = np.array(img)
                img[mask_h < 0.5] = 0.0

            img = self.transform(img)

            imgs += [img]

            index_mat = self.remap[vid]
            near_fars.append(self.all_near_fars[index_mat])
            intrinsics.append(self.all_intrinsics[index_mat])

            w2cs.append(self.all_extrinsics[index_mat] @ w2c_ref_inv)

            if os.path.exists(depth_filename):  # and i == 0
                depth_l, depth_h = self.read_depth(depth_filename)
                depths_h.append(depth_h)

        # ! estimate scale_mat
        scale_mat, scale_factor = self.cal_scale_mat(img_hw=[img_wh[1], img_wh[0]],
                                                     intrinsics=intrinsics, extrinsics=w2cs,
                                                     near_fars=near_fars, factor=1.1)

        # ! calculate the new w2cs after scaling
        new_near_fars = []
        new_w2cs = []
        new_c2ws = []
        new_affine_mats = []
        new_depths_h = []
        for intrinsic, extrinsic, near_far, depth in zip(intrinsics, w2cs, near_fars, depths_h):
            P = intrinsic @ extrinsic @ scale_mat
            P = P[:3, :4]
            # - should use load_K_Rt_from_P() to obtain c2w
            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)
            affine_mat = np.eye(4)
            affine_mat[:3, :4] = intrinsic[:3, :3] @ w2c[:3, :4]
            new_affine_mats.append(affine_mat)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1

            new_near_fars.append([0.95 * near, 1.05 * far])
            new_depths_h.append(depth * scale_factor)

        imgs = torch.stack(imgs).float()
        depths_h = np.stack(new_depths_h)
        masks_h = np.stack(masks_h)

        affine_mats = np.stack(new_affine_mats)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), np.stack(
            new_near_fars)

        if 'train' in self.split:
            start_idx = 0
        else:
            start_idx = 1

        sample['images'] = imgs  # (V, 3, H, W)
        sample['depths_h'] = torch.from_numpy(depths_h.astype(np.float32))  # (V, H, W)
        sample['masks_h'] = torch.from_numpy(masks_h.astype(np.float32))  # (V, H, W)
        sample['w2cs'] = torch.from_numpy(w2cs.astype(np.float32))  # (V, 4, 4)
        sample['c2ws'] = torch.from_numpy(c2ws.astype(np.float32))  # (V, 4, 4)
        sample['near_fars'] = torch.from_numpy(near_fars.astype(np.float32))  # (V, 2)
        sample['intrinsics'] = torch.from_numpy(intrinsics.astype(np.float32))[:, :3, :3]  # (V, 3, 3)
        sample['view_ids'] = torch.from_numpy(np.array(view_ids))
        sample['affine_mats'] = torch.from_numpy(affine_mats.astype(np.float32))  # ! in world space

        sample['light_idx'] = torch.tensor(light_idx)
        sample['scan'] = scan

        sample['scale_factor'] = torch.tensor(scale_factor)
        sample['img_wh'] = torch.from_numpy(img_wh)
        sample['render_img_idx'] = torch.tensor(image_perm)
        sample['partial_vol_origin'] = torch.tensor(self.partial_vol_origin, dtype=torch.float32)
        sample['meta'] = str(scan) + "_light" + str(light_idx) + "_refview" + str(ref_view)

        # - image to render
        sample['query_image'] = sample['images'][0]
        sample['query_c2w'] = sample['c2ws'][0]
        sample['query_w2c'] = sample['w2cs'][0]
        sample['query_intrinsic'] = sample['intrinsics'][0]
        sample['query_depth'] = sample['depths_h'][0]
        sample['query_mask'] = sample['masks_h'][0]
        sample['query_near_far'] = sample['near_fars'][0]

        sample['images'] = sample['images'][start_idx:]  # (V, 3, H, W)
        sample['depths_h'] = sample['depths_h'][start_idx:]  # (V, H, W)
        sample['masks_h'] = sample['masks_h'][start_idx:]  # (V, H, W)
        sample['w2cs'] = sample['w2cs'][start_idx:]  # (V, 4, 4)
        sample['c2ws'] = sample['c2ws'][start_idx:]  # (V, 4, 4)
        sample['intrinsics'] = sample['intrinsics'][start_idx:]  # (V, 3, 3)
        sample['view_ids'] = sample['view_ids'][start_idx:]
        sample['affine_mats'] = sample['affine_mats'][start_idx:]  # ! in world space

        sample['scale_mat'] = torch.from_numpy(scale_mat)
        sample['trans_mat'] = torch.from_numpy(w2c_ref_inv)

        # - generate rays
        if ('val' in self.split) or ('test' in self.split):
            sample_rays = gen_rays_from_single_image(
                img_wh[1], img_wh[0],
                sample['query_image'],
                sample['query_intrinsic'],
                sample['query_c2w'],
                depth=sample['query_depth'],
                mask=sample['query_mask'] if self.clean_image else None)
        else:
            sample_rays = gen_random_rays_from_single_image(
                img_wh[1], img_wh[0],
                self.N_rays,
                sample['query_image'],
                sample['query_intrinsic'],
                sample['query_c2w'],
                depth=sample['query_depth'],
                mask=sample['query_mask'] if self.clean_image else None,
                dilated_mask=mask_dilated,
                importance_sample=self.importance_sample)

        sample['rays'] = sample_rays

        return sample
