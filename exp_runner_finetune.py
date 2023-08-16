"""
for fine-tuning
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import logging
import numpy as np
import cv2 as cv
import trimesh
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory, HOCONConverter

from models.fields import SingleVarianceNetwork

from models.featurenet import FeatureNet

from models.trainer_finetune import FinetuneTrainer

from models.sparse_neus_renderer import SparseNeuSRenderer

from models.sparse_sdf_network import SparseSdfNetwork, FinetuneOctreeSdfNetwork

from data.dtu_fit import DtuFit
# from data.bmvs import BMVS

from utils.training_utils import tocuda

from termcolor import colored

from datetime import datetime


class Runner:
    def __init__(self, conf_path, mode='train', is_continue=False,
                 is_finetune=False, train_from_scratch=False,
                 local_rank=0, checkpoint_path=None, CASE_NAME=None, train_imgs_idx=None, test_imgs_idx=None,
                 timestamp='latest',
                 visibility_beta=0.015, visibility_gama=0.010,
                 visibility_penalize_ratio=0.8, visibility_weight_thred=[0.7],
                 dataset_near=425, dataset_far=900, clip_wh=[0, 0]):

        # Initial setting
        self.device = torch.device('cuda')
        self.num_devices = torch.cuda.device_count()

        self.logger = logging.getLogger('exp_logger')

        print(colored("detected %d GPUs" % self.num_devices, "red"))

        self.conf_path = conf_path
        self.conf = ConfigFactory.parse_file(conf_path)

        # modify the config
        imgs_idx_string = ''
        for img_idx in train_imgs_idx:
            imgs_idx_string += '_'
            imgs_idx_string += str(img_idx)

        ############### - modify the config file ###########
        self.conf['general']['base_exp_dir'] = self.conf['general']['base_exp_dir'].replace(
            "CASE_NAME", CASE_NAME) + "_imgs" + imgs_idx_string
        self.conf['dataset']['test_scan_id'] = self.conf['dataset']['test_scan_id'].replace("CASE_NAME", CASE_NAME)
        self.conf['dataset']['train_img_idx'] = train_imgs_idx
        self.conf['dataset']['test_img_idx'] = test_imgs_idx
        self.conf['dataset']['near'] = dataset_near
        self.conf['dataset']['far'] = dataset_far
        self.conf['dataset']['test_clip_wh'] = clip_wh
        self.conf['train']['visibility_beta'] = visibility_beta
        self.conf['train']['visibility_gama'] = visibility_gama
        self.conf['train']['visibility_penalize_ratio'] = visibility_penalize_ratio
        self.conf['train']['visibility_weight_thred'] = visibility_weight_thred

        if is_continue and timestamp == 'latest':
            if os.path.exists(self.conf['general.base_exp_dir']):
                timestamps = os.listdir(self.conf['general.base_exp_dir'])
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            is_continue = False
            timestamp = None
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now()) if timestamp is None else timestamp
        self.conf['general']['base_exp_dir'] = os.path.join(self.conf['general']['base_exp_dir'], self.timestamp)

        self.base_exp_dir = self.conf['general.base_exp_dir']
        print(colored("base_exp_dir:  " + self.base_exp_dir, 'yellow'))
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0
        self.val_step = 0

        # trainning parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.num_devices
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_milestone = self.conf.get_list('train.learning_rate_milestone')
        self.learning_rate_factor = self.conf.get_float('train.learning_rate_factor')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        self.N_rays = self.conf.get_int('train.N_rays')

        # neural networks
        self.is_continue = is_continue
        self.is_finetune = is_finetune
        self.train_from_scratch = train_from_scratch
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.finetune_lod = self.conf.get_int('model.finetune_lod')

        self.rendering_network_outside = None
        self.sdf_network_lod0 = None
        self.sdf_network_lod1 = None
        self.sdf_network_finetune = None

        self.variance_network_lod0 = None
        self.variance_network_lod1 = None

        self.pyramid_feature_network = None  # extract 2d pyramid feature maps from images, used for geometry
        self.pyramid_feature_network_lod1 = None  # may use different feature network for different lod

        # * pyramid_feature_network
        self.pyramid_feature_network = FeatureNet().to(self.device)

        self.sdf_network_lod0 = SparseSdfNetwork(**self.conf['model.sdf_network_lod0']).to(
            self.device)

        self.variance_network_lod0 = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)

        if self.finetune_lod == 1:
            self.pyramid_feature_network_lod1 = FeatureNet().to(self.device)

            self.sdf_network_lod1 = SparseSdfNetwork(**self.conf['model.sdf_network_lod1']).to(
                self.device)

            self.variance_network_lod1 = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)

        self.sdf_network_finetune = FinetuneOctreeSdfNetwork(**self.conf['model.sdf_network_finetune'])

        # Load checkpoint
        latest_model_name = None
        if checkpoint_path is None:
            if is_continue or (is_finetune and not train_from_scratch):
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name.startswith('ckpt'):
                        if model_name[-3:] == 'pth':  # and int(model_name[5:-4]) <= self.end_iter:
                            model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]
                latest_model_name = os.path.join(self.base_exp_dir, 'checkpoints', latest_model_name)
        else:
            latest_model_name = checkpoint_path

        if latest_model_name is not None and self.is_finetune:
            self.logger.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Renderer model
        self.trainer = FinetuneTrainer(
            self.rendering_network_outside,
            self.pyramid_feature_network,
            self.pyramid_feature_network_lod1,
            self.sdf_network_lod0,
            self.sdf_network_lod1,
            self.variance_network_lod0,
            self.variance_network_lod1,
            self.sdf_network_finetune,
            self.finetune_lod,
            **self.conf['model.trainer'],
            conf=self.conf)

        if latest_model_name is not None and self.is_continue:
            self.logger.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        self.data_setup()  # * data setup

        # * initialize finetune network
        if not self.is_continue:
            self.initialize_network()

        self.optimizer_setup()

        self.trainer = torch.nn.DataParallel(self.trainer).to(self.device)

        if self.mode[:5] == 'train':
            self.file_backup()

    def optimizer_setup(self):

        params_to_train = self.trainer.get_trainable_params()
        params = params_to_train['params']
        faster_params = params_to_train['faster_params']
        slower_params = params_to_train['slower_params']

        self.params_to_train = params + faster_params + slower_params
        self.optimizer = torch.optim.Adam([
            {'params': slower_params, 'lr': self.learning_rate / 2.},
            {'params': params, 'lr': self.learning_rate / 2.},
            {'params': faster_params}
        ],
            lr=self.learning_rate)

    def data_setup(self):
        """
        if use ddp, use setup() not prepare_data(),
        prepare_data() only called on 1 GPU/TPU in distributed
        :return:
        """
        dataset = DtuFit

        self.train_dataset = dataset(
            root_dir=self.conf['dataset.testpath'], split='train',
            N_rays=self.conf.get_int('train.N_rays'),
            scan_id=self.conf['dataset.test_scan_id'],
            n_views=self.conf.get_int('dataset.test_n_views'),
            img_wh=self.conf['dataset.test_img_wh'],
            clip_wh=self.conf['dataset.test_clip_wh'],
            train_img_idx=self.conf.get_list('dataset.train_img_idx', default=[]),
            test_img_idx=self.conf.get_list('dataset.test_img_idx', default=[]),
            h_patch_size=self.conf.get_int('model.h_patch_size', default=5),
            near=self.conf.get_float('dataset.near'),
            far=self.conf.get_float('dataset.far')
        )

        self.test_dataset = dataset(
            root_dir=self.conf['dataset.testpath'],
            split='test',
            scan_id=self.conf['dataset.test_scan_id'],
            N_rays=self.conf.get_int('train.N_rays'),
            img_wh=self.conf['dataset.test_img_wh'],
            clip_wh=self.conf['dataset.test_clip_wh'],
            n_views=self.conf.get_int('dataset.test_n_views'),
            train_img_idx=self.conf.get_list('dataset.train_img_idx', default=[]),
            test_img_idx=self.conf.get_list('dataset.test_img_idx', default=[]),
            near=self.conf.get_float('dataset.near'),
            far=self.conf.get_float('dataset.far')
        )

        self.train_dataloader = DataLoader(self.train_dataset,
                                           shuffle=True,
                                           num_workers=4 * self.batch_size,
                                           batch_size=self.batch_size,
                                           pin_memory=True,
                                           drop_last=True
                                           )

        self.test_dataloader = DataLoader(self.test_dataset,
                                          shuffle=False,
                                          num_workers=4 * self.batch_size,
                                          batch_size=self.batch_size,
                                          pin_memory=True,
                                          drop_last=True
                                          )
        self.test_dataloader_iterator = iter(self.test_dataloader)

    def initialize_network(self):
        sample = self.train_dataset.get_conditional_sample()
        sample = tocuda(sample, self.device)

        self.trainer.initialize_finetune_network(sample, train_from_scratch=self.train_from_scratch)

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        res_step = self.end_iter - self.iter_step

        dataloader = self.train_dataloader

        epochs = int(1 + res_step // len(dataloader))

        self.adjust_learning_rate()
        print(colored("starting training learning rate: {:.5f}".format(self.optimizer.param_groups[0]['lr']), "yellow"))

        background_rgb = None
        if self.use_white_bkgd:
            background_rgb = torch.ones([1, 3]).to(self.device)

        for epoch_i in range(epochs):

            print(colored("current epoch %d" % epoch_i, 'red'))
            dataloader = tqdm(dataloader)

            for batch in dataloader:

                batch['batch_idx'] = torch.tensor([x for x in range(self.batch_size)])  # used to get meta

                losses = self.trainer(
                    batch,
                    background_rgb=background_rgb,
                    iter_step=self.iter_step,
                    mode='train',
                )

                loss = losses['loss_lod0']
                losses_lod0 = losses['losses_lod0']

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params_to_train, 1.0)
                self.optimizer.step()

                self.iter_step += 1

                if self.iter_step % self.report_freq == 0:
                    self.writer.add_scalar('Loss/loss', loss, self.iter_step)

                    if losses_lod0 is not None:
                        self.writer.add_scalar('Loss/sparse_loss',
                                               losses_lod0[
                                                   'sparse_loss'].mean() if losses_lod0 is not None else 0,
                                               self.iter_step)
                        self.writer.add_scalar('Loss/color_loss',
                                               losses_lod0['color_mlp_loss'].mean()
                                               if losses_lod0['color_mlp_loss'] is not None else 0,
                                               self.iter_step)
                        self.writer.add_scalar('statis/psnr',
                                               losses_lod0['psnr_mlp'].mean()
                                               if losses_lod0['psnr_mlp'] is not None else 0,
                                               self.iter_step)

                    print(self.base_exp_dir)
                    self.logger.info(
                        'iter:{:8>d} '
                        'loss = {:.4f} '
                        'color_loss = {:.4f} '
                        'sparse_loss= {:.4f} '
                        'color_patch_loss = {:.4f} '
                        'mask_loss = {:.4f}'
                        '  lr = {:.5f}'.format(
                            self.iter_step, loss,
                            losses_lod0['color_mlp_loss'].mean() ,
                            losses_lod0['sparse_loss'].mean(),
                            losses_lod0['color_patch_loss'].mean(),
                            losses_lod0['mask_loss'].mean(),
                            self.optimizer.param_groups[0]['lr']))

                    if losses_lod0 is not None:
                        self.logger.info(
                            'iter:{:8>d} '
                            'weights_sum = {:.4f} '
                            'alpha_sum = {:.4f} '
                            'sparse_weight= {:.4f} '
                                .format(
                                self.iter_step,
                                losses_lod0['weights_sum'].mean(),
                                losses_lod0['alpha_sum'].mean(),
                                losses_lod0['sparse_weight'].mean(),
                            ))

                    ic(losses_lod0['variance'])

                if self.iter_step % self.save_freq == 0 and self.iter_step > 5000:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    self.validate()

                # - ajust learning rate
                self.adjust_learning_rate()

    def adjust_learning_rate(self):

        # the geometric part slow training in the early stage
        warmup_start = 500

        end = self.end_iter * 0.9
        if self.iter_step < warmup_start:
            learning_factor_slow = 0.
        else:
            alpha = 0.5
            progress = np.min([self.iter_step / end, 1.0])
            learning_factor_slow = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        # the rendering part fast training
        alpha = 0.5
        progress = np.min([self.iter_step / end, 1.0])
        learning_factor_fast = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        gs = self.optimizer.param_groups

        gs[0]['lr'] = self.learning_rate * learning_factor_slow * 0.5
        gs[1]['lr'] = self.learning_rate * learning_factor_slow * 0.5
        gs[2]['lr'] = self.learning_rate * learning_factor_fast

    def get_alpha_inter_ratio(self, start, end):
        if self.is_finetune and not self.train_from_scratch:
            return 1.0
        if end == 0.0:
            return 1.0
        elif self.iter_step < start:
            return 0.0
        else:
            return np.min([1.0, (self.iter_step - start) / (end - start)])

    def file_backup(self):
        # copy python file
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        # export config file
        with open(os.path.join(self.base_exp_dir, 'recording', 'config.conf'), "w") as fd:
            res = HOCONConverter.to_hocon(self.conf)
            fd.write(res)

    def load_checkpoint(self, checkpoint_name):

        def load_state_dict(network, checkpoint, comment):
            if network is not None:
                try:
                    pretrained_dict = checkpoint[comment]
                    model_dict = network.state_dict()
                    # 1. filter out unnecessary keys
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    # 2. overwrite entries in the existing state dict
                    model_dict.update(pretrained_dict)
                    # 3. load the new state dict
                    network.load_state_dict(pretrained_dict)
                except:
                    print(colored(comment + " load fails", 'yellow'))

        checkpoint = torch.load(checkpoint_name,
                                map_location=self.device)

        load_state_dict(self.rendering_network_outside, checkpoint, 'rendering_network_outside')

        load_state_dict(self.sdf_network_lod0, checkpoint, 'sdf_network_lod0')
        load_state_dict(self.sdf_network_lod1, checkpoint, 'sdf_network_lod1')

        if self.is_finetune:
            if self.finetune_lod == 0:
                load_state_dict(self.sdf_network_finetune, checkpoint, 'sdf_network_lod0')
            else:
                load_state_dict(self.sdf_network_finetune, checkpoint, 'sdf_network_lod1')

        if self.is_continue:
            load_state_dict(self.sdf_network_finetune, checkpoint, 'sdf_network_finetune')
            sparse_con_volume = checkpoint['sdf_network_finetune']['sparse_volume_lod0.volume']
            sparse_coords_volume = checkpoint['sdf_network_finetune']['sparse_coords_lod0']
            self.trainer.initialize_finetune_network(None, sparse_con_volume, sparse_coords_volume)

        load_state_dict(self.pyramid_feature_network, checkpoint, 'pyramid_feature_network')
        load_state_dict(self.pyramid_feature_network_lod1, checkpoint, 'pyramid_feature_network_lod1')

        load_state_dict(self.variance_network_lod0, checkpoint, 'variance_network_lod0')
        load_state_dict(self.variance_network_lod1, checkpoint, 'variance_network_lod1')

        if self.is_continue:
            self.iter_step = checkpoint['iter_step']
            self.val_step = checkpoint['val_step'] if 'val_step' in checkpoint.keys() else 0

        self.logger.info('End')

    def load_optimizer(self, checkpoint_name):
        checkpoint = torch.load(checkpoint_name,map_location=self.device)
        if self.is_continue:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self):

        def save_state_dict(network, checkpoint, comment):
            if network is not None:
                checkpoint[comment] = network.state_dict()

        checkpoint = {
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'val_step': self.val_step,
        }

        save_state_dict(self.sdf_network_lod0, checkpoint, "sdf_network_lod0")
        save_state_dict(self.sdf_network_lod1, checkpoint, "sdf_network_lod1")

        save_state_dict(self.rendering_network_outside, checkpoint, 'rendering_network_outside')
        save_state_dict(self.variance_network_lod0, checkpoint, 'variance_network_lod0')
        save_state_dict(self.variance_network_lod1, checkpoint, 'variance_network_lod1')

        save_state_dict(self.sdf_network_finetune, checkpoint, 'sdf_network_finetune')
        save_state_dict(self.pyramid_feature_network, checkpoint, 'pyramid_feature_network')
        save_state_dict(self.pyramid_feature_network_lod1, checkpoint, 'pyramid_feature_network_lod1')

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate(self, idx=-1, resolution_level=-1):
        # validate image

        ic(self.iter_step, idx)
        self.logger.info('Validate begin')

        if idx < 0:
            idx = self.val_step
        self.val_step += 1
        try:
            batch = next(self.test_dataloader_iterator)
        except:
            self.test_dataloader_iterator = iter(self.test_dataloader)  # reset
            batch = next(self.test_dataloader_iterator)

        background_rgb = None
        if self.use_white_bkgd:
            background_rgb = torch.ones([1, 3]).to(self.device)

        batch['batch_idx'] = torch.tensor([x for x in range(self.batch_size)])

        self.trainer(
            batch,
            background_rgb=background_rgb,
            iter_step=self.iter_step,
            save_vis=True,
            mode='val',
        )

    def gen_video(self):

        batch = self.test_dataloader_iterator.next()

        background_rgb = torch.ones([1, 3]).to(self.device)

        batch['batch_idx'] = torch.tensor([x for x in range(self.batch_size)])

        batch = tocuda(batch, self.device)

        self.trainer.module.gen_video(
            batch,
            background_rgb=background_rgb
        )


if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--is_finetune', default=False, action="store_true")
    parser.add_argument('--train_from_scratch', default=False, action="store_true")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--case_name', type=str, help='the input dtu scan')
    parser.add_argument('--train_imgs_idx', nargs='+', help='the input images idx')
    parser.add_argument('--test_imgs_idx', nargs='+', help='the input images idx')
    parser.add_argument('--checkpoint_path', type=str, help='the pretrained checkpoint of general model')
    ## perscene finetuning params
    parser.add_argument('--visibility_beta', type=float, default=0.015, help='used in occlusion-aware patch loss')
    parser.add_argument('--visibility_gama', type=float, default=0.010, help='used in occlusion-aware patch loss')
    parser.add_argument('--visibility_penalize_ratio', type=float, default=0.8,
                        help='used in occlusion-aware patch loss')
    parser.add_argument('--visibility_weight_thred', nargs='+', default=[0.7],
                        help='visibility_weight_thred')
    parser.add_argument('--near', type=float, default=425)
    parser.add_argument('--far', type=float, default=900)
    parser.add_argument('--clip_wh', nargs='+', help='clip image width and height', default=[0, 0])

    args = parser.parse_args()

    # torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True  # ! make training 2x faster

    runner = Runner(args.conf, args.mode, args.is_continue,
                    args.is_finetune, args.train_from_scratch,
                    args.local_rank, CASE_NAME=args.case_name,
                    checkpoint_path=args.checkpoint_path,
                    train_imgs_idx=[int(x) for x in args.train_imgs_idx],
                    test_imgs_idx=[int(x) for x in args.test_imgs_idx],
                    visibility_beta=args.visibility_beta,
                    visibility_gama=args.visibility_gama,
                    visibility_penalize_ratio=args.visibility_penalize_ratio,
                    visibility_weight_thred=[float(x) for x in args.visibility_weight_thred],
                    dataset_near=args.near,
                    dataset_far=args.far,
                    clip_wh=[int(x) for x in args.clip_wh]
                    )

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'test' or args.mode == 'val':
        runner.validate()
    elif args.mode == 'gen_video':
        runner.gen_video()
