#!/usr/bin/env bash
python exp_runner_finetune.py \
--mode train --conf ./confs/finetune.conf --is_finetune \
--checkpoint_path ./weights/ckpt.pth \
--case_name scan118  --train_imgs_idx 0 1 2 --test_imgs_idx 0 1 2 --near 700 --far 1100 \
--visibility_beta 0.015 --visibility_gama 0.010 --visibility_weight_thred 0.7