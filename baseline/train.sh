#!/bin/bash

# Deeplabv3plus_res101  PSPNet_res101  DualSeg_res101  BiSeNet  BiSeNetV2  DDRNet
# FCN_ResNet  SegTrans

python -m torch.distributed.launch --nproc_per_node=1 \
                train.py --out_stride 8 \
                --max_epochs 50 --val_epochs 5 --batch_size 16 --lr 5e-4
