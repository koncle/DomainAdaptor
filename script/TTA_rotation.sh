#!/bin/bash
# OfficeHome, PACS, VLCS, MiniDomainNet

python main.py \
  --rot \
  --dataset='PACS' \
  --save-path='pretrained_models/resnet18_rot_PACS' \
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  \
  --model='tta_model' \
  --TTA-head='rot' \
  --batch-size=128 \
  --num-epoch=30 \
  \
  --exp-num=-2 \
  --start-time=1 \
  --times=4 \
  --fc-weight=10.0 \
  --train='deepall' \
  --eval='deepall' \
  --loader='normal' \
