#!/bin/bash

python main.py \
  --dataset='PACS' \
  --save-path='pretrained_models/resnet18_PACS' \
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  \
  --model='erm' \
  --backbone='resnet18' \
  --batch-size=128 \
  --num-epoch=30 \
  \
  --exp-num=-2 \
  --start-time=0 \
  --times=5 \
  --train='deepall' \
  --eval='deepall' \
  --loader='normal' \
  --eval-step=1 \
  --scheduler='step' \
  --lr-decay-gamma=0.1 \
