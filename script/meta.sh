#!/bin/bash


python main.py \
  --dataset='PACS' \
  --save-path='AdaBN/meta_norm_PACS' \
  --gpu=0 \
  --do-train=True \
  --meta-lr=0.1 \
  --lr=1e-3 \
  \
  --replace \
  --meta-step=1 \
  --meta-second-order=False \
  --TTA-head='norm' \
  --model='DomainAdaptor' \
  --backbone='resnet50' \
  --batch-size=64 \
  --num-epoch=30 \
  \
  --exp-num -2 \
  --start-time=0 \
  --times=5 \
  --fc-weight=10.0 \
  --train='tta_meta' \
  --eval='tta_meta' \
  --loader='meta' \
