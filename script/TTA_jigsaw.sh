#!/bin/bash


python main.py \
  --img-size=225 \
  --jigsaw \
  --dataset='MDN' \
  --save-path='AdaBN/resnet18_jigsaw_MDN2' \
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  \
  --TTA-head='jigsaw' \
  --model='tta_model' \
  --backbone='resnet18' \
  --batch-size=128 \
  --num-epoch=30 \
  \
  --exp-num=-2 \
  --start-time=1 \
  --times=4 \
  --train='deepall' \
  --eval='deepall' \
  --loader='normal'