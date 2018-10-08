#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model khu1 --save khu1 --scale 2 --n_resblocks 12 --n_feats 32 --epochs 150 --batch_size 384 --lr 1.2e-4 --epochs 120&&
CUDA_VISIBLE_DEVICES=0 python3 main.py --model khu1 --save khu1 --scale 3 --n_resblocks 12 --n_feats 32 --epochs 150 --batch_size 384 --lr 1.2e-4 --epochs 120&&
CUDA_VISIBLE_DEVICES=0 python3 main.py --model khu1 --save khu1 --scale 4 --n_resblocks 12 --n_feats 32 --epochs 150 --batch_size 384 --lr 1.2e-4 --epochs 120
