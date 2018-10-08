#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python3 main.py --model khu2 --save khu2 --scale 2 --n_resblocks 12 --n_feats 16 --epochs 250 --batch_size 128 --lr 1e-4 --epochs 200&&
CUDA_VISIBLE_DEVICES=1 python3 main.py --model khu2 --save khu2 --scale 3 --n_resblocks 12 --n_feats 16 --epochs 250 --batch_size 128 --lr 1e-4 --epochs 200&&
CUDA_VISIBLE_DEVICES=1 python3 main.py --model khu2 --save khu2 --scale 4 --n_resblocks 12 --n_feats 16 --epochs 250 --batch_size 128 --lr 1e-4 --epochs 200
