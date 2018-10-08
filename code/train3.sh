#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 main.py --model khu3 --save khu3 --scale 2 --n_resblocks 12 --n_feats 28 --epochs 180 --batch_size 162 --lr 1.3e-4&&
CUDA_VISIBLE_DEVICES=2 python3 main.py --model khu3 --save khu3 --scale 3 --n_resblocks 12 --n_feats 28 --epochs 180 --batch_size 162 --lr 1.3e-4 &&
CUDA_VISIBLE_DEVICES=2 python3 main.py --model khu3 --save khu3 --scale 4 --n_resblocks 12 --n_feats 28 --epochs 180 --batch_size 162 --lr 1.3e-4 
