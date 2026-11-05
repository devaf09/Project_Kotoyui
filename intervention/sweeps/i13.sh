#!/bin/bash
# 実験I13（CPU だけ）: bash intervention/sweeps/i13.sh 0
export CUDA_VISIBLE_DEVICES=
python intervention/i13_fusion.py --seed $1
