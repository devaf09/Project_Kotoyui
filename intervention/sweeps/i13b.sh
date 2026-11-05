#!/bin/bash
# 実験I13b・I13c（CPU だけ）: bash intervention/sweeps/i13b.sh 0
export CUDA_VISIBLE_DEVICES=
for r in 1.5 4; do python intervention/i13b_erase.py --seed $1 --bad_ratio $r; done
