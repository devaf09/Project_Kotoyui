#!/bin/bash
# 実験I14（CPU だけ）: bash intervention/sweeps/i14.sh 0
export CUDA_VISIBLE_DEVICES=
for n in 50 10; do python intervention/i14_light_diff.py --seed $1 --n_train $n; done
