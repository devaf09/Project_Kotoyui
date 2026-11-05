#!/bin/bash
# 実験I11（CPU だけ）: bash intervention/sweeps/i11.sh 0
export CUDA_VISIBLE_DEVICES=
for n in 10 3; do python intervention/i11_calibrate.py --seed $1 --n $n; done
