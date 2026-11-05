#!/bin/bash
# 実験I17（CPU だけ、1 回 10 秒）: bash intervention/sweeps/i17.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2 3 4 5; do for n in 10 3; do python intervention/i17_arithmetic.py --seed $s --n_sent $n; done; done
