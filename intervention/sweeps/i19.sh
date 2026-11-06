#!/bin/bash
# 実験I19（CPU だけ、1 回 40 秒）: bash intervention/sweeps/i19.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2; do python intervention/i19_grain_arith.py --seed $s & done; wait
for s in 3 4 5; do python intervention/i19_grain_arith.py --seed $s & done; wait
