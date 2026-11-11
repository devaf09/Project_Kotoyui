#!/bin/bash
# 実験I23（CPU だけ、1 回 3 分。--order 組 は 30 分以上）: bash intervention/sweeps/i23.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2; do python intervention/i23_digits.py --seed $s & done; wait
for s in 3 4 5; do python intervention/i23_digits.py --seed $s & done; wait
