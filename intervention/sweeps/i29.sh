#!/bin/bash
# 実験I29（CPU だけ、1 回 10 秒ほど）: bash intervention/sweeps/i29.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2 3 4 5; do python intervention/i29_fix_memory.py --seed $s & done; wait
