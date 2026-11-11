#!/bin/bash
# 実験I22（CPU だけ、1 回 20 秒）: bash intervention/sweeps/i22.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2; do python intervention/i22_carry.py --seed $s & done; wait
for s in 3 4 5; do python intervention/i22_carry.py --seed $s & done; wait
