#!/bin/bash
# 実験I21（CPU だけ、1 回 30 秒）: bash intervention/sweeps/i21.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2; do python intervention/i21_board_alu.py --seed $s & done; wait
for s in 3 4 5; do python intervention/i21_board_alu.py --seed $s & done; wait
