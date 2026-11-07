#!/bin/bash
# 実験I20（CPU だけ、1 回 3 分）: bash intervention/sweeps/i20.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2; do python intervention/i20_judge_board.py --seed $s & done; wait
for s in 3 4 5; do python intervention/i20_judge_board.py --seed $s & done; wait
