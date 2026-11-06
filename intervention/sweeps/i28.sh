#!/bin/bash
# 実験I28（CPU だけ、1 回 1 分ほど）: bash intervention/sweeps/i28.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2 3 4 5; do python intervention/i28_route_board.py --seed $s & done; wait
for s in 0 1 2 3 4 5; do python intervention/i28_route_board.py --seed $s --no_free & done; wait
