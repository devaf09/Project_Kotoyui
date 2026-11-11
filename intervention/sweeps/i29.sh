#!/bin/bash
# 実験I29（CPU だけ、1 回 10 秒ほど）: bash intervention/sweeps/i29.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2 3 4 5; do python intervention/i29_fix_memory.py --seed $s & done; wait
for s in 0 1 2 3 4 5; do python intervention/i29_fix_memory.py --exp b --seed $s & done; wait   # I29b: 返ってくる答えにも誤り
for s in 0 1 2 3 4 5; do python intervention/i29c_which_is_wrong.py --seed $s & done; wait   # I29c: 手順と記憶の両方が誤っている
