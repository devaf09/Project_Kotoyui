#!/bin/bash
# 実験I18（CPU だけ）: bash intervention/sweeps/i18.sh
# A（足し算、1 回 20 秒）と B（小さな世界、1 回 5〜15 分）。B は 3 本ずつ並べて回す。
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2 3 4 5; do for n in 10 3; do python intervention/i18_combo.py --part A --seed $s --n_sent $n; done; done
for n in 10 50; do
  for s in 0 1 2; do python intervention/i18_combo.py --part B --seed $s --n_train $n & done; wait
  for s in 3 4 5; do python intervention/i18_combo.py --part B --seed $s --n_train $n & done; wait
done
