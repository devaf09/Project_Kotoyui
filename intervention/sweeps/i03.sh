#!/bin/bash
# 実験I3（CPU だけ）。seed ごとに 1 本: bash intervention/sweeps/i03.sh 0
s=$1
P="python intervention/i03_reverse_split.py --seed $s --out intervention/results/i03.jsonl"
export CUDA_VISIBLE_DEVICES=
for n in 50 10; do
  $P --n_train $n
  $P --n_train $n --min_support 0
  $P --n_train $n --min_support 6
done
# 反例の強さの下限を変える（10 文）
for mr in 2 3; do
  $P --n_train 10 --min_ref $mr
  $P --n_train 10 --min_support 0 --min_ref $mr
done
# 板どうしの組を緩める（色と味だけ分かっていればよい）
for n in 50 10; do
  $P --n_train $n --loose 1
  $P --n_train $n --loose 1 --min_support 0
done
