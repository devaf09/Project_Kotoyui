#!/bin/bash
# 実験I6（CPU だけ）。seed ごとに 1 本: bash intervention/sweeps/i06.sh 0
s=$1
export CUDA_VISIBLE_DEVICES=
for K in 12 24; do
  for P in 0 2 6; do
    for n in 10 50; do
      python intervention/i06_revise.py --K $K --P $P --n_train $n --seed $s
    done
  done
done
