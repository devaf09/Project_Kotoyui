#!/bin/bash
# 実験I4（CPU だけ）。seed ごとに 1 本: bash intervention/sweeps/i04.sh 0
s=$1
export CUDA_VISIBLE_DEVICES=
for w in small large; do
  for n in 2 3 5 10 20 50; do
    python intervention/i04_scale.py --world $w --n_train $n --seed $s
  done
done
