#!/bin/bash
# 実験I5（CPU だけ）。seed ごとに 1 本: bash intervention/sweeps/i05.sh 0
s=$1
export CUDA_VISIBLE_DEVICES=
for K in 3 6 12 24; do
  for n in 10 50; do
    python intervention/i05_more_words.py --K $K --n_train $n --seed $s
  done
done
