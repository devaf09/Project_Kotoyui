#!/bin/bash
# 実験I7（CPU だけ）。seed ごとに 1 本: bash intervention/sweeps/i07.sh 0
s=$1
export CUDA_VISIBLE_DEVICES=
for PA in 2 0; do
  for nB in 10 50; do
    python intervention/i07_experience.py --seed $s --PA $PA --nB $nB
  done
done
