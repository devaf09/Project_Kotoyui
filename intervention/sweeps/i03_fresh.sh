#!/bin/bash
# I3 の最終形と I1 の既定を、設計に使っていない seed 6〜11 で確かめる（CPU だけ）: bash intervention/sweeps/i03_fresh.sh 6
s=$1
export CUDA_VISIBLE_DEVICES=
for n in 50 10; do
  python intervention/i03_reverse_split.py --seed $s --n_train $n --loose 1 --min_support 0 --out intervention/results/i03_fresh.jsonl
  python intervention/i01_small_world.py --seed $s --n_train $n --baselines 0 --out intervention/results/i01_fresh.jsonl
done
