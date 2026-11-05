#!/bin/bash
# ツッコミどころ T1（CPU だけ）: bash intervention/sweeps/t1.sh 0
export CUDA_VISIBLE_DEVICES=
for n in 10 50; do python intervention/t1_ablation.py --seed $1 --n_train $n; done
