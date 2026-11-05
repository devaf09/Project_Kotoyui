#!/bin/bash
# 実験I10（CPU だけ）: bash intervention/sweeps/i10.sh 0
export CUDA_VISIBLE_DEVICES=
for n in 10 3; do python intervention/i10_hierarchy.py --seed $1 --n $n; done
