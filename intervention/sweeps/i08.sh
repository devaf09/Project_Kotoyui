#!/bin/bash
# 実験I8（CPU だけ、1 seed 30 分ほど）: bash intervention/sweeps/i08.sh 0
export CUDA_VISIBLE_DEVICES=
python intervention/i08_light_judge.py --seed $1
