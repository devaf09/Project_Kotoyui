#!/bin/bash
# 実験I9（CPU だけ、1 seed 40 分ほど）: bash intervention/sweeps/i09.sh 0
export CUDA_VISIBLE_DEVICES=
python intervention/i09_feedback.py --seed $1
python intervention/i09_feedback.py --seed $1 --aggregate 1
python intervention/i09_feedback.py --seed $1 --tie unknown
