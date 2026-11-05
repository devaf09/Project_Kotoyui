#!/bin/bash
# 実験I16（CPU だけ、1 回 15 秒）: bash intervention/sweeps/i16.sh
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2 3 4 5; do python intervention/i16_attention_real.py --seed $s; done
