#!/bin/bash
# 実験I1 の全部（CPU だけ）。seed ごとに 1 本: bash intervention/sweeps/i01.sh 0
# 比べる相手（GRU・トランスフォーマー）は語板の設定に関係しないので、1 回目に回した results/i01_baselines.jsonl を使う（--baselines 1 で回し直せる）
s=$1
P="python intervention/i01_small_world.py --seed $s --out ../Project_Kotoyui_private/intervention/results/i01.jsonl"
export CUDA_VISIBLE_DEVICES=
for n in 50 10; do
  $P --n_train $n --baselines 0
  $P --n_train $n --mode digital --baselines 0
  $P --n_train $n --split 0 --baselines 0
  $P --n_train $n --consolidate 0 --baselines 0
  $P --n_train $n --reread 0 --baselines 0
  $P --n_train $n --split 0 --consolidate 0 --reread 0 --baselines 0
  $P --n_train $n --theta 0.67 --baselines 0
  $P --n_train $n --theta 1.0 --baselines 0
  $P --n_train $n --theta 0.34 --baselines 0
done
