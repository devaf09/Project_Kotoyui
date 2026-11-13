#!/bin/bash
# 実験18a: 光の 2DTG を大きくする梯子（その 1）。1 枚の GPU で 2 本ずつ同時に回す（持ち主の依頼 2026-11-03、どれも GPU のメモリは 3GB ほど）。
# Project_Kotoyui の直下から回る。見込み: 約 50 分（長い方: 6×8・18000 ステップ + 人格の器）
export PYTHONIOENCODING=utf-8
P="python experiments/exp18_scale_gen.py --device cuda"
$P --kind text4f --stages 6 --width 8 --steps 18000 --ft_steps 2000 --tag t6x8_18k &
( $P --kind split4f --G 8 --stages 4 --width 4 --steps 9000 --tag s8_4x4
  $P --kind gru --hidden 512 --steps 18000 --tag gru512
  $P --kind gru --hidden 1024 --steps 18000 --tag gru1024 ) &
wait
