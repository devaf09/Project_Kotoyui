#!/bin/bash
# 実験18b: 光の 2DTG を大きくする梯子（その 2）。1 枚の GPU で 2 本同時に回す（持ち主の依頼 2026-11-03）。
# Project_Kotoyui の直下から回る。見込み: 約 1.5 時間（長い方: 8×16・18000 ステップ + 人格の器）
export PYTHONIOENCODING=utf-8
P="python experiments/exp18_scale_gen.py --device cuda"
$P --kind text4f --stages 8 --width 16 --steps 18000 --ft_steps 2000 --tag t8x16_18k &
$P --kind split4f --G 16 --stages 4 --width 4 --steps 9000 --tag s16_4x4 &
wait
