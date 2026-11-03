#!/bin/bash
# 実験08b: 光路を大きく・学習を長く（4f 系 6 段 × 幅 8、18000 ステップ）。マスの入れ方 + 像の面の板（実験08 で拡散より良かった）、符号なしの 2 つの読み方。seed 0。
# 見込み: 約 1 時間（1 本 約 25〜30 分、段 × 幅が 3 倍・ステップ 2 倍）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
P="python experiments/exp08_resonance_fix.py --seed 0 --stages 6 --width 8 --steps 18000 --out results/exp08.txt"
$P --input pixel --plates xy --read image
$P --input pixel --plates xy --read hartley
