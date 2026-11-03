#!/bin/bash
# 実験03: 4f 系 ＋ 平方根の光路（1 本・分担 4）と足し合わせで、現代の雑談の次のサブワード。口調のスイッチなし / あり、seed 0。
# データ: 実験02 と同じ（訓練 200 万位置、テスト 3 万位置、知らない話者 51 人）。各光路 9000 ステップ。
# 見込み: 約 1.5 時間（1 回 約 40 分。SN2-Parts-2D 実験84 の光路 1 本 9 分・分担 64 分から）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
python experiments/exp03_4f.py --seed 0 --out results/exp03.txt
python experiments/exp03_4f.py --tag --seed 0 --out results/exp03.txt
