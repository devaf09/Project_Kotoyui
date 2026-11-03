#!/bin/bash
# 実験02c: 光の分担の 2DTG の学習を 3 倍（各 2DTG 9000 ステップ）にして、GRU に負けたのが学習の足りなさかを確かめる。seed 0、スイッチなし / あり。
# 比べる: 実験02 の 3000 ステップ（23.75 / 23.34%）。訓練の正解率も記録する。見込み: 約 2 時間（1 本 約 1 時間、実験02 の約 1,600 秒 × 3 から）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=results/exp02.txt
python experiments/exp02_split_style.py --method split --steps 9000 --seed 0 --out $O
python experiments/exp02_split_style.py --method split_tag --steps 9000 --seed 0 --out $O
