#!/bin/bash
# 実験02b: 光の分担の 2DTG（口調のスイッチなし / あり）を seed 1・2 で（02a の続き）。見込み: 約 80 分（1 本 約 20 分 × 4）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=results/exp02.txt
for s in 1 2; do
  python experiments/exp02_split_style.py --method split --seed $s --out $O
  python experiments/exp02_split_style.py --method split_tag --seed $s --out $O
done
