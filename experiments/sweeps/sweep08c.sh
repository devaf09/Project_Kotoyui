#!/bin/bash
# 実験08c: 実験08 で最も良かった形（マスの入れ方 + 像の面の板 + 像の面で読む、4f 系 4 段 × 幅 4）を seed 1・2 でも。
# 見込み: 約 25 分（1 本 約 11 分、実験08 の実測）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
for s in 1 2; do
  python experiments/exp08_resonance_fix.py --input pixel --plates xy --read image --seed $s --out results/exp08.txt
done
