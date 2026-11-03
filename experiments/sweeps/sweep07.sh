#!/bin/bash
# 実験07: 「次の発言の共鳴」を光のグレインに（4f 系 4 段 × 4 + 平方根、ハートレー係数で読む）。10 択の返事選びで数え上げの PMI と比べる。
# 材料: RealPersonaChat・JMultiWOZ（教え 9 : 試し 1）、9000 ステップ。ハートレーは 3 seed、学習する符号の読み方（比べる）は seed 0。
# 見込み: 約 40 分（1 本 約 5 分 + 表の読み込みと 10 択の評価 約 5 分、SN2 実験84 の 4f 系の実測から）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
for s in 0 1 2; do
  python experiments/exp07_light_resonance.py --seed $s --out results/exp07.txt
done
python experiments/exp07_light_resonance.py --read codes --seed 0 --out results/exp07.txt
