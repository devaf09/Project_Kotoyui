#!/bin/bash
# 実験11: 推し量る力を測る（表に無い組だけの問・決め手の語を似た語に入れ替える介入・取り除く）。語板（kotoyui 0.2.0）と数え上げも同じ問で測る。
# 光路は実験08 の最良の形（3 seed）、比べる: デジタルの低ランク（seed 0）。
# 見込み: 約 50 分（1 本 約 11 分 + 測り 約 2 分）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
for s in 0 1 2; do
  python experiments/exp08_resonance_fix.py --input pixel --plates xy --read image --probe --seed $s --out results/exp11.txt
done
python experiments/exp08_resonance_fix.py --read lowrank --probe --seed 0 --out results/exp11.txt
