#!/bin/bash
# 実験10: 語のマスの並べ方（次の発言に一緒に出やすい語どうしを近くに）。実験08 の最良の形（マス + 像の面の板 + 像の面で読む、4f 系 4 段 × 幅 4）で 3 seed、ハートレーで読む形は seed 0。
# 見込み: 約 50 分（1 本 約 11 分 + 並べ替え 約 1.5 分、実験08 の実測）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
for s in 0 1 2; do
  python experiments/exp08_resonance_fix.py --input pixel --plates xy --read image --layout svd --seed $s --out results/exp10.txt
done
python experiments/exp08_resonance_fix.py --input pixel --plates xy --read hartley --layout svd --seed 0 --out results/exp10.txt
