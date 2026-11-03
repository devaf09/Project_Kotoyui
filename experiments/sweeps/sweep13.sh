#!/bin/bash
# 実験13: 推し量る力を、教えの会話を半分に分けて測り直す（A だけで作り、B にだけ出る新しい組を当てるか、AUC）。3 seed（半分の分け方も seed で変える）。
# 見込み: 約 40 分（1 本 約 13 分: 準備 1 分 + 光の光路 9000 ステップ 約 8 分 + 低ランク 約 3 分、実験08 の実測）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
for s in 0 1 2; do
  python experiments/exp13_half_split.py --seed $s --out results/exp13.txt
done
