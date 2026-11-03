#!/bin/bash
# 実験04: デジタルの符号を減らす（方法 1: 2 段で読む / 方法 2: 符号を学習しない / 方法 3: 画素・フーリエ（ハートレー）の係数 1 つを答え 1 つに）。
# データ: 実験03 と同じ（現代の雑談、口調のスイッチなし）。光路は 4f 系 6 段 × 幅 8 + 平方根、9000 ステップ、seed 0。比べる: 実験03 の光路 1 本（学習する符号）。
# 見込み: 約 2.5 時間（光路 1 本 約 15 分 × 8 本と、2 段で読む方式の 64 組の評価）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=results/exp04.txt
for m in hartley pixel fixed proto two_clust two_freq; do
  python experiments/exp04_codes.py --method $m --seed 0 --out $O
done
