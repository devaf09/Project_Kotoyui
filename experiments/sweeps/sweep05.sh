#!/bin/bash
# 実験05: 4f 系の分担の 2DTG（担当 4）に「2 段で読む」を組み合わせる（デジタルの符号をなくす）。引数で seed を 1 つ（sweep05.sh 0 など）。
# データ: 実験03・04 と同じ（現代の雑談、口調のスイッチなし）。各光路 9000 ステップ。
# 比べるために、同じ seed で実験03（学習する符号の分担・光路 1 本・足し合わせ）も回す（seed 0 は実験03 にあるので飛ばす）。
# 見込み: seed 1 つあたり約 1.5 時間（2 段で読む分担 約 50 分（担当 4 × 光路 2 本 × 約 5 分 + 評価）、実験03 約 36 分（実測））
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
S=${1:-0}
python experiments/exp05_split_two.py --seed $S --out results/exp05.txt
if [ "$S" != "0" ]; then
  python experiments/exp03_4f.py --seed $S --out results/exp03.txt
fi
