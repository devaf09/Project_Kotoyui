#!/bin/bash
# 実験08: 光の共鳴グレインを直す（実験07 は語 1 つだと、どの語でも出力が同じだった）。入れ方 × 像の面の板 × 読み方、seed 0、9000 ステップ。
# 比べる: 光を使わない学習する低ランク（デジタル約 420 万）。実験07 の形（pixel・板なし・hartley）は results/exp07.txt の seed 0 を使う。
# 見込み: 約 50 分（1 本 約 6〜7 分、実験07 の実測から）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
P="python experiments/exp08_resonance_fix.py --seed 0 --out results/exp08.txt"
$P --input pixel --plates xy --read hartley
$P --input spread --plates none --read hartley
$P --input spread --plates xy --read hartley
$P --input spread --plates xy --read image
$P --input pixel --plates xy --read image
$P --input spread --plates xy --read codes
$P --read lowrank
