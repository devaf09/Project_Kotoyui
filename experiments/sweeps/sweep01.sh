#!/bin/bash
# 実験01: 光の 2DWIG（袋を 64 × 64 の場に）と、同じ袋の回帰・数えた表を比べる。
# データ: 青空文庫 全 4,477 作品の一人称つき会話文（12 類）、現代の雑談の丁寧さ（2 類、訓練 10 万発言）。学習 5000 ステップ、光と回帰は 3 seed。
# 共通の待ち行列から GPU 1 枚で回す（GPU の番号は指定しない）。見込み: 約 1 時間（光 1 本 約 4〜5 分 × 12 本、SN2 の MNIST 板 16 枚 約 200 秒から）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=results/exp01.txt
for task in aozora12 polite; do
  python experiments/exp01_wig_bag.py --task $task --method nb --device cpu --out $O
  for s in 0 1 2; do
    python experiments/exp01_wig_bag.py --task $task --method plates --seed $s --out $O
    python experiments/exp01_wig_bag.py --task $task --method lr --seed $s --out $O
  done
done
for s in 0 1 2; do
  python experiments/exp01_wig_bag.py --task aozora12 --method plates --spread --seed $s --out $O
done
