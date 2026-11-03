#!/bin/bash
# 実験02a: 光の分担の 2DTG（口調のスイッチなし / あり）と、GRU・3-gram（同）を現代の雑談で比べる。seed 0（光）、GRU は 3 seed、3-gram は数えた表なので 1 回。
# データ: RealPersonaChat・名大会話コーパス、知らない話者 51 人でテスト。訓練 200 万位置、光は各 2DTG 3000 ステップ。
# 見込み: 約 50 分（光 1 本 約 20 分 × 2、SN2-Parts-2D 実験69 の 1,099 秒から）
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=results/exp02.txt
python experiments/exp02_split_style.py --method split --seed 0 --out $O
python experiments/exp02_split_style.py --method split_tag --seed 0 --out $O
for s in 0 1 2; do
  python experiments/exp02_split_style.py --method gru --seed $s --out $O
  python experiments/exp02_split_style.py --method gru_tag --seed $s --out $O
done
python experiments/exp02_split_style.py --method tri --device cpu --out $O
python experiments/exp02_split_style.py --method tri_tag --device cpu --out $O
