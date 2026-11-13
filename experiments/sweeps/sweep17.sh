#!/bin/bash
# 実験17: 場面を絞って（20 字までの発言）、光の 2DTG で文を一から作れるか。語尾の個性（お嬢様・老人・後輩・侍・俺様）を口調の帯で切り替える。
# 光の 2DTG・GRU を各 9000 ステップ （ローカル AI の評点は Ollama を動かせるときに別に付ける）。Project_Kotoyui の直下から回る。
# 見込み: 約 50 分（光の 2DTG 約 25 分・GRU 約 5 分・材料の用意 約 5 分（AI の評点なし））
export PYTHONIOENCODING=utf-8
python experiments/exp17_short_gen.py --device cuda
