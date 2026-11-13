#!/bin/bash
# 実験20: 前の発言（流れ）と人物の手がかりを足して人称を当てる。seed 0・1 を 1 枚の GPU で同時に。Project_Kotoyui の直下から回る。
# 見込み: 約 10 分
export PYTHONIOENCODING=utf-8
P="python experiments/exp20_context_subject.py --device cuda"
$P --seed 0 &
$P --seed 1 &
wait
