#!/bin/bash
# 実験19: 主語を隠して人称（話し手・相手・第三者）を当てる。seed 0・1 を 1 枚の GPU で同時に。Project_Kotoyui の直下から回る。
# 見込み: 約 10 分
export PYTHONIOENCODING=utf-8
P="python experiments/exp19_hidden_subject.py --device cuda"
$P --seed 0 &
$P --seed 1 &
wait
