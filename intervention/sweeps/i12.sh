#!/bin/bash
# 実験I12（CPU だけ）: I8b の器（口ごとに 1 種類）で、結論を 1 つに決めず明るさの割合を次の器へ渡す: bash intervention/sweeps/i12.sh 0
export CUDA_VISIBLE_DEVICES=
python intervention/i08_light_judge.py --seed $1 --mem labeled --soft 1 --skip_rule 1 --out ../Project_Kotoyui_private/intervention/results/i12.jsonl
