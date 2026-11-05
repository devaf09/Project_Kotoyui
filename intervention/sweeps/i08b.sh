#!/bin/bash
# 実験I8b（CPU だけ）: 器の学び方を変える。経験は 1 回目に集めて保存したものを使う: bash intervention/sweeps/i08b.sh 0
export CUDA_VISIBLE_DEVICES=
python intervention/i08_light_judge.py --seed $1 --mem labeled --out ../Project_Kotoyui_private/intervention/results/i08b.jsonl
python intervention/i08_light_judge.py --seed $1 --mem balanced --skip_rule 1 --out ../Project_Kotoyui_private/intervention/results/i08b.jsonl
