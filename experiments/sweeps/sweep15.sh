#!/bin/bash
# 実験15: 光の 2DWIG で、返事の候補の一人称の類（一人称を隠して）と丁寧さを判定する。Project_Kotoyui の直下から回る。
# 見込み: 約 15 分（1 本 約 5 分 + 材料の読み込み、実験01 の実測）
export PYTHONIOENCODING=utf-8
python experiments/exp15_persona_wig.py --task persona
python experiments/exp15_persona_wig.py --task polite --per_class 60000
