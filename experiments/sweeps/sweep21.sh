#!/bin/bash
# 実験21: 係り受け解析（GiNZA）の手がかりを足して人称を当てる（解析は済ませてある、seed 0）。Project_Kotoyui の直下から回る。見込み: 約 8 分
export PYTHONIOENCODING=utf-8
python experiments/exp21_parse_subject.py --stage eval --seed 0 --device cuda
