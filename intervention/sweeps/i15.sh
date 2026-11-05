#!/bin/bash
# 実験I15（CPU だけ、1 回 15 秒）: bash intervention/sweeps/i15.sh
for s in 0 1 2 3 4 5; do python intervention/i15_real_hierarchy.py --seed $s; done
