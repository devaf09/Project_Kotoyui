"""I49g の数の向きの数え（tools/past/i49g_number_vecs.py）の関門。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import i49g_number_vecs as NV  # noqa: E402


def test_spellings():
    assert NV.spell(3, "半角") == "3" and NV.spell(12, "全角") == "１２" and NV.spell(12, "漢数字") == "十二" and NV.spell(0, "漢数字") == "零"


def test_gap_means_fall_for_a_counting_line():
    vecs = {n: np.exp(1j * 0.3 * n * np.arange(8)).astype(np.complex64) for n in range(5)}   # 近い数ほど似ている見本
    g = NV.by_gap(vecs)
    assert g["1"]["近さ"] > g["2"]["近さ"] > g["3"]["近さ"]
    assert NV.step_dirs({0: np.array([0.0, 0]), 1: np.array([1.0, 0]), 2: np.array([2.0, 0])})["表裏を残した近さの平均"] == 1.0
