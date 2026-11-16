"""I48g（属性の帯を波にする）の小さな確かめ。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48g_attr_band_wave as G  # noqa: E402

BOARDS = [dict(name="猫", pos="名詞", attrs={"C": {"歩く/する": 0.7, "食べる/する": 0.3}, "D": {"クラス:動物": 1.0}}),
          dict(name="歩く", pos="動詞", attrs={"C": {"主語:動物": 0.9, "目的語:動物": 0.1}})]


def test_band_wave_reads_back_ratios():
    bd = G.Bands(BOARDS, 1024, "p8", "band")
    v = bd.read(BOARDS[0], "C")
    assert abs(v[bd.aix["C"]["歩く/する"]] - 0.7) < 0.05 and abs(v[bd.aix["C"]["食べる/する"]] - 0.3) < 0.05
    assert v[bd.aix["C"]["主語:動物"]] == 0.0                      # 持っていない属性は床の下
    assert bd.read(BOARDS[1], "D") is None
    assert bd.bytes_per_word(BOARDS[0]) == 2 * 4 * 1024


def test_odd_test_table_and_wave_agree():
    T = [dict(noun="猫", verb="歩く", role="する")] * 4
    a = G.odd_test(G.Table(), BOARDS, T)
    b = G.odd_test(G.Bands(BOARDS, 1024, "c64", "band"), BOARDS, T)
    assert a["役の入れ替え"] == b["役の入れ替え"] == 1.0
