"""I49e の物差しの数え（tools/past/i49e_axis_order.py）の関門。"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import i49e_axis_order as AO  # noqa: E402


def test_board_axis_is_front_minus_back():
    cb = [dict(name="象", pos="名詞", attrs=dict(B={"性質:大きい": 0.3, "性質:大きな": 0.2, "性質:小さい": 0.1})),
          dict(name="鼠", pos="名詞", attrs=dict(B={"性質:小さな": 0.4})),
          dict(name="走る", pos="動詞", attrs=dict(B={"性質:大きい": 0.9})),
          dict(name="雲", pos="名詞", attrs=dict(A={}))]
    ba = AO.board_axes(cb)
    assert abs(ba["象"]["大きい/小さい"] - 0.4) < 1e-9 and abs(ba["鼠"]["大きい/小さい"] + 0.4) < 1e-9
    assert "走る" not in ba and "雲" not in ba                 # 名詞だけ、B 帯の無い語は入れない


def test_order_count_keeps_ties_apart():
    o = AO.order_count([(2, 1), (1, 2), (1, 1), (3, 0)])
    assert (o["当たり"], o["外れ"], o["同点"]) == (2, 1, 1)
    assert o["当たりの割合_同点を除く"] == round(2 / 3, 4) and o["当たりの割合_同点は半分"] == 0.625


def test_pairs_are_written_for_every_axis():
    assert set(AO.PAIRS) == set(AO.AXES) and all(len(v) >= 14 for v in AO.PAIRS.values())
