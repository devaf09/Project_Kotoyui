"""I48an（並びの板と属性の板）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48an_attr_board as AN  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_role_keys():
    keys = [("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")]
    assert AN.role_keys(keys, 0) == {"左": ("友達", "NOUN"), "右": ("散歩", "NOUN"), "述語": ("行く", "VERB")}
    assert AN.role_keys(keys, 2) == {"左": ("行く", "VERB"), "右": None, "述語": None}


def test_q_board_spreads_by_attrs_and_keeps_main():
    """友達 と を Q に教えると、同じクラスの 先生 の間でも「と」が上がる（属性で広がる）。本体の板は変わらない。何も教えていなければ Q の点は 0。"""
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    m = AN.QModel(S, fill, "基準").fit(tr, tr[:4]).init_q()
    keys = [("先生", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")]
    assert m.q_score(keys, 0) == 0
    W0 = m.rR.W.copy()
    c = m.cid("と")
    before = m.gap_scores(keys, AK.made_items(S, keys, None)[0], ())["今"][c]
    tk, tg = AK.made_sentence("友達", "散歩")
    AN.teach(m, ("基本", "友達と散歩に行った", tk, tg), ("Q", "Q0", "掛け算"), None)
    after = m.gap_scores(keys, AK.made_items(S, keys, None)[0], ())["今"][c]
    assert after > before and np.array_equal(m.rR.W, W0)
    assert any(e.get("板") == "Q" for e in m.log)
