"""I48ap（単位ごとの介入の板）・I48ar（位相で重ねた板）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48ar_superpose as AR  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def model():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    m = AR.RModel(S, fill, "基準").fit(tr, tr[:4]).init_units("B")
    for a, b in (("友達", "散歩"), ("先生", "散歩"), ("犬", "散歩")):
        keys, gaps = AK.made_sentence(a, b)
        m.teach_unit(("と", f"{a}と{b}に行った", keys, gaps))
    m.refresh()
    return m


def test_lookup_and_pack():
    ub, uw = AP.pack(np.array([5, 2, 5]), np.array([0.25, 0.5, 0.25]))
    assert list(ub) == [2, 5] and np.allclose(uw, [0.5, 0.5])
    assert np.isclose(AP.lookup(ub, uw, np.array([5, 7]), np.array([2.0, 3.0])), 1.0)


def test_units_undo_and_main_untouched():
    m = model()
    assert len(m.units) == 9 and m.Pn.sum() > 0
    W0 = m.pg.W.copy()
    m.undo([u["番号"] for u in m.units])
    assert np.allclose(m.Pn, 0) and np.allclose(m.Pd, 0, atol=1e-6) and np.array_equal(m.pg.W, W0)


def test_phase_stack_exact_with_enough_j_and_remove():
    """J が単位の数の倍数なら、復号は 1 枚ずつの線形の点（P ＋ 役の和）と同じ。取り消すと板から消える。"""
    m = model()
    assert sum(u["詰め物"] == "と" for u in m.units) == 3
    st = AR.Stack(m, m.units, 6)
    keys = [("先生", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")]
    xP, xQ = AR.gap_inputs(m, keys, AK.made_items(m.S, keys, None)[0])
    c = m.cid("と")
    z, d, _ = st.light(xP, xQ)[c]
    U = st.U[c]
    lin = U["P"] @ xP + sum(U[r] @ x for r, x in xQ.items())
    assert np.allclose(d, lin, atol=1e-4)
    st.remove(c, 0)
    z2, d2, _ = st.light(xP, xQ)[c]
    assert abs(d2[0]) < 1e-4 and np.allclose(d2[1:], lin[1:], atol=1e-4)
