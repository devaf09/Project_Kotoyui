"""I48ae・I48ai（節ごとの印、OK を覚える段）の確かめ（小さな手作りの語板。GiNZA を使わない）。"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ai_remember as R  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_clause_marks_only_on_its_clause():
    """過去の印は、た の付いた述語の節にだけ付く（文で 1 組なら全部の間に付く）。"""
    s = sent("犬")
    func = G.Func(FB)
    assert AE.clause_marks(s, func) == [("過去",), ("過去",), ("過去",)]       # 述語は最後の 1 つだけなので、どの間も同じ節
    assert G.Func(FB).marks(s["toks"]) == {"過去"}


def test_memo_ok_raises_and_dark_is_light():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("D",))
    rows = [(*S.attrs(n, "NOUN"), y) for n, y in (("犬", "の"), ("友達", "と")) * 3]
    pg = G.PairGrain(256).fit(iter(rows), Counter(y for *_, y in rows))
    m = R.Memo(pg, 1e-3)
    m.refresh()
    f, v = S.attrs("先生", "NOUN")
    ci = pg.cls.index("と")
    p0 = m.post(f, v)[ci]
    m.ok(ci, f, v)
    m.refresh()
    assert m.post(f, v)[ci] > p0                           # OK を足すと、その板の点が上がる
    p1 = m.post(f, v)[ci]
    m.bad(ci, f, v)
    assert abs(m.post(f, v)[ci] - p1) < 1e-3               # だめは 1/1,000 なので、ほとんど動かない
    assert np.isclose(m.Wd.sum(), 1e-3 * v.sum())
