"""i48core（整理の入口）と、選びの本体の差し込みの口の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_core_names_import():
    from i48core import boards, hands, learn, measure, model, space, text, units
    assert text.split is G.split and space.Fill is AD.Fill and model.Model is AK.Model
    assert boards.PairGrain is G.PairGrain and callable(measure.measure) and callable(units.pack) and learn.BLOCK > 0
    assert all(len(h) == 6 for h in hands.HANDS)


class Bump:
    """試しの差し込み: 1 つの詰め物の今の点に、決まった数を足す。"""

    def __init__(self, c, amount):
        self.c, self.amount, self.n_refresh = c, amount, 0

    def score(self, m, keys, x, f, v):
        out = np.zeros(m.nc, dtype=np.float32)
        out[self.c] = self.amount
        return out

    def refresh(self, m):
        self.n_refresh += 1

    def clone(self):
        return Bump(self.c, self.amount)


def test_extras_add_to_now_and_clone():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    m = AK.Model(S, fill, "基準").fit(tr, tr[:4])
    s = tr[2]
    keys = AK.keys_of(s)
    x = AE.items(S, s, "節")[0]
    fic = m.who.fiction(keys)
    g0 = m.gap_scores(keys, x, fic)["今"].copy()
    b = Bump(0, 2.5)
    m.extras.append(b)
    m.refresh()
    assert b.n_refresh == 1
    g1 = m.gap_scores(keys, x, fic)["今"]
    assert np.isclose(g1[0] - g0[0], 2.5) and np.allclose(g1[1:], g0[1:])
    m2 = m.clone()
    assert m2.extras[0] is not b and m2.extras[0].amount == 2.5
    m.extras.clear()
    m.refresh()
    assert np.allclose(m.gap_scores(keys, x, fic)["今"], g0)


def test_registry_defaults():
    from i48core import registry
    assert all(len(e) == 5 for e in registry.EXPERIMENTS)
    on = registry.on_by_default()
    assert "三つの共振 R1・R2・R3" in on and "驚きを帯に寄せる・ならす" not in on and "暗い側（外した選びを下げる）" not in on
