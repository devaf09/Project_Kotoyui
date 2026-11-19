"""I48ba（と を向きで分ける）の確かめ（小さな手作りの文）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ba_split_to as BA  # noqa: E402
from test_i48ab_long_grain import sent  # noqa: E402


def test_relabel_only_to_gaps_and_keeps_lemma():
    cls = {"友達": "個人", "先生": "個人", "犬": "動物", "猫": "動物", "散歩": "行ない"}
    sp = BA.Splitter(lambda l, u: cls.get(l, "品詞:" + u))
    ref = [sent(n) for n in ("友達", "先生", "友達", "先生")] + [sent(n) for n in ("犬", "猫")]
    sp.fit(ref, np.random.default_rng(0), kmax=2)
    assert sp.K >= 1 and sp.n == 4                              # 「と」の間は 友達・先生 の 4 つ
    s = sent("友達")
    s2 = sp.relabel(s)
    g = G.split(s2)[1]
    assert g[0].startswith("と#") and g[1:] == G.split(s)[1][1:]
    assert [t["lemma"] for t in s2["toks"]] == [t["lemma"] for t in s["toks"]]
    assert sp.relabel(sent("犬")) is not None and G.split(sp.relabel(sent("犬")))[1] == G.split(sent("犬"))[1]
