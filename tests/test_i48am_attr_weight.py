"""I48am（述語の語板で介入の重みを決める）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48am_attr_weight as AM  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent, t  # noqa: E402


CB2 = CB + [dict(name="食べる", pos="動詞", freq=5, attrs=dict(C={"目的語:物": 1.0}))]


def eat(n):
    return dict(id="y", text="", toks=[t(n, n, "NOUN"), t("を", "を", "ADP", "助詞-格助詞", "case"), t("食べ", "食べる", "VERB", "動詞-一般", "root"),
                                      t("た", "た", "AUX", "助動詞", "aux"), t("。", "。", "PUNCT", "補助記号-句点", "punct")])


def test_left_attr_ratio_favours_seen_attrs():
    """行く の左に人（友達・先生）、食べる の左に動物（犬・猫）が来る学びでは、行く の板で人の属性の比が 1 より大きく、動物の属性は 1 より小さい。"""
    S = G.AttrSpace(CB2, G.Func(FB), 256, bands=("B", "C", "D"))
    tr = [sent(n) for n in ("友達", "先生") * 5] + [eat(n) for n in ("犬", "猫") * 5]
    LA = AM.LeftAttr(S).fit(tr)
    assert LA.n_pairs == 30                               # 行く の文は (人, 行く)・(散歩, 行く)、食べる の文は (動物, 食べる)
    names = AM.attr_names(S, "先生", "NOUN")
    r = dict(zip(names, LA.ratio(("行く", "VERB"), ("先生", "NOUN"))))
    assert r["D|クラス:人"] > 1
    r_dog = dict(zip(AM.attr_names(S, "犬", "NOUN"), LA.ratio(("行く", "VERB"), ("犬", "NOUN"))))
    assert r_dog["D|クラス:動物"] < 1
    pi, pv = LA.ids(("行く", "VERB"), pred=True)
    inv = {v: k for k, v in LA.idx.items()}
    assert not any(inv[i].startswith("品詞:") for i in pi) and np.isclose(pv.sum(), 1)


def test_reweight_keeps_unit_length_and_only_touches_left_word():
    S = G.AttrSpace(CB2, G.Func(FB), 256, bands=("B", "C", "D"))
    LA = AM.LeftAttr(S).fit([sent(n) for n in ("友達", "先生") * 5] + [eat(n) for n in ("犬", "猫") * 5])
    keys = [("先生", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")]
    import i48ak_teach as AK
    x = AK.made_items(S, keys, ["と", "に", "た。"])[0]
    x1, info = AM.reweight(LA, keys, x, AM.TYPES[1])
    assert np.isclose(np.linalg.norm(x1["A"][0][1]), 1, atol=1e-5) and info["述語"] == "行く"
    assert x1["A"][1] is x["A"][1]                        # 型 1 は右の語を替えない
    x0, info0 = AM.reweight(LA, keys, x, AM.TYPES[0])
    assert info0 == {} and x0["A"][0] is x["A"][0]
