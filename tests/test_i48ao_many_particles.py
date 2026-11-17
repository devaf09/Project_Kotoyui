"""I48ao（助詞をまとめて教える、中間の合わせ方）の確かめ。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48an_attr_board as AN  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_teach_all_well_formed():
    """自作の教える文: 要点と詰め物の数が合い、どの文にも教える助詞の間がある。計 70 文。"""
    assert len(AO.TEACH_ALL) == 70
    for p, text, keys, gaps in AO.TEACH_ALL:
        assert len(keys) == len(gaps) and p in gaps and p in AO.PARTICLES, text


def test_middle_is_mean_of_ringing_roles():
    """中間: 鳴った役（点が 0 でない役）だけの点の平均。掛け算（和）より小さく、鳴った役が 1 つならその役の点。"""
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    m = AO.OModel(S, fill, "基準").fit(tr, tr[:4]).init_q()
    keys = [("先生", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")]
    tk, tg = AK.made_sentence("友達", "散歩")
    AN.teach(m, ("と", "友達と散歩に行った", tk, tg), ("Q", "Q0", "掛け算"), None)
    prod = m.q_score(keys, 0)
    m.how = "中間"
    mid = m.q_score(keys, 0)
    c = m.cid("と")
    assert prod[c] > 0 and 0 < mid[c] <= prod[c]
