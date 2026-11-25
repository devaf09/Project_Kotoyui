"""I48m（返事の始めの句が相手の行の語を受けるか）の小さな確かめ（numpy だけで動く）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
import i48m_echo_phrase as M  # noqa: E402
from i48_wave_board import Waves  # noqa: E402


def tok(i, w, pos):
    return dict(i=i, text=w, lemma=w, pos=pos, dep="dep", head=0)


def toks(*tw):
    return [tok(i, w, p) for i, (w, p) in enumerate(tw)]


def test_first_phrase_skips_uke_and_counts_echo():
    prev = toks(("海", "NOUN"), ("に", "ADP"), ("行く", "VERB"), ("。", "PUNCT"))
    rep = toks(("そう", "ADV"), ("、", "PUNCT"), ("海", "NOUN"), ("です", "AUX"), ("ね", "PART"), ("。", "PUNCT"), ("山", "NOUN"), ("も", "ADP"))
    assert [t["text"] for t in M.first_phrase(rep)] == ["海", "です", "ね"]
    wv = Waves(["海", "行く", "山"], 64, 0)
    nr = M.Near(wv, 64)
    r = M.echo(prev, rep, nr)
    assert r["same"] and r["shape"] == "X+です+ね"
    r2 = M.echo(prev, toks(("山", "NOUN"), ("が", "ADP"), ("高い", "ADJ")), nr)
    assert not r2["same"]
    out = M.count([(prev, rep), (prev, rep)], nr)
    assert out["同じ語"]["受ける割合"] == 1.0 and out["多い形"][0] == ("X+です+ね", 2)
