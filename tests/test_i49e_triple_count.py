"""I49e の「2 つから残りの 1 つ」の数え（tools/i49e_triple_count.py）の関門。言葉の型は使わず、見本のベクトルで確かめる。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import i49e_triple_count as TC  # noqa: E402


def e(i, dim=16):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


def test_rank_puts_ties_in_front():
    tb = TC.Table({("a", "NOUN"): e(0), ("b", "NOUN"): e(0), ("c", "NOUN"): e(1)})
    assert tb.rank_of(e(0), ("a", "NOUN"), [("a", "NOUN"), ("b", "NOUN"), ("c", "NOUN")]) == 2
    assert tb.rank_of(e(1), ("c", "NOUN"), [("a", "NOUN"), ("b", "NOUN"), ("c", "NOUN")]) == 1


def test_sum_of_two_finds_the_third():
    cat, fish, eat, sky = ("猫", "NOUN"), ("魚", "NOUN"), ("食べる", "VERB"), ("空", "NOUN")
    vec = {cat: e(0), fish: e(1), eat: (e(0) + e(1)) / np.sqrt(2), sky: e(2)}
    other = [("犬", "NOUN"), ("骨", "NOUN"), ("かむ", "VERB")]
    vec.update({other[0]: e(3), other[1]: e(4), other[2]: e(5)})
    page = [cat, fish, eat, sky]
    items = [dict(組=(cat, fish, eat), 文=(1, 0), ページの並び=page, ページ=set(page), 作品=set(vec)),
             dict(組=tuple(other), 文=(1, 1), ページの並び=list(other), ページ=set(other), 作品=set(vec))]
    res, n = TC.count(items, vec)
    assert n == 2
    o = res["どうしたを隠す・ア_ページ"]
    assert o["足した向き"]["数"] == 1 and o["足した向き"]["一位"] == 1.0    # 2 つ目はページの候補が 1 つしか無いので数えない


def test_triples_need_ga_and_wo():
    import pytest
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("ja_ginza")
    except OSError:
        pytest.skip("ja_ginza が無い")
    tri, why = TC.triples_of(nlp("猫が魚を食べた。"))
    assert tri == [(("猫", "NOUN"), ("魚", "NOUN"), ("食べる", "VERB"))]
    tri, why = TC.triples_of(nlp("猫は魚を食べた。"))
    assert tri == [] and why["が格が無い"] == 1
