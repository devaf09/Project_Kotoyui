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


def _toks(*rows):
    """(原形, 品詞, 係り, 親の番号) の並び → triples2 が読む形（番号は 0 から、親が自分なら根）。"""
    return [dict(i=i, l=l, p=p, d=d, h=h) for i, (l, p, d, h) in enumerate(rows)]


def test_triples2_takes_wa_ni_and_pairs():
    s = _toks(("猫", "NOUN", "nsubj", 4), ("は", "ADP", "case", 0), ("庭", "NOUN", "obl", 4), ("で", "ADP", "case", 2),
              ("寝る", "VERB", "root", 4))
    assert TC.triples2(s) == [("三つ組", "は", "で", (("猫", "NOUN"), ("庭", "NOUN"), ("寝る", "VERB")))]
    s = _toks(("魚", "NOUN", "obj", 2), ("を", "ADP", "case", 0), ("食べる", "VERB", "root", 2))
    assert TC.triples2(s) == [("目的と述語", None, "を", (("魚", "NOUN"), ("食べる", "VERB")))]


def test_tables_back_off_in_order():
    cat, fish, eat, bird = ("猫", "NOUN"), ("魚", "NOUN"), ("食べる", "VERB"), ("鳥", "NOUN")
    tb = TC.Tables()
    tb.add((cat, fish, eat))
    tb.add((bird, fish, eat))
    s, lv = tb.scores(0, (cat, fish, eat), [cat, bird], backoff=False)   # だれがを隠す: (魚, 食べる) の 2 語の表
    assert lv == "2語" and s == {cat: 1, bird: 1}
    s, lv = tb.scores(2, (("犬", "NOUN"), fish, eat), [eat, ("走る", "VERB")], backoff=True)   # 2 語の表に無い → 述語側の 1 語（魚）
    assert lv == "述語側の1語" and s[eat] == 2
    assert tb.scores(2, (("犬", "NOUN"), ("骨", "NOUN"), eat), [eat], backoff=True) == (None, "答えられない")
    assert TC.rank_by({cat: 1, bird: 1}, cat) == 2


def test_talk_lines_drop_speaker_marks():
    import i49e_make_talk as MT
    assert MT.lines_of("子: 猫が魚を食べたよ。\n大人：そうなの。\n（場面）\n") == ["猫が魚を食べたよ。", "そうなの。"]


def test_copy_tables_is_independent():
    t = TC.Tables()
    t.add((("猫", "NOUN"), ("魚", "NOUN"), ("食べる", "VERB")))
    c = TC.copy_tables(t)
    c.add((("猫", "NOUN"), ("魚", "NOUN"), ("食べる", "VERB")))
    assert sum(t.t2[0].values()) == 1 and sum(c.t2[0].values()) == 2


def test_hits_by_pair_splits_questions():
    cat, fish, eat, sky = ("猫", "NOUN"), ("魚", "NOUN"), ("食べる", "VERB"), ("空", "NOUN")
    vec = {w: e(i) for i, w in enumerate([cat, fish, eat, sky])}
    tb = TC.Tables()
    tb.add((cat, fish, eat))
    pairs = {frozenset([(0, cat), (1, fish)])}
    items = [dict(組=(cat, fish, eat), ページ={cat, fish, eat, sky})]
    o = TC.hits_by_pair(items, tb, pairs, vec)
    assert o["どうした"]["組があった"] == dict(問い=1, 一位=1.0) and o["だれが"]["組が無かった"]["問い"] == 1


def test_share_borrows_from_similar_words():
    cat, kitten, fish, eat, run, sky = (("猫", "NOUN"), ("子猫", "NOUN"), ("魚", "NOUN"), ("食べる", "VERB"), ("走る", "VERB"), ("空", "NOUN"))
    vec = {cat: e(0), kitten: (e(0) + 0.1 * e(5)), fish: e(1), eat: e(2), run: e(3), sky: e(4)}
    tris = [(cat, fish, eat)]
    sh = TC.Share(tris, vec, [kitten, fish, eat, run, sky])
    s = sh.scores("b", 2, (kitten, fish, None), [eat, run])   # 子猫 と 魚 は見た事が無い組 → 猫 と 魚 の数を借りる
    assert s[eat] > 0.9 and s[run] == 0.0
    rng = np.random.default_rng(0)                         # 波の板は、向きが散らばった（密な）ベクトルで確かめる
    dv = {w: np.exp(1j * rng.uniform(0, 2 * np.pi, 256)).astype(np.complex64) for w in (cat, fish, eat, run, sky)}
    dv[kitten] = dv[cat]
    sh = TC.Share(tris, dv, [kitten, fish, eat, run, sky])
    s = sh.scores("d", 2, (kitten, fish, None), [eat, run])
    assert s[eat] > 5 * s[run]
