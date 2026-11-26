"""I49zzw: 前の 3 つ組から次の 3 つ組を出す数え（小さな作り物の話）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzw_next_triple_run as W

N = lambda x: (x, "NOUN")
V = lambda x: (x, "VERB")


def _talks(n=24):
    """「買う」の次は「食べる」、「借りる」の次は「読む」。人と物は話ごとに替わる。"""
    people, foods, books = ["太郎", "花子", "次郎", "三郎"], ["りんご", "みかん", "パン"], ["本", "雑誌", "新聞"]
    out = []
    for i in range(n):
        p, f, b = N(people[i % 4]), N(foods[i % 3]), N(books[(i // 2) % 3])
        tris = [(p, f, V("買う")), (p, f, V("食べる")), (p, b, V("借りる")), (p, b, V("読む"))]
        out.append((tris, {w for t in tris for w in t}))
    return out


def _attr(w):
    names = ["太郎", "花子", "次郎", "三郎", "りんご", "みかん", "パン", "本", "雑誌", "新聞", "買う", "食べる", "借りる", "読む"]
    v = np.zeros(len(names) + 3, dtype=np.complex64)
    v[names.index(w[0])] = 1.0
    v[len(names) + (0 if names.index(w[0]) < 4 else 1 if names.index(w[0]) < 10 else 2)] = 1.0
    return v


def test_count_learns_what_follows():
    m = W.Count()
    for tris, _ in _talks():
        for a, b in zip(tris, tris[1:]):
            m.learn(a, b)
    cands = [V("買う"), V("食べる"), V("借りる"), V("読む")]
    prev = (N("太郎"), N("りんご"), V("買う"))
    assert W.pick(cands, m.scores(2, prev, cands)) == V("食べる")
    assert W.pick(cands, [0.0] * 4, fallback=V("買う")) == V("買う")           # 点が全部 0 なら、引き継ぎ


def test_run_counts_and_beats_carry_over_on_the_verb():
    d = W.run(None, _attr, _talks(), ks=(0,), dims=512)
    v = d["元の順"]["どうした"]
    assert v["全部"]["問い"] == 12 * 3                                       # 後半 12 話 × 続いた 2 行の組 3 つ
    assert v["全部"]["問い"] == v["前と同じ語が続く"]["問い"] + v["前と違う語に替わる"]["問い"]
    assert v["全部"]["引き継ぎ"] == 0                                        # 動詞は、いつも替わる
    assert v["全部"]["遷移の数え"] > 0.8 * v["全部"]["問い"] and v["全部"]["遷移の波"] > 0.8 * v["全部"]["問い"]
    s = d["元の順"]["だれが"]["全部"]
    assert s["引き継ぎ"] == s["問い"]                                        # 人は、話の中で替わらない
    assert d["合計"]["どうした"]["全部"]["問い"] == v["全部"]["問い"]
