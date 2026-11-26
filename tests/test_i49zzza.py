"""I49zzza: 話の中でもう出た語を持つ／遠ざける形（小さな作り物の話）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzza_talk_state_run as A


def _talks(n=20):
    """人は 2 人が 1 行ずつ交代する（続かない。1 つ前の人へ戻る）。物と動きは、毎行ちがう語。"""
    out = []
    for i in range(n):
        p, q = "人%d" % (i % 5), "人%d" % ((i + 1 + i // 5) % 5 if (i + 1 + i // 5) % 5 != i % 5 else (i + 2) % 5)
        tris = [((p if j % 2 == 0 else q), "物%d" % ((i + j) % 7), "動き%d" % ((i * 3 + j) % 6)) for j in range(6)]
        out.append((tris, {w for t in tris for w in t}))
    return out


def test_back_of_and_tally():
    assert A.back_of(["a"]) == (None, 1)
    assert A.back_of(["a", "b", "b"]) == ("a", 2)
    t = A.RunTally()
    t.learn(["a", "b"], "a")
    assert not t.stay(1) and t.stay(2)                      # 1 行の後は戻る方が多い。聞いていない所は「続く」


def test_alternating_people_are_caught_by_run_length():
    d = A.run(_talks(), ks=(0,))["元の順"]
    who = d["だれが"]
    assert who["引き継ぎ"] == 0                              # 人は毎行替わる
    assert who["続いた数で選ぶ"] >= 0.75 * who["問い"]       # 3 行目からは、1 つ前の人へ戻ると当たる
    assert who["続いた数＋話の中の人"] >= 0.75 * who["問い"]
    for seat in ("何を", "どうした"):
        c = d[seat]
        assert c["問い"] == who["問い"] and c["話でもう出た語を除く"] >= c["遷移の数え"]
        assert c["当てずっぽう_除いた後"] >= c["当てずっぽう"] - c["正解がもう出た語"]
