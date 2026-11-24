"""今の土台の入口: 本体と、短い覚えを付けた形が、入口から取れて同じ動きをする。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
from test_i49zw import attr as small  # noqa: E402

import i49_foundation as FD  # noqa: E402
import i49h_tally_run as HR  # noqa: E402


def test_half_is_the_same_as_the_tally_tool():
    assert FD.HALF == HR.HALF


def test_body_learns_and_answers():
    get = lambda w: small(w[0])
    b = FD.body(get, get)
    tri = (("犬", "NOUN"), ("骨", "NOUN"), ("かじる", "VERB"))
    b.learn(tri)
    b.tick()
    assert b.ask(2, (tri[0], tri[1], None), [tri[2], ("読む", "VERB")])["答え"] == tri[2]


def test_with_memory_reflects_and_restores():
    get = lambda w: small(w[0])
    m = FD.with_memory(get, get)
    snap = m.save()
    tri = (("犬", "NOUN"), ("骨", "NOUN"), ("かじる", "VERB"))
    cands = [tri[2], ("読む", "VERB")]
    m.ask(2, (tri[0], tri[1], None), cands)
    m.teach(2, tri, None)
    m.settle(tri)
    m.tick()
    assert m.reflected == 1 and m.n == 0
    assert m.body.ask(2, (tri[0], tri[1], None), cands)["答え"] == tri[2]      # 反映の後は本体だけで答える
    m.restore(snap)
    assert m.body.ask(2, (tri[0], tri[1], None), cands)["答え"] is None          # セーブから戻すと、教わる前に戻る
