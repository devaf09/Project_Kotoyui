"""I49zzs: 掛け合わせた波と、外れた時だけ位相を動かす学び（小さい世界）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzs_bind as B
import i49zzs_bind_run as R


def test_phases_do_not_follow_the_numbering():
    """記号の番号に比例した位相だと、足し算の答えが初めから入る。そうなっていない事。"""
    t = B.sym_phases(6, 512)
    assert t.dtype == np.float32 and t.shape == (6, 512)
    same = np.abs(np.exp(1j * (t[1] + t[4] - t[2] - t[3])).mean())      # 1＋4 と 2＋3
    assert same < 0.2


def test_world_holds_out_mirrors_together():
    tr, te = R.world(10, 0.2)
    assert len(tr) + len(te) == 100
    seen = {(a, b) for a, b, _ in tr}
    assert all((b, a) not in seen for a, b, _ in te)
    assert all(c == (a + b) % 10 for a, b, c in tr + te)


def test_pair_sum_cannot_hold_the_ring_but_binding_can():
    tr, _ = R.world(10, 0.0)
    assert abs(B.accuracy(B.PairSum(10, tr), tr) - 0.1) < 1e-6
    assert B.accuracy(B.BindCount(10, tr, 1024), tr) == 1.0


def test_cat_and_dog_world():
    tr, _ = R.world(2, 0.0)
    assert B.accuracy(B.PairSum(2, tr), tr) == 0.5
    assert B.accuracy(B.BindCount(2, tr, 256), tr) == 1.0


def test_nudge_reaches_unseen_pairs_only_when_there_is_a_rule():
    tr, te = R.world(10, 0.2)
    m = B.BindNudge(10, 1024, shared=True)
    before = B.accuracy(m, te)
    miss = m.learn(tr, eta=0.3)
    assert miss[-1] == 0 and B.accuracy(m, tr) == 1.0
    assert B.accuracy(m, te) > before + 0.3
    tr0, te0 = R.world(10, 0.5, null=True)
    m0 = B.BindNudge(10, 1024, shared=True)
    m0.learn(tr0, eta=0.3)
    assert B.accuracy(m0, te0) < 0.3


def test_nudge_does_nothing_when_right():
    m = B.BindNudge(4, 64, shared=False)
    c = int(np.argmax(m.scores(1, 2)))
    keep = m.a.copy()
    assert m.hear(1, 2, c) is False
    assert np.array_equal(keep, m.a)


def test_class_world_reaches_unseen_pairs_with_separate_tables():
    """席ごとに語が違っても、語に類が有れば、外れた時だけ動かす形は聞いていない組に届く。数えるだけでは届かない。"""
    tr, te, n = R.class_world(4, 5, 0.2)
    assert n == 20 and len(tr) + len(te) == 400
    seen = {(s, o) for s, o, _ in tr}
    assert all((s, o) not in seen for s, o, _ in te)
    assert B.accuracy(B.BindCount(n, tr, 1024, 4), te) < 0.45
    m = B.BindNudge(n, 1024, False, 4)
    m.learn(tr, eta=0.3)
    assert B.accuracy(m, tr) == 1.0 and B.accuracy(m, te) > 0.7
