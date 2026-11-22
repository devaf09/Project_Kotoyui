"""alt1（自分の変化を言えるか）の関門。numpy だけで動く。小さな条件で、道具が壊れていない事だけを見る。"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention", "alt"))
import alt1_self_change as A  # noqa: E402


def args(**kw):
    d = dict(trials=6, seed=0, dim=64, bands=8, longest=256.0, known=3, base=200, read=60, probe=20, mix=1.0)
    d.update(kw)
    return argparse.Namespace(**d)


def test_sur_of_matches_vessel_guess():
    rng = np.random.default_rng(0)
    v = A.V.Vessel(32, bands=5)
    for x in A.unit(A.cvec(rng, 40, 32)):
        v.hear(x)
    X = A.unit(A.cvec(rng, 7, 32)).astype(np.complex64)
    got = A.sur_of(v.S, X)
    want = np.stack([v.guess(x) for x in X], axis=1)
    assert np.allclose(got, want, atol=1e-5)


def test_reproducible_and_reading_does_not_touch_snapshot():
    a = args()
    r1, r2 = A.one(3, "new", a), A.one(3, "new", a)
    assert r1["target"] == r2["target"] and np.array_equal(r1["drop"][0], r2["drop"][0])
    assert a.known <= r1["target"] < 2 * a.known
    assert 0 <= A.one(3, "re", a)["target"] < a.known and A.one(3, "usual", a)["target"] == -1


def test_easy_case_is_answered_right_after_reading():
    res = A.run(args())
    assert res["new/drop/0"]["all"] == 1.0                 # はっきりした作品を読んだ直後は、前の自分と比べれば当たる
    assert res["new/turn_auc"][-1] > 0.9                   # 初めての物を読むと、遅い波の向きが、いつもより大きく替わる
