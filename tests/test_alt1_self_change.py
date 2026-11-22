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


def test_real_works_tool_runs_on_made_up_vectors(tmp_path):
    """alt1b（本物の作品用の道具）を、自作のベクトルの置き場で通す。作品の読み込みは PC だけなので、ここでは通さない。"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import alt1b_read_works as B
    rng = np.random.default_rng(0)
    wd = A.World(rng, 64, 4, 1.0)
    vecs = [wd.sent(w, 300) for w in range(8)]
    ws = B.Works(np.random.default_rng(1), vecs, 3)
    assert ws.usual(130).shape == (130, 64) and ws.work(4, 60).shape == (60, 64) and ws.probe(5, 20).shape == (20, 64)
    res = A.run(args(trials=4), world=lambda r: B.Works(r, vecs, 3))
    assert res["new/drop/0"]["all"] >= 0.75
    near = B.alike(vecs)
    assert near["同じ作品の前と後ろ"] > near["別の作品"]
