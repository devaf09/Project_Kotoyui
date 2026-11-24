"""I49zw の関門: 2 語の組を、広げてから掛けて束ねる。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

# まばらな属性の波（属性 12 個）。人 = 0・1、食べ物 = 4・5、道具 = 8・9、動詞は別の所
A = {"太郎": {0: 1, 1: .5}, "花子": {0: 1, 1: .4, 2: .2}, "りんご": {4: 1, 5: .5}, "みかん": {4: 1, 5: .6}, "金づち": {8: 1, 9: .5},
     "食べる": {6: 1}, "かじる": {6: 1, 7: .3}, "振る": {10: 1}, "のこぎり": {8: 1, 9: .4, 11: .2}}


def attr(w):
    if w not in A:
        return None
    v = np.zeros(12, dtype=np.complex64)
    for i, r in A[w].items():
        v[i] = r
    return v


def test_sparse_product_keeps_only_shared_attributes():
    import i49zw_dense_pair as D
    p = D.PairSolve(attr, mode="まばら")
    assert p.sign(2, ("太郎", "りんご", None)) is None          # 共通の属性が無い 2 語は、掛けると何も残らない
    s = p.stat("太郎", "りんご")
    assert s["共通の属性"] == 0 and s["残る長さ"] == 0
    assert p.stat("太郎", "花子")["共通の属性"] == 2


def test_dense_sign_is_a_pair_code():
    import i49zw_dense_pair as D
    p = D.PairSolve(attr, dim=1024)
    cos = lambda x, y: float(np.real(np.vdot(x, y)))
    a = p.sign(2, ("太郎", "りんご", None))
    assert a is not None and abs(float(np.linalg.norm(a)) - 1) < 1e-4
    near = cos(a, p.sign(2, ("花子", "みかん", None)))             # 似た語どうしの組は、似た印
    half = cos(a, p.sign(2, ("太郎", "金づち", None)))             # 片方だけ同じ組は、遠い
    assert near > .8 and half < .2 and near > half
    assert cos(a, p.sign(2, ("りんご", "太郎", None))) < .2          # 席を入れ替えると、別の印


def test_solve_picks_by_the_pair():
    import i49zw_dense_pair as D
    for mode, want in (("広げる", "食べる"), ("まばら", None)):
        p = D.PairSolve(attr, mode=mode, dim=1024)
        for tri in (("太郎", "りんご", "食べる"), ("太郎", "金づち", "振る"), ("花子", "のこぎり", "振る")):
            p.learn(tri)
        o = p.ask(2, ("花子", "みかん", None), ["食べる", "振る", "かじる"])
        assert o["答え"] == want                                  # 聞いた事の無い組でも、似た組から当てる。まばらな掛け算では印が作れない
    assert p.ask(2, ("知らない", "みかん", None), ["食べる"])["答え"] is None


def test_both_adds_points():
    import i49zw_dense_pair as D
    b = D.Both(D.PairSolve(attr, dim=1024), D.PairSolve(attr, mode="まばら"))
    b.learn(("太郎", "りんご", "食べる"))
    b.tick()
    assert b.ask(2, ("花子", "みかん", None), ["食べる", "振る"])["答え"] == "食べる"


def test_run_shapes():
    import i49zp_text_talks_run as ZP
    import i49zw_dense_pair_run as R
    names = list(A)
    lines = ["太郎がりんごを食べる", "花子がみかんをかじる", "太郎が金づちを振る", "花子がのこぎりを振る", "太郎がみかんを食べる", "",
             "花子がりんごを食べる", "太郎がのこぎりを振る", "花子が金づちを振る", "太郎がりんごをかじる"]
    talks, _, _ = ZP.talks_of(lines)
    get = lambda w: attr(w[0])
    st = R.sparse_stats(get, talks, 1024)
    assert st["属性の総数"] == 12 and st["だれが×何を"]["何も残らない組の割合"] == 1.0 and len(names) >= st["属性を持つ語"]
    res = R.run(get, get, talks, ks=(0,), dim=1024)
    d = res["元の順"]
    for name in R.NAMES:
        assert 0 <= d[name]["正しい物"] <= 1
    assert d["組だけ・まばら"]["正しい物"] == 0 and d["組だけ・広げる"]["正しい物"] > 0
    assert len(d["直の比べ"]) == len(R.DUELS)


def test_amp_modes():
    import i49zw_dense_pair as D
    cos = lambda x, y: float(np.real(np.vdot(x, y)))
    for amp in ("そのまま", "属性を 1", "位相だけ"):
        p = D.PairSolve(attr, dim=1024, amp=amp)
        a = p.sign(2, ("太郎", "りんご", None))
        assert cos(a, p.sign(2, ("花子", "みかん", None))) > .7 and abs(cos(a, p.sign(2, ("太郎", "金づち", None)))) < .2
        if amp == "位相だけ":
            m = np.abs(a)
            assert float(m.max() - m.min()) < 1e-4            # どの本も同じ大きさ（力が全部の本に配られている）


def test_run_amp_shapes():
    import i49zp_text_talks_run as ZP
    import i49zw_dense_pair_run as R
    lines = ["太郎がりんごを食べる", "花子がみかんをかじる", "太郎が金づちを振る", "花子がのこぎりを振る", "太郎がみかんを食べる"]
    talks, _, _ = ZP.talks_of(lines)
    d = R.run_amp(lambda w: attr(w[0]), talks, ks=(0,), dim=1024)["元の順"]
    assert len(d) == 7 and len(d["直の比べ"]) == 4
    assert all(0 <= v["正しい物"] <= 1 for k, v in d.items() if k != "直の比べ")


def test_run_level_shapes():
    import i49zp_text_talks_run as ZP
    import i49zw_dense_pair_run as R
    lines = ["太郎がりんごを食べる", "花子がみかんをかじる", "太郎が金づちを振る", "花子がのこぎりを振る", "太郎がみかんを食べる", "花子がりんごを食べる"]
    talks, _, _ = ZP.talks_of(lines)
    get = lambda w: attr(w[0])
    d = R.run_level(get, get, talks, ks=(0,), dim=1024)["元の順"]
    assert sum(h["問い"] for h in d.values()) == 18
    for h in d.values():
        assert h["解く形と組"] - h["解く形"] == h["当たりに変わった"] - h["外れに変わった"]
