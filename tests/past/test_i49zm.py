"""I49zm の関門: 属性の波と、語ごとに固有の位相を、1 本に並べて持つ形。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402
from test_i49zg import TRIS, _qs  # noqa: E402


def test_identity_nearly_orthogonal_and_stable():
    from i49zm_two_channel import TwoChannel
    t = TwoChannel(V.get, n_id=1024)
    ws = sorted(NAMES)
    ids = [t.ident(w) for w in ws]
    for i in range(len(ws)):
        assert abs(float(np.linalg.norm(ids[i])) - 1.0) < 1e-5
        for j in range(i):
            assert abs(np.vdot(ids[i], ids[j])) < 0.15           # 違う語どうしは、ほぼ直交
    assert np.array_equal(t.ident(ws[0]), ids[0])                # 同じ語は、いつ聞いても同じ波


def test_mix0_equals_attr_only():
    import i49k_seat_vec as KV
    from i49zm_two_channel import TwoChannel
    a = KV.Unbind(KV.SeatVec(V.get, weight="数", bind="位相"), online=True)
    b = KV.Unbind(KV.SeatVec(TwoChannel(V.get, mix=0.0).of, weight="数", bind="位相"), online=True)
    c = sorted(NAMES)
    for tri in TRIS * 2:
        for si, q in _qs(tri):
            assert a.ask(si, q, c)["答え"] == b.ask(si, q, c)["答え"]
        a.learn(tri)
        b.learn(tri)


def test_identity_channel_tells_known_partner():
    """属性がまったく同じ 2 語でも、固有の側が有れば、見せた語と座った事の有る方が上に来る。"""
    import i49k_seat_vec as KV
    from i49zm_two_channel import TwoChannel
    same = np.ones(8, dtype=np.complex64)
    attr = lambda w: same                                         # noqa: E731  どの語も属性が同じ
    for mix, want in ((0.0, False), (1.0, True)):
        u = KV.Unbind(KV.SeatVec(TwoChannel(attr, n_id=256, mix=mix).of, weight="数", bind="位相"), online=True)
        for tri in [("甲", "魚", "釣る"), ("乙", "肉", "焼く"), ("甲", "魚", "釣る")]:
            u.learn(tri)
        p = u.ask(0, (None, "肉", "焼く"), ["甲", "乙"])["点"]
        assert (p["乙"] > p["甲"] + 1e-3) == want


def test_word_without_attr_still_has_wave():
    from i49zm_two_channel import TwoChannel
    t = TwoChannel(lambda w: None if w == "無" else np.ones(4), n_id=16)
    assert t.of("無") is None                                     # 属性の長さがまだ分からない間は作らない
    v = t.of("有")
    assert v.shape == (20,)
    u = t.of("無")
    assert u.shape == (20,) and not u[:4].any() and u[4:].any()


def test_run_shapes():
    import i49zm_two_channel_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強"), ("友達", "魚", "釣る")], W)]
    saved = []
    res = Z.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")], saved.append)
    for start in ("白紙から", "青空の 3 つ組を先に"):
        r = res[start]
        assert len(r) == len(Z.VERS) + 1 and len(r["直の比べ"]) == 4 and saved
        assert "段ごとの当たり" in r["解く形｜属性＋固有｜混ぜ 1｜1024 本"] and "動き" in r["解く形｜属性＋固有｜混ぜ 1｜1024 本｜振幅を直す"]
        for k, d in r.items():
            if k != "直の比べ":
                assert 0 <= d["正しい物"] <= d["合わせて"] <= 1


def test_wave_channel_pads_missing_side():
    """語の波を並べる形で、属性は有るが波の無い語も、長さがそろう（PC で長さ違いのエラーになった所）。"""
    import i49k_seat_vec as KV
    from i49zm_two_channel import TwoChannel
    attr = lambda w: None if w == "波だけ" else np.ones(3)          # noqa: E731
    wave = lambda w: None if w == "属性だけ" else np.arange(1, 5) * (1 + len(w))   # noqa: E731
    t = TwoChannel(attr, mix=1.0, wave_of=wave)
    assert t.of("属性だけ") is None                                # 波の側の長さがまだ分からない
    assert t.of("両方").shape == (7,)
    a, b = t.of("属性だけ"), t.of("波だけ")
    assert a.shape == b.shape == (7,) and not a[3:].any() and not b[:3].any()
    u = KV.Unbind(KV.SeatVec(TwoChannel(attr, mix=1.0, wave_of=wave).of, weight="数", bind="位相"), online=True)
    for tri in [("両方", "属性だけ", "波だけ"), ("属性だけ", "両方", "波だけ")]:
        u.learn(tri)
    assert u.ask(0, (None, "属性だけ", "波だけ"), ["両方", "属性だけ"])["答え"] is not None
