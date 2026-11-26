"""I49zzzd: 持った波で「だれが」を選ぶ、持った波で動きを引く（小さな作り物の話。本体は代わりの物）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools", os.path.join("intervention", "past")):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzw_next_triple_run as W
import i49zzzd_hold_run as D
from test_i49zzw import N, V, _attr, _talks
from test_i49zzx import PairBody


def _wave():
    w = W.Wave(512, _attr, W.SCALE)
    for tris, _ in _talks():
        for a, b in zip(tris, tris[1:]):
            w.learn(a, b)
    return w


def test_hold_fades_by_half_life_and_does_not_delete():
    h = D.Hold(_wave(), (1, 4))
    h.hear((N("太郎"), N("りんご"), V("買う")))
    for _ in range(4):
        h.hear((N("花子"), N("本"), V("読む")))
    assert abs(h.n[4][0][N("太郎")] - 0.5) < 1e-5                           # 4 行で半分
    assert abs(h.n[1][0][N("太郎")] - 0.5 ** 4) < 1e-6 and h.n[1][0][N("太郎")] > 0   # 薄まるが、消えない
    people = [N("太郎"), N("花子"), N("次郎")]
    assert W.pick(people, h.word(4, 0, people)) == N("花子") and W.pick(people, h.attr(4, 0, people)) == N("花子")
    f = h.fork()
    f.hear((N("次郎"), N("本"), V("読む")))
    assert N("次郎") not in h.n[1][0] and h.v[1][0].dtype == np.complex64     # 分けた先を進めても、元は変わらない


def test_scores_k_with_the_last_line_equals_the_transition_wave():
    w = _wave()
    prev = (N("太郎"), N("りんご"), V("買う"))
    cands = [V("買う"), V("食べる"), V("借りる"), V("読む")]
    a, b = w.scores(2, prev, cands), D.scores_k(w, 2, [w._key(j, prev[j]) for j in range(3)], cands)
    assert np.allclose(a, b, atol=1e-5) and W.pick(cands, b) == V("食べる")
    h = D.Hold(w, (1, 4))
    h.hear(prev)
    h.hear((N("花子"), N("本"), V("借りる")))
    cands = cands + [V("歩く")]                                                # 聞いた事の無い語が混ざっても同じ
    fast = D.Fast(w)
    m = fast.many(2, [h.v[1], h.v[4]], cands)
    assert np.allclose(m[0], D.scores_k(w, 2, h.v[1], cands), atol=1e-4) and np.allclose(m[1], D.scores_k(w, 2, h.v[4], cands), atol=1e-4)
    people = [N("太郎"), N("花子"), N("次郎")]
    assert np.allclose(fast.attr(h, 0, people)[1], h.attr(4, 0, people), atol=1e-4)
    w.learn(prev, (N("太郎"), N("りんご"), V("歩く")))                         # 聞いた後は、表を作り直す
    assert np.allclose(fast.many(2, [h.v[1]], cands)[0], D.scores_k(w, 2, h.v[1], cands), atol=1e-4)


def test_held_wave_finds_the_person_without_being_told_the_talk():
    talks = _talks()
    d = D.run(talks[:12], talks[12:], _attr, None, halves=(1, 1024), dims=512)
    c = d["だれが"]
    assert c["問い"] == 12 * 3 and c[D.HAND] == c["問い"]
    for kind in (D.ATTR, D.WORD):
        assert c[D.name_of(kind, 1)] == c["問い"]                           # 短く持てば、今の話の人
        assert c[D.name_of(kind, 1024)] < 0.5 * c["問い"]                   # 長く持つと、前の話の人と混ざる（4 人が順に出る世界）
    v = d["どうした"]
    assert v[D.LAST] == v["問い"] and v[D.name_of(D.HELD, 1)] >= 0.6 * v["問い"]


def test_null_world_person_changes_every_line():
    """人が行ごとに替わる（決まりが無い）世界では、持った波は当たらない。"""
    people = [N(x) for x in ("太郎", "花子", "次郎", "三郎")]
    talks, k = [], 0
    for tris, words in _talks():
        new = []
        for t in tris:
            k += 1
            new.append((people[int(k * 0.6180339887 * 4) % 4],) + tuple(t[1:]))
        talks.append((new, {w for t in new for w in t}))
    c = D.run(talks[:12], talks[12:], _attr, None, halves=(1, 16), dims=512)["だれが"]
    assert all(c[D.name_of(kind, h)] <= 0.5 * c["問い"] for kind in (D.ATTR, D.WORD) for h in (1, 16))


def test_free_run_counts_scene_and_never_repeats_a_verb():
    talks = _talks()
    d = D.run(talks[:12], talks[12:], _attr, PairBody(), halves=(1, 16), free=4, steps=3, show=1, dims=512)
    real = d["場面"]["本物の話"]
    assert real[1][0] == 4 and real[1][1] == 4 and real[3][2] == 4          # 本物の話の動きは、聞いた話の中でいつも同じ話に出ている
    names = [D.LAST + D.NEW, D.name_of(D.HELD, 1) + D.NEW, D.name_of(D.HELD, 16) + D.NEW]
    assert set(d["場面"]) == set(names) | {"本物の話"} and all(d["同じ物をもう一度出した数"][n] == 0 for n in names)
    assert len(d["自分で続ける"]) == 3 and all(len(x["続き"]) == 3 and x["続き"][0].endswith("。") for x in d["自分で続ける"])
