"""I49zzze〜h: 切れ目の信号、テーマ波と話の束ね、疑問の物差し、器の前に立つ部品（小さな作り物の話。本体は代わりの物）。"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools", os.path.join("intervention", "past")):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzw_next_triple_run as W
import i49zzzd_hold_run as D
import i49zzze_boundary_run as E
import i49zzzf_theme_run as F
import i49zzzg_doubt_run as G
import i49zzzh_shelter as S
import i49zzzh_shelter_run as H
from test_i49zzw import N, V, _attr, _talks
from test_i49zzx import PairBody

PEOPLE = ["太郎", "花子", "次郎", "三郎"]
WORDS = PEOPLE + ["包丁", "ペン", "窓", "野菜", "鍋", "手紙", "切手", "持つ", "見る", "切る", "煮る", "書く", "貼る"]


def _attr2(w):
    v = np.zeros(len(WORDS) + 3, dtype=np.complex64)
    i = WORDS.index(w[0])
    v[i] = 1.0
    v[len(WORDS) + (0 if i < 4 else 1 if i < 11 else 2)] = 1.0
    return v


def _scenes(n=32):
    """2 つの場面。初めの行の物だけが違い、2 行目（窓を見る）は同じ。3 行目からが場面ごとに違う。人は、場面と関係なく替わる。"""
    out = []
    for i in range(n):
        p = N(PEOPLE[int(i * 0.6180339887 * 4) % 4])
        if i % 2 == 0:
            tris = [(p, N("包丁"), V("持つ")), (p, N("窓"), V("見る")), (p, N("野菜"), V("切る")), (p, N("鍋"), V("煮る"))]
        else:
            tris = [(p, N("ペン"), V("持つ")), (p, N("窓"), V("見る")), (p, N("手紙"), V("書く")), (p, N("切手"), V("貼る"))]
        out.append((tris, {w for t in tris for w in t}))
    return out


def _irregular(n=48):
    """人が、話ごとに決まり無く替わる世界（切れ目をまたぐ組を聞かせても、切れ目の人は読めない）。"""
    out = []
    for i, (tris, _) in enumerate(_talks(n)):
        p = N(PEOPLE[int((i + 1) * (i + 2) * 0.6180339887) % 4])
        tris = [(p,) + tuple(t[1:]) for t in tris]
        out.append((tris, {w for t in tris for w in t}))
    return out


# ---- I49zzze: 切れ目の信号 ----

def test_auc_and_threshold_from_counts():
    assert E.auc([3, 4], [1, 2]) == 1.0 and E.auc([1, 2], [3, 4]) == 0.0 and E.auc([1, 1], [1, 1]) == 0.5 and E.auc([], [1]) is None
    vals, labs = [0.1, 0.2, 0.8, 0.9, 0.3], [False, False, True, True, False]
    th = E.choose(vals, labs)
    assert 0.3 < th < 0.8 and E.prf(vals, labs, th) == dict(適合率=1.0, 再現率=1.0, F1=1.0)


def test_boundary_signals_stand_at_real_boundaries_in_a_small_world():
    t = _irregular()
    r = E.run(t[:24], t[24:], _attr, dims=512)
    for name in E.LEARN:
        d = r[name]
        assert d["行"] == 24 * 4 and d["切れ目の行"] == 24
        s = d["信号"]
        assert s["持った波と合わない（だれが、半減期 1）"]["見分け"] > 0.7      # 人が替わる所で立つ（同じ人が続く切れ目も有るので 1 にはならない）
        assert s["遷移の驚き（どうした）"]["見分け"] is not None and "F1" in s["人が替わった"]
        assert any(k.startswith("速い波と遅い波のずれ") for k in s)
    a, b = (r[n]["信号"]["遷移の驚き（どうした）"]["切れ目の行の平均"] for n in E.LEARN)
    assert a > b                                                             # 切れ目をまたぐ組を聞かせると、切れ目での動きの驚きは下がる


def test_signals_do_not_stand_when_there_is_no_boundary():
    """話の中の並びを、どの行の後でも同じ決まりで続く 1 本の流れにすると（切れ目に意味が無い）、見分けは .5 の近くに来る。"""
    tris = [(N("太郎"), N("りんご"), V(v)) for v in ("買う", "食べる", "借りる", "読む")]
    talks = [([tris[(i * 3 + k) % 4] for k in range(3)], set()) for i in range(40)]
    r = E.run(talks[:20], talks[20:], _attr, dims=512)[E.LEARN[1]]["信号"]["遷移の驚き（%s）" % E.ALL3]
    assert abs(r["切れ目の行の平均"] - r["それ以外の行の平均"]) < 0.05           # 信号の大きさに差が出ない（見分けは、ごく小さな差の順位になるので見ない）


# ---- I49zzzf: テーマ波、話の束ね、結び付けた鍵 ----

def test_bound_key_remembers_person_and_verb_together():
    w = W.Wave(512, _attr2, W.SCALE)
    b = F.Bound(w)
    for _ in range(3):
        b.learn((N("太郎"), N("窓"), V("見る")), (N("太郎"), N("野菜"), V("切る")))
        b.learn((N("花子"), N("窓"), V("見る")), (N("花子"), N("手紙"), V("書く")))
    cands = [V("切る"), V("書く"), V("煮る")]
    assert W.pick(cands, list(b.scores((N("太郎"), N("窓"), V("見る")), cands))) == V("切る")
    assert W.pick(cands, list(b.scores((N("花子"), N("窓"), V("見る")), cands))) == V("書く")
    assert not np.any(F.Bound(w).scores((N("太郎"), N("窓"), V("見る")), cands))   # 何も聞いていなければ 0


def test_learn_together_equals_learning_every_pair_of_lines():
    tris = _scenes(2)[0][0]
    a, b = W.Wave(512, _attr2, W.SCALE), W.Wave(512, _attr2, W.SCALE)
    for i, x in enumerate(tris):
        for j, y in enumerate(tris):
            if i != j:
                a.learn(x, y)
    F.learn_together(b, tris)
    assert a.cnt == b.cnt and a.total == b.total and set(a.m) == set(b.m)
    assert all(np.allclose(a.m[k][w], b.m[k][w], atol=1e-4) for k in a.m for w in a.m[k]) and all(np.allclose(a.all[k], b.all[k], atol=1e-4) for k in a.all)


def test_theme_wave_keeps_the_scene_where_the_neighbour_table_cannot():
    t = _scenes()
    r = F.run(t[:16], t[16:], _attr2, PairBody(), sample=0, free=8, steps=2, show=1, dims=512)
    c = r["どうした"]
    assert c["問い"] == 16 * 3
    assert c[F.NEAR] < c[F.NARROW]                          # 隣の表だけでは、窓を見る の次が決まらない。話の束ねと順位を足すと決まる
    assert c[F.TOG] < c[F.NEAR]                             # 話の束ねだけでは、場面は分かるが、順番は分からない
    sc = r["場面"]
    for n in (F.TOG, F.NARROW):
        assert sc[n][0] == 8 * 2 and sc[n][1] == 16 and sc[n][2] == 16 and sc[n][3] == 0   # 出した動きも物も、全部、本物の続きに有る。別の話とは重ならない
    assert sc[F.NEAR][1] < 16 and sc[F.BND][1] < 16         # 隣の表だけ、結び付けた鍵（人は場面と関係が無い世界）は、場面を外す
    assert list(F.ranks([0.5, 2.0, 0.5, -1.0])) == [1.5, 0.0, 1.5, 3.0]
    assert len(r["自分で続ける"]) == len(F.NAMES) and len(r["自分で続ける"][0]["初めの行"]) == F.THEME


def test_theme_run_with_found_boundaries_reports_what_it_found():
    t = _scenes()
    r = F.run(t[:16], t[16:], _attr2, None, sample=0, free=0, boundary=F.MODES[1], signal="持った波と合わない（何を、半減期 1）", dims=512)
    f = r["切れ目を見つける"]
    assert f["行"] == 16 * 4 and f["切れ目の行"] == 16 and 0 <= f["当たり"] <= f["見つけた"]
    assert r["どうした"]["問い"] == 16 * 3 and r["場面"] == {}


# ---- I49zzzg: 疑問の物差し ----

def test_fill_split_and_misfit():
    assert G.fill({}) == (None, 0.0, 0.0)
    w, ent, near = G.fill({N("太郎"): 2.0, N("花子"): 2.0})
    assert w == N("太郎") and abs(ent - 1.0) < 1e-9 and near == 1.0           # 2 語が同じだけ: 1 ビット、上位 2 つは同じ
    w, ent, near = G.fill({N("太郎"): 3.0})
    assert (w, ent, near) == (N("太郎"), 0.0, 0.0)
    assert G.misfit({N("太郎"): 3.0, N("花子"): 1.0}, N("花子")) == 0.75 and G.misfit({}, N("太郎")) == 1.0
    b = G.by_bins([(i, i < 6) for i in range(8)])
    assert [x["数"] for x in b] == [2, 2, 2, 2] and [x["当たり"] for x in b] == [2, 2, 2, 0]


def test_split_is_larger_where_the_fill_is_wrong():
    """人が 1 人で続く話と、行ごとに替わる話が交ざる世界。替わる話では、前提が割れ、埋めた語も外れる。"""
    talks = []
    for i, (tris, _) in enumerate(_talks(48)):
        if i % 2:
            tris = [(N(PEOPLE[int((i * 4 + k + 1) * 0.6180339887 * 4) % 4]),) + tuple(t[1:]) for k, t in enumerate(tris)]
        talks.append((tris, set()))
    r = G.run(talks[:24], talks[24:], _attr, dims=512)
    d = r["足りない"]["だれが 半減期 1"]
    assert d["問い"] == 24 * 4 and 0 < d["当たり"] < d["問い"]
    assert d[G.SPLITS[0]]["見分け"] > 0.55 and d[G.SPLITS[1]]["見分け"] > 0.6   # 外れた行の方が、割れ方が大きい
    assert sum(x["数"] for x in d[G.SPLITS[0]]["等分"]) == d["問い"]
    m = r["合わない"]["半減期 1"]
    assert m["話の初めの行"]["行"] == 24 and sum(m["一番合わない席"].values()) == 24 * 4


# ---- I49zzzh: 持つだけの帯、薄める、切って待たせる、危ない印 ----

def _hold(halves=(1, 4, 64)):
    return D.Hold(W.Wave(512, _attr, W.SCALE), halves)


A, B, C = (N("太郎"), N("りんご"), V("買う")), (N("花子"), N("本"), V("読む")), (N("次郎"), N("パン"), V("食べる"))


def test_private_band_is_held_but_never_handed_to_the_reply_side():
    s = S.Shelter(_hold(), private=(64,))
    s.take(A)
    assert s.drive() == (1, 4)
    with pytest.raises(KeyError):
        s.query(64)
    assert np.abs(s.peek(64)[0]).sum() > 0 and s.query(4) is s.hold.v[4]      # 持ってはいる。作る側からは見える
    people = [N("太郎"), N("花子")]
    plain = S.Shelter(_hold((1, 4)))
    plain.take(A)
    assert all(np.allclose(s.query(h)[j], plain.query(h)[j]) for h in (1, 4) for j in range(3))   # 持つだけの帯が有っても、引く側の波は同じ
    assert W.pick(people, s.hold.word(1, 0, people)) == W.pick(people, plain.hold.word(1, 0, people))


def test_thin_wait_and_reopen_without_deleting():
    full, thin, cut = S.Shelter(_hold()), S.Shelter(_hold()), S.Shelter(_hold())
    full.take(A)
    thin.set_gain(0.5)
    assert thin.take(A) == S.THIN and np.allclose(thin.hold.v[4][0], 0.5 * full.hold.v[4][0], atol=1e-5)
    cut.take(A)
    before = [x.copy() for x in cut.hold.v[4]]
    cut.set_gain(0.0)
    assert cut.take(B) == S.WAIT and cut.take(C) == S.WAIT and len(cut.wait) == 2
    assert all(np.allclose(x, y) for x, y in zip(before, cut.hold.v[4]))      # 切っている間は、器は変わらない
    assert cut.set_gain(1.0) is True and cut.wait == []                       # 開くと、待たせた行を聞く（捨てない）
    n = cut.hold.n[4][0]
    assert n[N("次郎")] == 1.0 and abs(n[N("花子")] - 0.5 ** (1 / 4)) < 1e-6 and n[N("太郎")] == 1.0   # 長く待った行ほど薄い。前から持っていた物は薄めない
    assert cut.log == {S.HEARD: 1, S.THIN: 0, S.WAIT: 2, S.GATE: 0}


def test_gate_always_passes_even_while_closed():
    s = S.Shelter(_hold(), gate=lambda tri: tri[2] == V("読む"))
    s.set_gain(0.0)
    assert s.take(A) == S.WAIT and s.take(B) == S.GATE and s.flagged == [B]
    assert s.hold.n[1][0].get(N("花子")) == 1.0 and N("太郎") not in s.hold.n[1][0]    # 危ない印の行は、切っていても、そのまま聞く


def test_pressure_lowers_gain_only_when_the_signal_rises_above_its_own_past():
    p = S.Pressure()
    assert all(p.step(1.0)[1] == 1.0 for _ in range(20))                      # 変わらない信号では、閉じない
    gains = [p.step(x)[1] for x in (2.0, 3.0, 5.0, 9.0)]
    assert gains[0] == 1.0 and gains[-1] == 0.0 and gains == sorted(gains, reverse=True)   # 今までで一番重い時に、切る
    q = S.Pressure()
    assert q.step(5.0) == (0.0, 1.0)


def test_shelter_run_counts_three_ways_on_the_same_stream():
    t = _irregular()
    r = H.run(t[:24], t[24:], _attr, dims=512)
    c = r["だれが"]
    assert all(c[m]["問い"] == 24 * 4 for m in H.MODES) and c[H.MODES[0]]["半減期 1"] > 0.5 * 24 * 4
    assert sum(sum(d.values()) for d in r["段ごとの行の数"].values()) == 24 * 4
    assert r["持つだけの帯"][H.MODES[0]] == [] and r["持つだけの帯"][H.MODES[2]] == [64]
    log = r["入り方"]
    assert sum(log[H.MODES[0]].values()) == 48 * 4 and log[H.MODES[1]][S.WAIT] == 0
    assert sum(log[H.MODES[2]].values()) == 48 * 4 and log[H.MODES[2]][S.WAIT] >= r["待たせたままの行"]
