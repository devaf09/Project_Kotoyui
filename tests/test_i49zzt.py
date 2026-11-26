"""I49zzt: 言葉の 3 つ組を掛け合わせた波で持つ部品（小さい世界）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzs_bind_run as R
import i49zzt_triple as T


def _world(hold=0.2):
    tr, te, n = R.class_world(4, 5, hold)
    name = lambda t: (("誰%d" % t[0], "NOUN"), ("何%d" % t[1], "NOUN"), ("する%d" % t[2], "VERB"))
    verbs = [("する%d" % i, "VERB") for i in range(4)]
    return [name(t) for t in tr], [name(t) for t in te], verbs


def _acc(m, tris, verbs):
    return sum(m.ask(2, (t[0], t[1], None), verbs)["答え"] == t[2] for t in tris) / len(tris)


def test_count_recalls_heard_triples_but_not_unheard():
    tr, te, verbs = _world()
    m = T.TripleCount(1024)
    for t in tr:
        m.learn(t)
    assert _acc(m, tr, verbs) > 0.95
    assert _acc(m, te, verbs) < 0.5


def test_nudge_reaches_unheard_pairs():
    tr, te, verbs = _world()
    m = T.TripleNudge(1024, eta=0.3)
    for _ in range(12):
        if sum(m.learn(t, verbs) for t in tr) == 0:
            break
    assert _acc(m, tr, verbs) > 0.95
    assert _acc(m, te, verbs) > 0.6


def test_new_words_get_their_own_phase_and_dtype():
    m = T.TripleNudge(64)
    a, b = m.row(0, "猫"), m.row(0, "犬")
    assert (a, b) == (0, 1) and m.row(0, "猫") == 0
    assert m.th[0].dtype == np.float32 and not np.allclose(m.th[0][0], m.th[0][1])
    o = m.ask(1, ("猫", None, "食べる"), ["魚", "骨"])
    assert o["答え"] in ("魚", "骨") and set(o["点"]) == {"魚", "骨"}


def test_nudge_does_nothing_when_right():
    m = T.TripleNudge(64)
    tri = ("猫", "魚", "食べる")
    first = m.ask(2, tri, ["食べる", "かじる"])["答え"]
    tri = ("猫", "魚", first)
    for _ in range(50):
        if m.learn(tri, ["食べる", "かじる", "魚", "骨", "猫", "犬"]) == 0:
            break
    keep = [t.copy() for t in m.th]
    assert m.learn(tri, ["食べる", "かじる", "魚", "骨", "猫", "犬"]) == 0
    assert all(np.array_equal(x, y) for x, y in zip(keep, m.th))


def test_run_asks_every_form_the_same_questions():
    import i49zzt_triple_run as RUN
    from test_i49zw import attr as small
    from test_i49zzg import _talks
    get = lambda w: small(w[0])
    forms = {"動かす": (256, 0.3, 2)}
    d = RUN.run(get, get, _talks(), ks=(0,), forms=forms, count_dims=256)
    c = d["元の順"]["全部"]
    assert c["問い"] == 6 * 2 * 3                                    # 後半 6 話 × 3 つ組 2 つ × 席 3 つ
    assert c["問い"] == d["元の順"]["2 語の組を前に聞いた"]["問い"] + d["元の順"]["2 語の組は初めて"]["問い"]
    assert all(0 <= c[n] <= c["問い"] for n in ("本体", "掛け合わせて数える", "動かす"))
    assert d["合計"]["全部"]["問い"] == c["問い"]


def _class_attr():
    """類ごとの属性（類の印 1 つ ＋ 語ごとの印 1 つ）。類は、誰が と 何を で別の決め方（i49zzs_bind_run.class_world と同じ）。"""
    g, m = 4, 5
    n = g * m
    mix = R.scramble(n)
    cls = {"誰": [mix.index(i) % g for i in range(n)], "何": [mix[i] % g for i in range(n)]}

    def attr(w):
        name = w[0]
        if name[0] not in cls:
            return None
        i = int(name[1:])
        v = np.zeros(g + n, dtype=np.complex64)
        v[cls[name[0]][i]] = 1.0
        v[g + i] = 0.5
        return v
    return attr


def test_attribute_phases_let_counting_reach_unheard_pairs():
    """位相を属性から始めると、似た語の組の数えが届くので、数えるだけで聞いていない組が当たる。"""
    tr, te, verbs = _world(0.5)
    plain, withattr = T.TripleCount(2048), T.TripleCount(2048, attr_of=_class_attr())
    for t in tr:
        plain.learn(t)
        withattr.learn(t)
    assert _acc(plain, te, verbs) < 0.5
    assert _acc(withattr, te, verbs) > 0.8
    assert _acc(withattr, tr, verbs) > 0.8


def test_similar_attributes_start_with_near_phases():
    m = T.TripleNudge(1024, attr_of=_class_attr())
    r = [m.row(0, ("誰%d" % i, "NOUN")) for i in range(20)]
    attr = _class_attr()
    same = [(i, j) for i in range(20) for j in range(i) if np.argmax(np.real(attr(("誰%d" % i, 0))[:4])) == np.argmax(np.real(attr(("誰%d" % j, 0))[:4]))]
    diff = [(i, j) for i in range(20) for j in range(i) if (i, j) not in same]
    near = lambda ps: float(np.mean([np.cos(m.th[0][r[i]] - m.th[0][r[j]]).mean() for i, j in ps]))
    assert near(same) > near(diff) + 0.2


def test_run_with_attribute_forms():
    import i49zzt_triple_run as RUN
    from test_i49zw import attr as small
    from test_i49zzg import _talks
    get = lambda w: small(w[0])
    d = RUN.run(get, get, _talks(), ks=(0,), forms={}, count_dims=256, attr=True, attr_forms={"属性で数える": (None, 2.0), "属性で動かす": (0.1, 2.0)})
    c = d["元の順"]["全部"]
    assert all(0 <= c[n] <= c["問い"] for n in ("属性で数える", "属性で動かす"))
