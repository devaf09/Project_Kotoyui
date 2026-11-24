"""I49zzd の関門: 介入で教わった事を、本体と同じ形の外付けに持つ。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49zw import attr as small  # noqa: E402


class Fixed:
    def __init__(self, word):
        self.word, self.heard = word, []

    def ask(self, si, tri, cands):
        return dict(段="本体", 点={self.word: 1.0}, 答え=self.word)

    def learn(self, tri):
        self.heard.append(tri)

    def tick(self):
        pass


class Echo:
    """聞いた 3 つ組の、その席の語を答える外付け（一番最近の物）。drop された回数も数える。"""

    def __init__(self):
        self.heard, self.dropped = [], 0

    def ask(self, si, tri, cands):
        if not self.heard:
            return dict(段="答えられない", 点={}, 答え=None)
        return dict(段="聞いた", 点={self.heard[-1][si]: 1.0}, 答え=self.heard[-1][si])

    def learn(self, tri):
        self.heard.append(tri)

    def drop(self):
        self.dropped += 1

    def tick(self):
        pass


def test_no_teaching_means_same_as_body_and_ext_hears_nothing():
    import i49zzd_taught_body as Z
    t = Z.TaughtBody(Fixed("振る"), Echo())
    o = t.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    assert o["答え"] == "振る" and t.last == (False, False, "振る", None)
    t.settle(("太郎", "りんご", "食べる"))
    assert not t.ext.heard and t.ext.dropped == 1 and not t.body.heard and t.tally == {}


def test_ext_hears_taught_triple_only_after_all_its_questions():
    import i49zzd_taught_body as Z
    t = Z.TaughtBody(Fixed("振る"), Echo())
    tri = ("太郎", "りんご", "食べる")
    t.ask(0, (None, "りんご", "食べる"), ["太郎", "花子"])
    t.teach(0, tri, "振る")
    assert not t.ext.heard                                              # 教えた直後は、まだ外付けに聞かせない
    assert t.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])["答え"] == "振る" and t.last[0] is False     # 同じ 3 つ組のほかの席で、答えを知らない
    t.settle(tri)
    assert t.ext.heard == [tri] and not t.body.heard and t.n == 1       # 本体には混ぜない


def test_ext_is_used_only_after_it_has_won_more_in_that_cell():
    import i49zzd_taught_body as Z
    t = Z.TaughtBody(Fixed("振る"), Echo())
    tri = ("太郎", "りんご", "食べる")
    t.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    t.teach(2, tri, "振る")
    t.settle(tri)
    q = ("花子", "りんご", None)
    assert t.ask(2, q, ["食べる", "振る"])["答え"] == "振る" and t.last == (True, False, "振る", "食べる")     # まだ数えが無いので本体
    t.teach(2, ("花子", "りんご", "食べる"), "振る")
    t.settle(("花子", "りんご", "食べる"))
    assert t.tally[("本体", "聞いた")] == [0, 1, 1]
    o = t.ask(2, q, ["食べる", "振る"])
    assert o["答え"] == "食べる" and o["段"] == "外付け" and t.used == dict(本体=2, 外付け=1)


def test_unheard_run_with_ext_body():
    import i49zp_text_talks_run as ZP
    import i49zx_teach_board_run as R
    lines = ["太郎が金づちを振る", "花子が金づちを振る", "",
             "花子が金づちを振る", "太郎が金づちを振る", "花子が金づちを振る", "",
             "太郎がりんごを食べる", "花子がみかんをかじる", "太郎がみかんを食べる", "",
             "花子がりんごをかじる", "太郎がりんごを食べる", "花子がみかんをかじる", "太郎がみかんを食べる"]
    talks, _, _ = ZP.talks_of(lines)
    get = lambda w: small(w[0])
    d = R.run_unheard(get, get, talks, rates=(1.0,), ks=(0,), dim=1024, ext=True)["元の順"]
    base, s = d["本体だけ（後半を聞かない）"], d["介入 100%"]
    assert s["本体"] == base["本体"] and s["本体の答えが板なしと違った数"] == 0       # 本体は変わらない（外せば元に戻る）
    assert s["教えた数"] > 0 and s["板が答えを持った問い"] > 0 and "段ごとの数え" in s
    assert s["板を付けた形"] >= base["本体"]


def test_settle_counts_only_questions_that_were_taught():
    import i49zx_teach_board as X
    import i49zzd_taught_body as Z
    t = Z.TaughtBody(Fixed("振る"), Echo())
    t.ext.heard.append(("花子", "みかん", "食べる"))
    t.n = 1
    tri = ("太郎", "りんご", "食べる")
    t.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    t.ask(0, (None, "りんご", "食べる"), ["太郎", "花子"])
    t.teach(2, tri, "振る")
    t.settle(tri)
    assert sum(h[2] for h in t.tally.values()) == 1                     # 教えが届いた席（どうした）の問いだけを数える
    t.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    t.settle(tri)
    assert sum(h[2] for h in t.tally.values()) == 1                     # 教えが無ければ、数えは進まない
    t.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    t.learn(tri)
    assert sum(h[2] for h in t.tally.values()) == 2                     # 本体が 3 つ組を聞く時は、全部数える
    b = X.Taught(Fixed("振る"), X.TeachBoard(small, dim=1024))
    b.teach(2, tri, "振る")
    b.settle(tri)
    b.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    b.settle(tri)
    assert b.tally == [0, 0, 0]                                         # 介入の板の形でも同じ
    b.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    b.teach(2, tri, "振る")
    b.settle(tri)
    assert b.tally == [0, 1, 1]


def _talks():
    import i49zp_text_talks_run as ZP
    lines = ["太郎が金づちを振る", "花子が金づちを振る", "",
             "花子が金づちを振る", "太郎が金づちを振る", "花子が金づちを振る", "",
             "太郎がりんごを食べる", "花子がみかんをかじる", "太郎がみかんを食べる", "",
             "花子がりんごをかじる", "太郎がりんごを食べる", "花子がみかんをかじる", "太郎がみかんを食べる"]
    return ZP.talks_of(lines)[0]


def test_direct_mixes_taught_triples_into_body():
    import i49zzd_taught_body as Z
    d = Z.Direct(Fixed("振る"))
    tri = ("太郎", "りんご", "食べる")
    d.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    d.settle(tri)
    assert not d.body.heard                                             # 教えが無ければ、本体は聞かない
    d.teach(2, tri, "振る")
    assert not d.body.heard
    d.settle(tri)
    assert d.body.heard == [tri] and d.n == 1


def test_unheard_run_direct_and_only_when_wrong():
    import i49zx_teach_board_run as R
    get = lambda w: small(w[0])
    s = R.run_unheard(get, get, _talks(), rates=(1.0,), ks=(0,), dim=1024, ext=True, direct=True)["元の順"]["介入 100%"]
    assert s["本体に直に混ぜる（比べ役）"]["教えた数"] == s["教えた数"] > 0 and 0 <= s["本体に直に混ぜる（比べ役）"]["当たり"] <= 1
    used = sum(c["外付けを使った問い"] for c in s["外付けを使った段の組"].values())
    assert used == s["使った答え"]["外付け"]
    assert sum(c["当たりに変わった"] for c in s["外付けを使った段の組"].values()) == s["当たりに変わった"]
    w = R.run_unheard(get, get, _talks(), rates=(1.0,), ks=(0,), dim=1024, ext=True, when="外れた時")["元の順"]["介入 100%"]
    assert 0 < w["教えた数"] <= s["教えた数"] and w["本体の答えが板なしと違った数"] == 0


def test_reflect_moves_taught_triples_into_body_and_empties_ext():
    import i49zzd_taught_body as Z
    t = Z.TaughtBody(Fixed("振る"), Echo(), fresh=Echo, reflect=True)
    tri = ("太郎", "りんご", "食べる")
    t.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    t.teach(2, tri, "振る")
    t.settle(tri)
    assert t.ext.heard == [tri] and not t.body.heard                    # 話の途中は、外付けだけが知っている
    t.rest()
    assert t.body.heard == [tri] and not t.ext.heard and t.n == 0 and t.reflected == 1     # 話の終わりに本体へ反映し、外付けは空
    t.rest()
    assert t.body.heard == [tri]                                        # 反映する物が無ければ、何もしない


def test_save_and_restore_bring_body_back():
    import i49zzd_taught_body as Z
    t = Z.TaughtBody(Fixed("振る"), Echo(), fresh=Echo, reflect=True)
    snap = t.save()
    tri = ("太郎", "りんご", "食べる")
    t.teach(2, tri, "振る")
    t.settle(tri)
    t.rest()
    assert t.body.heard == [tri]
    t.restore(snap)
    assert t.body.heard == [] and not t.ext.heard                       # セーブした所まで戻る
    assert snap.heard == []                                             # セーブそのものは、反映で書き換わらない


def test_unheard_run_reflect():
    import i49zx_teach_board_run as R
    get = lambda w: small(w[0])
    d = R.run_unheard(get, get, _talks(), rates=(1.0,), ks=(0,), dim=1024, ext=True, direct=True, reflect=True)["元の順"]
    s = d["介入 100%"]
    r = s["話の終わりに本体へ反映する"]
    assert r["反映した三つ組"] == s["教えた数"] > 0 and r["セーブから戻した本体の答えが元と違った数"] == 0
    assert r["当たり"] >= d["本体だけ（後半を聞かない）"]["本体"]
