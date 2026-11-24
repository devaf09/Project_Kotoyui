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
