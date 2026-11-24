"""I49zb の関門: いいとこ取りの形。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402
from test_i49u import Fixed, PTS, C  # noqa: E402
from i49u_flow_pick import FlowPick  # noqa: E402
from i49zb_mix import Mix  # noqa: E402


class Level(Fixed):
    def __init__(self, lv, ans):
        self.lv, self.ans = lv, ans

    def ask(self, si, tri, cands):
        return dict(段=self.lv, 点={self.ans: 1.0} if self.ans else {}, 答え=self.ans)


def test_mix_uses_table_on_upper_levels_and_solve_below():
    q = (2, ("海", "魚", None), C)
    assert Mix(Level("2語", "泳ぐ"), Fixed(PTS)).ask(*q)["答え"] == "泳ぐ"
    m = Mix(Level("もう片方", "泳ぐ"), Fixed(PTS))
    o = m.ask(*q)
    assert o["答え"] == "勉強" and o["段"] == "解く" and m.used == dict(表=0, 解く=1)
    assert Mix(Level("答えられない", None), Fixed(PTS)).ask(*q)["答え"] == "勉強"
    assert Mix(Level("借りる", "泳ぐ"), Level("答えられない", None)).ask(*q)["答え"] == "泳ぐ"


def test_said_then_board_pick():
    fp = FlowPick(Fixed(PTS), V.get, pick="出た語→板", top=2, by_seat=True, half=1, carry=True)
    fp.learn(("海", "魚", "釣る"))
    assert fp.ask(2, ("海", "魚", None), C)["答え"] == "釣る"       # この話で出た語
    fp.tick()
    assert fp.said == {} and fp.ctx is not None                  # 出た語は話ごとに空、板は持ち越す
    assert fp.ask(2, ("海", "魚", None), C)["答え"] in ("勉強", "釣る")
    near = FlowPick(Fixed(PTS), V.get, pick="出た語・九割以内")
    near.learn(("友達", "海", "泳ぐ"))
    assert near.ask(2, ("海", "魚", None), C)["答え"] == "勉強"     # 泳ぐ は九割の外


def test_run_shapes():
    import i49zb_mix_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = Z.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["青空の 3 つ組を先に"]
    assert set(r) == {v[0] for v in Z.VERS}
    u = r["表 → 解く"]["使った答え"]
    assert u["表"] + u["解く"] == 12 and "当たりに変わった数" in r["表 → 解く"]


def test_mix_counts_which_side_hits_per_level():
    m = Mix(Level("もう片方", "泳ぐ"), Fixed(PTS), upper=None)
    q = (2, ("海", "魚", None), C)
    assert m.ask(*q)["答え"] == "泳ぐ"                           # 数えが無い間は表
    m.learn(("海", "魚", "勉強"))                                # 解く形（勉強）が当たり
    assert m.tally["もう片方"] == [0, 1, 1]
    assert m.ask(*q)["答え"] == "勉強"                           # 次からは解く形
    m.learn(("海", "魚", "泳ぐ"))
    m.learn(("海", "魚", "泳ぐ"))                                # 問わずに聞いた分は数えない
    assert m.tally["もう片方"] == [1, 1, 2]
    assert m.ask(*q)["答え"] == "泳ぐ"                           # 同じなら表
