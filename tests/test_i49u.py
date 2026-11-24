"""I49u の関門: 上位の候補から話の流れで選び直す。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402
from i49u_flow_pick import FlowPick  # noqa: E402


class Fixed:
    """点を決め打ちで返す元の形。"""
    def __init__(self, pts):
        self.pts = pts

    def ask(self, si, tri, cands):
        return dict(段="解く", 点=dict(self.pts), 答え=max(self.pts, key=self.pts.get))

    def learn(self, tri):
        pass

    def tick(self):
        pass


PTS = {"勉強": 1.0, "釣る": 0.95, "泳ぐ": 0.5, "ゲーム": 0.1}
C = list(PTS)


def test_keeps_answer_without_flow_and_repicks_with_flow():
    fp = FlowPick(Fixed(PTS), V.get, pick="上位", top=2)
    assert fp.ask(2, ("海", "魚", None), C)["答え"] == "勉強"
    fp.learn(("海", "魚", "釣る"))
    assert fp.ask(2, ("海", "魚", None), C)["答え"] == "釣る" and fp.picked == 1
    fp.tick()
    assert fp.ask(2, ("海", "魚", None), C)["答え"] == "勉強"


def test_pool_limits_and_minus_self():
    for pick, kw in (("九割以内", {}), ("上位", dict(top=2)), ("出た語", dict(top=2))):
        fp = FlowPick(Fixed(PTS), V.get, pick=pick, **kw)
        fp.learn(("友達", "海", "泳ぐ"))
        assert fp.ask(2, ("海", "魚", None), C)["答え"] in ("勉強", "釣る"), pick     # 泳ぐ は上位の外
    fp = FlowPick(Fixed(PTS), V.get, pick="順位を足す")
    fp.learn(("友達", "海", "釣る"))
    assert fp.ask(2, ("海", "魚", None), C)["答え"] == "釣る"
    a = FlowPick(Fixed(PTS), V.get, pick="上位", top=2)
    b = FlowPick(Fixed(PTS), V.get, pick="上位", top=2, minus_self=True)
    for f in (a, b):
        f.learn(("海", "魚", "釣る"))
    assert a.near("釣る") > b.near("釣る")
    assert FlowPick(Fixed(PTS), V.get, pick="そのまま").ask(2, ("海", "魚", None), C)["答え"] == "勉強"


def test_run_shapes():
    import i49u_flow_run as U
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = U.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["白紙から"]["解く・属性（席あり）"]
    assert set(r) == {n for n, _ in U.PICKS} and r["そのまま"]["答えが変わった数"] == 0
    assert "当たりに変わった数" in r["流れ・上位 3"] and 0 <= r["出た語・上位 3"]["一位"] <= 1
