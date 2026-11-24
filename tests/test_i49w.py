"""I49w の関門: 正しい物と、正しいと言える物を両方数える。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


class Always:
    def __init__(self, w):
        self.w = w

    def ask(self, si, tri, cands):
        return dict(段="x", 点={self.w: 1.0}, 答え=self.w if si == 2 else None)

    def learn(self, tri):
        pass

    def tick(self):
        pass


def test_grades_right_and_also_right():
    import i49w_also_right_run as T
    W = set(NAMES)
    talks = [([("友達", "海", "泳ぐ"), ("友達", "海", "釣る")], W)]
    ok_t = T.ok_sets(t for tris, _ in talks for t in tris)
    ok_w = T.ok_sets([t for tris, _ in talks for t in tris] + [("友達", "海", "勉強")])
    assert ok_t[(2, ("友達", "海", None))] == {"泳ぐ", "釣る"}
    g = T.grade(Always("泳ぐ"), talks, ok_t, ok_w)
    assert g["問い"] == 6 and g["正しい物"] == round(1 / 6, 4)
    assert g["話だけ"]["正しいと言える物"] == round(1 / 6, 4) and g["話だけ"]["合わせて"] == round(2 / 6, 4)
    g = T.grade(Always("勉強"), talks, ok_t, ok_w)
    assert g["正しい物"] == 0 and g["話だけ"]["合わせて"] == 0 and g["話と青空"]["合わせて"] == round(2 / 6, 4)
    assert g["話と青空"]["席ごと"]["どうした"]["合わせて"] == 1.0


def test_run_shapes():
    import i49w_also_right_run as T
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = T.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["青空の 3 つ組を先に"]["解く・属性（席あり）"]
    assert r["問い"] == 9 and r["話だけ"]["合わせて"] >= r["正しい物"] and r["話と青空"]["合わせて"] >= r["話だけ"]["合わせて"]
