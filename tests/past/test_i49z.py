"""I49z の関門: 薄れる仮の板、席ごとの板。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402
from test_i49u import Fixed, PTS, C  # noqa: E402
from i49u_flow_pick import FlowPick  # noqa: E402


def test_board_fades_and_carries():
    fp = FlowPick(Fixed(PTS), V.get, pick="上位", top=2, half=1, carry=True)
    fp.learn(("海", "魚", "釣る"))
    first = fp.ctx.copy()
    fp.tick()
    assert fp.ctx is not None                               # 話をまたいで持つ
    fp.learn(("学校", "先生", "勉強"))
    rest = fp.ctx - (V["学校"] + V["先生"] + V["勉強"])
    assert np.allclose(rest, first * 0.5, atol=1e-4)         # 1 組で半分に薄れる
    plain = FlowPick(Fixed(PTS), V.get, pick="上位", top=2)
    plain.learn(("海", "魚", "釣る"))
    plain.tick()
    assert plain.ctx is None


def test_seat_board_only_holds_that_seat():
    fp = FlowPick(Fixed(PTS), V.get, pick="上位", top=2, by_seat=True)
    fp.learn(("海", "魚", "釣る"))
    assert np.allclose(fp.seat_ctx[2], V["釣る"]) and np.allclose(fp.seat_ctx[0], V["海"])
    assert fp.near("釣る", 2) > 0.99 and fp.near("釣る", 0) < 0.9
    assert fp.ask(2, ("海", "魚", None), C)["答え"] == "釣る"


def test_run_shapes():
    import i49z_board_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = Z.run(V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["白紙から"]
    assert len(r) == 21 and r["板無し"]["答えが変わった数"] == 0
    for d in r.values():
        assert 0 <= d["正しい物"] <= d["合わせて"] <= 1
