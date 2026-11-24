"""I49ze の関門: 語の波を並べて持つ仮の板。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


class _Rel:
    def ask(self, si, tri, cands):
        return {"答え": None, "点": {}}

    def learn(self, tri):
        pass

    def tick(self):
        pass


def test_rows_near_and_fade():
    from i49u_flow_pick import FlowPick
    fp = FlowPick(_Rel(), V.get, pick="九割以内", half=1, by_seat=True, carry=True, board="並べる")
    assert fp.near("友達", 0) is None
    fp.learn(("友達", "魚", "釣る"))
    assert abs(fp.near("友達", 0) - 1.0) < 1e-5            # 同じ語は近さ 1
    assert fp.near("友達", 1) < 0.999                        # 別の席の板には居ない
    fp.learn(("先生", "海", "泳ぐ"))
    assert fp.near("友達", 0) >= 0.5 - 1e-5 and fp.near("友達", 0) < 1.0 - 1e-5   # 1 組で半分（別の語の方が近ければそちら）
    assert abs(fp.near("先生", 0) - 1.0) < 1e-5
    fp.tick()
    assert fp.rows                                           # 話をまたいでも残る
    for _ in range(200):
        fp.learn(("先生", "海", "泳ぐ"))
    assert len(fp.rows) < 200 * 3                            # 薄れ切った波は並びから外れる
    assert all(isinstance(v, float) and v > 0 for v in fp.row_w)


def test_rows_reset_without_carry():
    from i49u_flow_pick import FlowPick
    fp = FlowPick(_Rel(), V.get, pick="上位", board="並べる")
    fp.learn(("友達", "魚", "釣る"))
    assert abs(fp.near("魚") - 1.0) < 1e-5
    fp.tick()
    assert fp.near("魚") is None


def test_run_shapes():
    import i49ze_row_board_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = Z.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["白紙から"]
    assert len(r) == 19 and r["板無し"]["答えが変わった数"] == 0
    assert "当たりに変わった数" in r["並べた板｜九割以内｜席ごと｜半分まで 1 組"]
    for d in r.values():
        assert 0 <= d["正しい物"] <= d["合わせて"] <= 1
    assert np.isfinite(r["並べた板｜上位 3｜1 枚｜薄れない"]["正しい物"])
