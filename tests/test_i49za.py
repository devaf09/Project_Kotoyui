"""I49za の関門: 板を「ふだん」と比べる（行の折り畳みの決まり）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402
from test_i49u import Fixed, PTS  # noqa: E402
from i49u_flow_pick import FlowPick  # noqa: E402

# 属性 4 つ: [どの語にも有る, 人, 海の物, 動き]
A = {"友達": [1, 1, 0, 0], "先生": [1, 1, 0, 0], "魚": [1, 0, 1, 0], "海": [1, 0, 1, 0], "釣る": [1, 0, .5, 1], "勉強": [1, 0, 0, 1]}
B = {k: np.array(v, dtype=np.complex64) for k, v in A.items()}


def test_usual_removes_what_every_word_has():
    plain = FlowPick(Fixed(PTS), B.get, pick="上位", half=4, carry=True)
    cut = FlowPick(Fixed(PTS), B.get, pick="上位", half=4, carry=True, usual="明るい分だけ")
    for fp in (plain, cut):
        for t in [("友達", "魚", "勉強"), ("先生", "海", "勉強"), ("友達", "海", "釣る")] * 3:
            fp.hear_usual(t)                                  # ふだん: どの属性も同じくらい出る
        fp.learn(("友達", "海", "釣る"))
        fp.learn(("先生", "魚", "釣る"))                        # 板: 海の物が、ふだんより多い
    assert plain.near("勉強") > 0.5                            # どの語にも有る属性だけで近く見える
    assert cut.near("魚") > cut.near("勉強") + 0.3              # ふだんを除くと、海の物が残る
    half = FlowPick(Fixed(PTS), B.get, pick="上位", usual="残す・ほかは半分")
    sub = FlowPick(Fixed(PTS), B.get, pick="上位", usual="引く")
    for fp in (half, sub):
        for t in [("友達", "魚", "勉強"), ("先生", "海", "勉強")] * 3:
            fp.hear_usual(t)
        fp.learn(("友達", "海", "釣る"))
        assert fp.near("魚") is not None and -1 <= fp.near("魚") <= 1
    assert FlowPick(Fixed(PTS), B.get, pick="上位", usual="引く").near("魚") is None


def test_seat_usual_and_run_shapes():
    fp = FlowPick(Fixed(PTS), B.get, pick="上位", by_seat=True, usual="明るい分だけ")
    fp.learn(("友達", "海", "釣る"))
    assert fp.u_n == [1, 1, 1, 3]
    import i49za_fold_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = Z.run(V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["青空の 3 つ組を先に"]
    assert len(r) == 49 and "当たりに変わった数" in r["九割以内｜席ごと｜半分まで 1 組｜明るい分だけ"]
    for d in r.values():
        assert 0 <= d["正しい物"] <= d["合わせて"] <= 1
