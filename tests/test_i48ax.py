"""I48ax（言えるか）の確かめ（小さな手作りの語板）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_sayable_judge():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    say = AX.Sayable(S, [sent("友達"), sent("犬")])
    x = AE.items(S, sent("友達"), "節")[0]              # 友達 と 散歩
    assert say.judge(x, "と") == "言える" and say.judge(x, "の") == "言えない"
    x2 = AE.items(S, sent("先生"), "節")[0]             # 先生 と 散歩: 組は学びに無い。左の語も無い
    assert say.judge(x2, "と") == "分からない"
    x3 = dict(x2, l="犬")                                # 犬 と 散歩: 組は無いが、左「犬」・右「散歩」に それぞれ「の」がある
    x3["r"] = "散歩"
    assert say.judge(dict(x3, l="友達", r="行く"), "と") in ("言えそう", "分からない")
