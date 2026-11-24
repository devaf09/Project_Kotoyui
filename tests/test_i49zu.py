"""I49zu の関門: 「属性の似た語に負ける」の数えを、切り出しを通さない材料で数え直す道具。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import V  # noqa: E402
from test_i49zt import LINES  # noqa: E402


def test_run_shapes():
    import i49zp_text_talks_run as ZP
    import i49zu_who_wins_clean_run as U
    talks, _, _ = ZP.talks_of(LINES)
    res = U.run(lambda w: V.get(w[0]), talks)
    assert list(res) == ["元の順", "並べ替え 1"]
    for d in res.values():
        for name in ("属性だけ", "属性＋固有の位相", "属性＋固有の位相＋振幅"):
            assert 0 <= d[name]["正解が二〜三位だった割合"] <= 1 and 0 <= d[name]["正解が一位だった割合"] <= 1
