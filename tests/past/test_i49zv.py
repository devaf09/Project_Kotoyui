"""I49zv の関門: 席を増やした時の、正解の順位の移り方を数える道具。"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import V  # noqa: E402
from test_i49zr import LINES  # noqa: E402


def test_run_shapes():
    import i49zr_frame_run as R
    import i49zv_frame_who_wins_run as W
    talks, _, _ = R.talks_of(LINES)
    res = W.run(lambda w: V.get(w[0]), talks)
    assert list(res) == ["元の順", "並べ替え 1"]
    for d in res.values():
        n = d["見せない"]["問い"]
        assert n > 0 and n == d["見せる"]["問い"] == sum(sum(r.values()) for r in d["移り方"].values())
        for side in ("見せない", "見せる"):
            assert abs(sum(d[side][r] for r in W.RANKS) - 1) < 1e-3
            assert sum(sum(s.values()) for s in d[side]["席ごと"].values()) == n
