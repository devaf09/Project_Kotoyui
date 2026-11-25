"""I49f の比べの道具（tools/past/i49f_odd_run.py）の関門: 部品を付けた思考のループが、見本の口で回り、記録が数えられる。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i49f_odd_run as OR  # noqa: E402
from i49a_inner_vessel import Vessel  # noqa: E402
from i49c_think_loop import Words  # noqa: E402
from i49d_two_loops import Plain, Thinker  # noqa: E402
from i49f_odd_to_vessel import OddToGain  # noqa: E402


def test_mid_band_is_inside_and_part_moves_only_gain():
    assert 0 < OR.MID < 20
    rng = np.random.default_rng(0)
    ws = [(f"w{i}", rng.standard_normal(16).astype(np.complex64), {}) for i in range(30)]
    th = Thinker(Vessel(16, bands=20), Words(ws), None, Plain())
    part = th.attach(OddToGain())
    th.run(60)
    ar = part.arrays()
    assert ar["factor"].shape[1] == 20 and np.all(th.v.gain <= th.v.gmax)
