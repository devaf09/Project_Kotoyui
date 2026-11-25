"""I49e の席（intervention/past/i49e_three.py）の関門。行づくりの型は使わず、見本の口で確かめる。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
from i49a_inner_vessel import Vessel  # noqa: E402
from i49c_think_loop import Words  # noqa: E402
from i49d_two_loops import Pick, Scored, Thinker  # noqa: E402
from i49e_listen import LivePickup  # noqa: E402
from i49e_three import ME, Seat, live_weight  # noqa: E402


class Flat:
    """見本の行づくり: 詰め物なし。"""

    def line(self, keys, marks):
        return [dict(間=i, 詰め物="", 割合={}) for i in range(len(keys))]


def vec(i, dim=32):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    x[(i + 1) % dim] = 0.3
    return x


NAMES = ["猫", "山", "川", "魚", "空", "犬"]
V = {n: vec(i) for i, n in enumerate(NAMES)}


def make(k=2):
    live, pos_of = LivePickup(), {}
    th = Thinker(Vessel(32, bands=8), Words(), None, Scored(Flat(), pos_of, pick=Pick(live_weight(live, pos_of), k)))
    return Seat(th, live, pos_of), live


def turn(*names):
    return [(n, "NOUN", V.get(n)) for n in names]


def test_listens_without_speaking_then_speaks_on_its_turn():
    seat, live = make()
    seat.listen("本", turn("山", "川", "空"))
    seat.listen("A", turn("猫", "魚"))
    assert not seat.th.said                               # 番の途中では話さない
    line = seat.speak()
    assert line is not None and len(line) == 2 and set(line) <= set(NAMES)
    assert live.last[0] == ME                             # 声に出した行は、番として場に出る
    assert not seat.th.tray.items                         # 言い終えたら入れ物は空


def test_picks_what_gets_picked_up():
    seat, live = make()
    for _ in range(6):                                    # 「猫」はいつも拾われ、「山・川」は拾われない、を聞いて育つ
        live.hear("A", [("猫", "NOUN"), ("山", "NOUN"), ("川", "NOUN")])
        live.hear("B", [("猫", "NOUN")])
    seat.listen("A", turn("山", "川", "猫", "魚"))
    line = seat.speak()
    assert "猫" in line and not ({"山", "川"} & set(line))


def test_feedback_comes_from_the_next_turn():
    seat, live = make()
    seat.listen("A", turn("猫", "魚", "空"))
    line = seat.speak()
    n0 = len(live.rec)
    seat.listen("B", turn(line[0], "犬"))                 # 先生が、ことゆいの語を 1 つ拾った
    got = live.rec[n0:]
    assert len(got) == len(line) and sum(r[3] for r in got) == 1
    assert (ME, "B") in live.pair


def test_silent_stage_and_no_forced_speech():
    seat, live = make()
    seat.listen("A", turn("猫", "魚"))
    line = seat.speak(aloud=False)
    assert line is not None and live.last[0] == "A"       # 聞くだけの段: 行は作るが、場には出さない
    seat.listen("A", turn("猫"))
    assert seat.speak() is None                           # 材料が 1 つでは話さない（無理には話させない）
    assert seat.log[-1][1] is None
    n = len(seat.th.said)
    seat.listen("B", turn("山", "川"))
    assert len(seat.th.said) == n                         # 話さなかった後も、次の番の途中では話し出さない
    assert seat.speak() is not None
