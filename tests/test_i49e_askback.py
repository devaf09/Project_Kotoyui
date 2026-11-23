"""I49e の聞き返しのあるやり取り（intervention/i49e_askback.py）の関門。先生は見本の関数で確かめる。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
from i49a_inner_vessel import Vessel  # noqa: E402
from i49c_think_loop import Words  # noqa: E402
from i49d_two_loops import Pick, Scored, Thinker  # noqa: E402
from i49e_askback import Exchange  # noqa: E402
from i49e_listen import LivePickup  # noqa: E402
from i49e_three import Seat, live_weight  # noqa: E402


class Flat:
    def line(self, keys, marks):
        return [dict(間=i, 詰め物="", 割合={}) for i in range(len(keys))]


def vec(i, dim=32):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


NAMES = ["猫", "魚", "山", "川", "空", "犬", "海", "舟"]
V = {n: vec(i) for i, n in enumerate(NAMES)}
V["子猫"] = (vec(0) + 0.2 * vec(9)).astype(np.complex64)      # 「猫」に近い語
V["小魚"] = (vec(1) + 0.2 * vec(10)).astype(np.complex64)     # 「魚」に近い語


def make(k=2):
    live, pos_of = LivePickup(), {}
    th = Thinker(Vessel(32, bands=8), Words(), None, Scored(Flat(), pos_of, pick=Pick(live_weight(live, pos_of), k)))
    return Seat(th, live, pos_of)


def turn(*names):
    return [(n, "NOUN", V.get(n)) for n in names]


def test_ends_without_forcing_speech():
    seat = make()
    seat.listen("本", turn("猫"))
    out = Exchange(seat, lambda line: 1 / 0).run()        # 言わなければ先生は呼ばれない
    assert out["終わり"] == "話さなかった" and out["番"] == []


def test_understood_goes_on_until_nothing_new():
    seat = make()
    seat.listen("本", turn("猫", "魚"))
    replies = iter([turn("山", "川"), turn("山")])

    def teacher(line):
        return dict(分かった=True, 返事=next(replies), 意味=[])

    out = Exchange(seat, teacher, rounds=5).run()
    assert [t["分かった"] for t in out["番"]] == [True, True]
    assert out["番"][1]["行"] == ["山", "川"] and not out["番"][1]["言い直し"]
    assert out["終わり"] == "言う事が無くなった"           # 2 回目の返事は語が 1 つ: 材料が少ないので言わない


def test_asked_back_resays_with_the_words_near_what_it_meant():
    seat = make()
    seat.listen("本", turn("猫", "魚"))
    calls = []

    def teacher(line):
        calls.append(list(line))
        if len(calls) == 1:                               # 分からない: 聞き返し（遠い語）と「こういう意味？」（近い語 2 つ + 遠い語）
            return dict(分かった=False, 返事=turn("山", "川"), 意味=turn("子猫", "海", "小魚"))
        return dict(分かった=True, 返事=turn("空"), 意味=[])

    out = Exchange(seat, teacher, rounds=5).run()
    assert calls[0] == ["猫", "魚"]
    assert calls[1] == ["子猫", "小魚"]                    # 言いたかった向きに近い語を選んで言い直す
    t = out["番"]
    assert t[1]["言い直し"] and t[1]["分かった"] and out["言い直しで分かった"] == 1
    assert all(v > 0.5 for v in t[1]["近さ"].values())
    assert t[0]["拾った先生の語"] == 2
    w = seat.th.builder.pick.weight_of                    # 選ぶ重さは、言い直しの後で元に戻る
    assert w("猫") == seat.live.value("猫", "NOUN")


def test_stops_at_rounds_and_when_it_cannot_resay():
    seat = make()
    seat.listen("本", turn("猫", "魚"))
    out = Exchange(seat, lambda line: dict(分かった=False, 返事=turn("山", "川"), 意味=turn("子猫", "小魚")), rounds=2).run()
    assert out["終わり"] == "番の数に届いた" and len(out["番"]) == 2
    seat = make()
    seat.listen("本", turn("猫", "魚"))
    out = Exchange(seat, lambda line: dict(分かった=False, 返事=turn("山"), 意味=[]), rounds=4).run()
    assert out["終わり"] == "言い直せなかった" and len(out["番"]) == 1
