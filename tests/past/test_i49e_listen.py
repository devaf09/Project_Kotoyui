"""I49e の聞くだけで育つ拾われやすさ（intervention/past/i49e_listen.py）の関門。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
from i49e_listen import LivePickup, auc  # noqa: E402

N, V = "NOUN", "VERB"


def talk(lp, rounds):
    for _ in range(rounds):
        lp.hear("A", [("猫", N), ("見る", V)])
        lp.hear("B", [("猫", N), ("可愛い", "ADJ")])       # B は「猫」を拾い、「見る」は拾わない


def test_grows_by_listening_only():
    lp = LivePickup()
    assert lp.value("猫", N) == lp.value("見る", V)        # 聞く前は同じ
    talk(lp, 10)
    assert lp.value("猫", N) > 0.8 and lp.value("見る", V) < 0.2
    r = lp.report()
    assert r["後半"]["聞いて育てた見込み"] == 1.0
    assert r["全体"]["初めの見込みだけ"] is not None and r["全体"]["初めの見込みだけ"] < 0.9


def test_prior_is_used_until_heard_and_same_speaker_is_not_counted():
    lp = LivePickup(prior=lambda l, p: 0.9 if l == "犬" else None)
    assert lp.value("犬", N) == 0.9
    lp.hear("A", [("犬", N)])
    lp.hear("A", [("鳥", N)])                              # 同じ話し手が続けた時は数えない
    assert not lp.rec and lp.value("犬", N) == 0.9
    lp.hear("B", [("魚", N)])                              # 「鳥」は拾われなかった
    assert lp.value("鳥", N) < 0.5 and len(lp.rec) == 1


def test_memory_fades_but_is_not_deleted():
    lp = LivePickup(half=2)
    talk(lp, 3)
    h = lp.heard[("猫", N)]
    for i in range(20):
        lp.hear("A" if i % 2 else "B", [(f"w{i}", N)])
    assert 0 < lp.heard[("猫", N)] < h * 0.01
    assert auc([0.1, 0.9], [False, True]) == 1.0 and auc([1, 2], [True, True]) is None


def test_base_rate_is_kept_per_speaker_pair():
    lp = LivePickup()
    for _ in range(20):
        lp.hear("本", [("山", N), ("川", N), ("海", N)])     # 本の語は、ほとんど拾われない
        lp.hear("A", [("猫", N)])
        lp.hear("B", [("猫", N)])                          # A の語は、B にいつも拾われる
    assert lp._prior(("新しい語", N), ("本", "A")) < 0.1 < 0.9 < lp._prior(("新しい語", N), ("A", "B"))
