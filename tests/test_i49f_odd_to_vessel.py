"""I49f 違和感 → 器（intervention/i49f_odd_to_vessel.py）の関門。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
from i49a_inner_vessel import Vessel  # noqa: E402
from i49c_think_loop import Words  # noqa: E402
from i49d_two_loops import Near, Thinker, demo_world  # noqa: E402
from i49f_odd_to_vessel import OddToGain  # noqa: E402


def run(node=None, steps=300):
    vocab, talk = demo_world()
    th = Thinker(Vessel(64, bands=8), Words(), Near(list(vocab), list(vocab.values())))
    if node is not None:
        th.attach(node)
    for s in talk:
        th.words.push(s)
        th.run(40)
    th.run(steps)
    return th


def test_without_the_part_nothing_changes():
    a, b = run(), run()
    assert np.array_equal(np.array(a.v.log["gain"]), np.array(b.v.log["gain"])) and a.said == b.said


def test_stronger_than_usual_raises_gain_and_weaker_lowers():
    node = OddToGain()
    th = run(node)
    arr = node.arrays()
    assert arr["diff"].shape == arr["factor"].shape and arr["diff"].shape[1] == 8
    up, down = arr["diff"] > 0, arr["diff"] < 0
    assert up.any() and down.any()
    assert (arr["factor"][up] > 1).all() and (arr["factor"][down] < 1).all()
    assert (th.v.gain <= th.v.gmax + 1e-6).all() and (th.v.gain >= 1.0 / th.v.gmax - 1e-6).all()


def test_gain_differs_from_the_plain_run():
    plain, th = run(), run(OddToGain())
    assert not np.allclose(np.array(plain.v.log["gain"]), np.array(th.v.log["gain"]))


def test_seal_takes_the_part_off():
    vocab, _ = demo_world()
    th = Thinker(Vessel(64, bands=8), Words(), Near(list(vocab), list(vocab.values())))
    th.attach(OddToGain())
    th.seal()
    assert th.nodes == []
