"""I49e 関係の表を回す道具（tools/i49e_relation_run.py）の関門。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i49e_relation_run as RR  # noqa: E402
from i49e_relation import Relation  # noqa: E402


def e(i, dim=16):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


V = {"猫": e(0), "子猫": e(0) + 0.1 * e(9), "魚": e(2), "食べる": e(3), "捕る": e(4)}


def test_strict_first_puts_ties_in_front():
    assert RR.strict_first({"a": 2.0, "b": 1.0}, "a") and not RR.strict_first({"a": 1.0, "b": 1.0}, "a")
    assert not RR.strict_first({}, "a") and not RR.strict_first({"a": 0.0}, "a")


def test_no_borrow_does_not_borrow():
    nb = RR.no_borrow_class()(V.get)
    nb.learn(("猫", "魚", "食べる"))
    assert nb.ask(2, ("子猫", "空", None), ["食べる", "捕る"])["段"] == "答えられない"
    r = Relation(V.get)
    r.learn(("猫", "魚", "食べる"))
    assert r.ask(2, ("子猫", "魚", None), ["食べる", "捕る"])["段"] == "片方だけ借りる"


def test_stream_asks_before_learning():
    talks = [([("猫", "魚", "食べる")], {"猫", "魚", "食べる", "捕る"})] * 2
    rows = RR.stream(Relation(V.get), talks, every=1)
    assert rows[0]["白紙"] == 1.0 and rows[0]["一位"] == 0.0     # 初めは白紙（聞く前に問う）
    assert rows[1]["一位"] == 1.0 and rows[1]["段"]["2語"] == 1.0
