"""I49c の思考ループの関門。numpy だけで動く。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i49c_think_loop as L  # noqa: E402
from i49a_inner_vessel import Vessel  # noqa: E402


def make(budget=200):
    loop = L.Loop(Vessel(64, bands=8), L.Words(L.demo_words()))
    names = loop.attach(L.Names())
    loop.run(budget)
    return loop, names


def test_reads_every_word_by_itself_in_order():
    loop, _ = make()
    read = [w for w in loop.log["word"] if w is not None]
    assert read == [w[0] for w in L.demo_words()]            # 外から押し込まず、ループが自分で全部読む


def test_one_step_is_one_vessel_tick_and_budget_stops():
    loop, _ = make(budget=77)
    assert loop.n == 77 and loop.v.t == 77 and len(loop.log["item"]) == 77


def test_speaks_after_input_ends_then_goes_quiet():
    loop, names = make(budget=400)
    it = np.array(loop.log["item"])
    last_read = int(np.where(it == 0)[0].max())
    outs = [s for s, _ in loop.out]
    assert any(s > last_read for s in outs)                  # 読む物が無くなった後、自分から話す
    assert (it[-200:] == 3).all()                            # その後は静まる（同じ事をくり返さない）
    assert names.said and names.said[-1][1].startswith("A")  # 最後に聞いていた話題の事を言う


def test_reproducible_and_all_drives_logged():
    a, b = make()[0].arrays(), make()[0].arrays()
    assert a["drive"].shape == (200, 4)
    for k in ("item", "spread", "out_step", "act", "gain"):
        assert np.array_equal(a[k], b[k])


def test_knob_changes_behaviour_and_is_logged():
    loop = L.Loop(Vessel(64, bands=8), L.Words(L.demo_words()))
    loop.set("次へ", -10.0)                                  # 読みたさを下げると、読まなくなる
    loop.run(50)
    assert loop.counts()["次へ"] == 0 and loop.log["knob"] == [(0, "次へ", -10.0)]


def test_seal_removes_nodes_and_blocks_adjustment():
    loop, names = make(budget=30)
    seen = len(names.names)
    loop.seal()
    loop.run(30)
    assert len(names.names) == seen and loop.nodes == []     # 部品は外れ、もう呼ばれない
    with pytest.raises(RuntimeError):
        loop.set("出力", 1.0)
    with pytest.raises(RuntimeError):
        loop.attach(L.Names())
    assert len(loop.log["item"]) == 60                       # 記録は残る


def test_word_without_vector_only_advances_time():
    loop = L.Loop(Vessel(8, bands=4), L.Words([("ん", None, {"文の終わり": True})]))
    rec = loop.step()
    assert rec["項目"] == "次へ" and rec["印"] == {"文の終わり": True} and loop.v.t == 1 and not loop.v.S.any()


def test_stops_early_when_empty():
    loop = L.Loop(Vessel(8, bands=4))
    assert loop.run(100) == 1
