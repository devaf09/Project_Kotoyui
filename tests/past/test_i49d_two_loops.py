"""I49d: 思考のループと感情・状態のループを分けた最小版の関門。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
from i49a_inner_vessel import Vessel  # noqa: E402
from i49c_think_loop import Node, Words  # noqa: E402
from i49d_two_loops import ACTS, Near, OddWave, Pick, Plain, Scored, Thinker, Tray, demo_world  # noqa: E402


def make(talk_idx=(0,)):
    vocab, talk = demo_world()
    th = Thinker(Vessel(64, bands=8), Words(), Near(list(vocab), list(vocab.values())))
    for i in talk_idx:
        th.words.push(talk[i])
    return th, vocab, talk


def acts(th):
    return [ACTS[i] for i in th.log["act"]]


def test_tray_keeps_one_per_name_and_upgrades_to_heard():
    t = Tray()
    v = np.ones(4)
    t.add("x", v, "連想", 0)
    t.add("x", v, "連想", 1)
    assert t.names("連想") == ["x"]
    t.add("x", v, "語", 2)
    assert t.names("語") == ["x"] and len(t.items) == 1
    t.add("文の終わり", None, "印", 2)
    assert t.has_mark("文の終わり") and len(t.vecs()) == 1


def test_lack_is_one_for_empty_tray_and_zero_for_empty_vessel():
    th, vocab, _ = make(())
    assert th.lack() == 0.0
    th.v.hear(vocab["A0"])
    assert th.lack() == 1.0
    th.tray.add("A0", vocab["A0"], "語", 0)
    assert th.lack() < 0.5


def test_speaks_only_after_sentence_end_mark():
    th, _, talk = make((0,))
    th.run(len(talk[0]) * 2 + 2)
    a = acts(th)
    assert a.count("出力") == 1
    assert a.index("出力") > max(i for i, x in enumerate(a) if x == "読む")
    assert sorted(th.said[0][1]) == sorted(n for n, _, _ in talk[0])


def test_does_not_repeat_and_rests_after_speaking():
    th, _, _ = make((0,))
    th.run(40)
    a = acts(th)
    assert a.count("出力") == 1
    assert a[-10:] == ["待つ"] * 10                      # 言い終えたら、休む（回り続けない）
    assert "もう言った" in [o for row in th.log["odd"] for o in row]


def test_new_words_wake_it_and_stale_items_are_dropped():
    th, _, talk = make((0,))
    th.run(40)
    th.words.push(talk[2])
    th.run(40)
    assert len(th.said) >= 2
    assert sorted(th.said[1][1]) == sorted(n for n, _, _ in talk[2])


def test_assoc_only_adds_when_lack_goes_down():
    th, _, _ = make((0, 1, 2, 3))
    lacks = []

    class See(Node):
        def after(self, loop, rec):
            if rec["した事"] == "連想":
                lacks.append((before[0], rec["不足"]))
    th.attach(See())
    before = [None]
    for _ in range(120):
        before[0] = th.lack()
        th.step()
    assert lacks                                          # 連想は起きている
    # 連想の歩では器は 1 刻み進むだけなので、不足は大きくは増えない（足す前より減る物だけを足す）
    assert all(b - a > -0.05 for a, b in [(x, y) for y, x in lacks])


def test_builder_socket_is_replaceable():
    class Never:
        def build(self, tray, vessel, said):
            return [], {"だめ": "いつも違和感"}
    vocab, talk = demo_world()
    th = Thinker(Vessel(64, bands=8), Words(talk[0]), Near(list(vocab), list(vocab.values())), Never())
    th.run(30)
    assert not th.said and acts(th)[-1] == "待つ"


def test_without_near_it_still_runs():
    _, talk = demo_world()
    th = Thinker(Vessel(64, bands=8), Words(talk[0]))
    th.run(20)
    assert len(th.said) == 1 and "連想" not in acts(th)


def test_seal_and_determinism():
    a, _, _ = make((0, 1, 2, 3))
    b, _, _ = make((0, 1, 2, 3))
    a.run(150)
    b.run(150)
    assert a.log["act"] == b.log["act"] and a.said == b.said
    a.set("k", 3)
    a.seal()
    with pytest.raises(RuntimeError):
        a.set("k", 4)
    with pytest.raises(RuntimeError):
        a.attach(Node())


def test_plain_builder_oddness():
    v = Vessel(8, bands=4)
    t = Tray()
    x = np.zeros(8); x[0] = 1
    y = np.zeros(8); y[1] = 1
    t.add("x", x, "語", 0)
    _, odd = Plain().build(t, v, [])
    assert set(odd) == {"途中", "材料が少ない"}
    t.add("y", y, "語", 1)
    t.add("文の終わり", None, "印", 1)
    line, odd = Plain().build(t, v, [])
    assert not odd
    _, odd = Plain().build(t, v, [(0, line)])
    assert set(odd) == {"もう言った"}


def test_odd_wave_holds_and_fades_fast_before_slow():
    w = OddWave(Vessel(4, bands=6).a)
    for _ in range(5):
        w.feel({"x": 1.0})
    up = w.S[:, 0].copy()
    assert up[0] > up[-1] > 0                             # 速い波が先に立つ
    for _ in range(40):
        w.feel({"y": 0.5})                                # 途中で種類が増えてもよい
    assert w.S[0, 0] < w.S[-1, 0]                         # 速い波は先に消え、遅い波に残る
    H, R = w.arrays()
    assert H.shape == (45, 6, 2) and R.shape == (45, 2) and H[0, :, 1].max() == 0


def test_odd_wave_does_not_change_behaviour_and_is_tallied():
    a, _, _ = make((0, 1, 2, 3))
    a.run(150)
    kinds = a.odd.kinds
    assert "材料が不足" in kinds and "もう言った" in kinds
    t = a.tally()
    assert t["下がり方"]["材料が不足"]["連想"] < 0           # 連想は、不足が減る時だけ足す
    assert len(t["器との動き"]["材料が不足"]) == 8
    arr = a.arrays()
    assert arr["odd_wave"].shape[0] == 150 and list(arr["odd_kinds"]) == kinds


class FakeScorer:
    def line(self, keys, marks):
        self.seen = (list(keys), list(marks))
        return [dict(間=i, 詰め物="が" if i == 0 else "", 割合=dict(迷い=0.2 + 0.1 * i, 組の珍しさ=0.9 - 0.1 * i)) for i in range(len(keys))]


def test_scored_builder_fills_line_and_feeds_wave_without_changing_flow():
    vocab, talk = demo_world()
    words = [(n, v, dict(m, 過去=True) if m.get("文の終わり") else m) for n, v, m in talk[0]]
    words[-1] = (words[-1][0], words[-1][1], dict(words[-1][2], 展開=0.25))
    fs = FakeScorer()
    th = Thinker(Vessel(64, bands=8), Words(words), None, Scored(fs, {n: "NOUN" for n in vocab}))
    th.run(20)
    assert th.said[0][1] == ["A0", "が", "A1", "A2"]      # 聞いた順に並べ、詰め物が入る
    assert fs.seen[1] == ["過去"]
    plain = Thinker(Vessel(64, bands=8), Words(talk[0]))
    plain.run(20)
    assert th.log["act"] == plain.log["act"]              # 行づくりの違和感は、流れを変えない
    H, R = th.odd.arrays()
    k = th.odd.kinds
    assert np.isclose(R[:, k.index("迷い")].max(), 0.4) and np.isclose(R[:, k.index("組の珍しさ")].max(), 0.9)
    assert np.isclose(R[:, k.index("展開")].max(), 0.25)


def test_pick_keeps_top_k_in_heard_order():
    vocab, talk = demo_world()
    w = {"A0": 0.1, "A1": 0.9, "A2": 0.5}
    fs = FakeScorer()
    th = Thinker(Vessel(64, bands=8), Words(talk[0]), None, Scored(fs, {n: "NOUN" for n in vocab}, pick=Pick(w.get, 2)))
    th.run(20)
    assert [k for k, _ in fs.seen[0]] == ["A1", "A2"]
    assert Pick(lambda n: None, 2).pick([dict(名前="x"), dict(名前="y"), dict(名前="z")]) == [dict(名前="x"), dict(名前="y")]
