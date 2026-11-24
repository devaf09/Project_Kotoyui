"""I49zf の関門: 板で選び直す時の関門（出た語が有る時だけ ／ 同じ波が板に有る候補だけ）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


class _Rel:
    def __init__(self, pts):
        self.pts = pts

    def ask(self, si, tri, cands):
        return {"答え": max(self.pts, key=self.pts.get), "点": dict(self.pts)}

    def learn(self, tri):
        pass

    def tick(self):
        pass


def _fp(gate, **kw):
    from i49u_flow_pick import FlowPick
    return FlowPick(_Rel({"先生": 3.0, "友達": 2.0, "海": 1.0}), V.get, pick="上位", top=3, board="並べる", gate=gate, **kw)


def test_same_wave_gate():
    fp = _fp("同じ波")
    c = ["先生", "友達", "海"]
    assert fp.ask(0, (None, "魚", "釣る"), c)["答え"] == "先生"          # 板が空なら一位のまま
    fp.learn(("魚", "ゲーム", "釣る"))
    assert fp.ask(0, (None, "魚", "釣る"), c)["答え"] == "先生"          # 候補と同じ波が板に無ければ動かさない
    fp.learn(("友達", "魚", "釣る"))
    assert fp.same_on_board("友達") and not fp.same_on_board("先生")
    assert fp.ask(0, (None, "魚", "釣る"), c)["答え"] == "友達"          # 板に同じ波が有る候補に替える
    assert fp.picked == 1


def test_same_wave_gate_by_seat():
    fp = _fp("同じ波", by_seat=True, half=1, carry=True)
    fp.learn(("魚", "友達", "釣る"))
    c = ["先生", "友達", "海"]
    assert fp.ask(0, (None, "魚", "釣る"), c)["答え"] == "先生"          # 別の席に出た語は数えない
    assert fp.ask(1, ("魚", None, "釣る"), c)["答え"] == "友達"


def test_said_gate():
    fp = _fp("出た語")
    fp.learn(("魚", "ゲーム", "釣る"))
    c = ["先生", "友達", "海"]
    assert fp.ask(0, (None, "魚", "釣る"), c)["答え"] == "先生"
    assert fp.picked == 0


def test_run_shapes():
    import i49zf_gate_run as Z
    import i49ze_row_board_run as ZE
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    n = len(ZE.VERS)
    res = Z.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    assert len(ZE.VERS) == n
    r = res["白紙から"]
    assert len(r) == 17 and r["板無し"]["答えが変わった数"] == 0
    for d in r.values():
        assert 0 <= d["正しい物"] <= d["合わせて"] <= 1


def test_same_wave_reset_per_talk_follows_recency():
    fp = _fp("同じ波", half=1)
    c = ["先生", "友達", "海"]
    fp.learn(("友達", "魚", "釣る"))
    fp.learn(("海", "魚", "釣る"))
    assert fp.ask(0, (None, "魚", "釣る"), c)["答え"] == "海"            # 新しく出た語ほど重い
    fp.tick()
    assert fp.ask(0, (None, "魚", "釣る"), c)["答え"] == "先生"          # 話が変わると板は空


def test_same_weight_matches_said_rule():
    """同じ波の重さで選ぶ形は、決まりで書いた「出た語」と同じ答えになる。"""
    from i49u_flow_pick import FlowPick
    pts = {"先生": 3.0, "友達": 2.0, "海": 1.0}
    a = _fp("同じ波・重さ", half=1)
    b = FlowPick(_Rel(pts), V.get, pick="出た語", top=3)
    c = ["先生", "友達", "海"]
    for tri in [("魚", "ゲーム", "釣る"), ("友達", "海", "釣る"), ("魚", "海", "泳ぐ"), ("先生", "友達", "釣る")]:
        assert a.ask(0, (None, "魚", "釣る"), c)["答え"] == b.ask(0, (None, "魚", "釣る"), c)["答え"]
        a.learn(tri)
        b.learn(tri)
    assert a.ask(0, (None, "魚", "釣る"), c)["答え"] == b.ask(0, (None, "魚", "釣る"), c)["答え"] == "先生"
