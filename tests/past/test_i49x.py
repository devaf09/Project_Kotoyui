"""I49x の関門: 見せた 2 語の点を足すか、掛けるか。"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


class Two:
    """見せた語ごとに、決め打ちの点を返す元の形。"""
    def __init__(self, by):
        self.by = by

    def ask(self, si, tri, cands):
        shown = [w for w in tri if w is not None]
        if len(shown) == 2:
            return dict(点={"勉強": 1.0}, 答え="勉強")
        return dict(点=dict(self.by.get(shown[0], {})), 答え=None)

    def learn(self, tri):
        pass

    def tick(self):
        pass


C = ["勉強", "釣る", "泳ぐ"]


def test_combining_rules():
    import i49x_and_run as X
    r = Two({"海": {"勉強": 0.9, "釣る": 0.5, "泳ぐ": 0.1}, "魚": {"勉強": 0.0, "釣る": 0.6, "泳ぐ": 0.7}})
    a = X.answers(r, 2, ("海", "魚", None), C)
    assert a["元のまま"] == "勉強" and a["別々に出して足す"] == "釣る" and a["掛ける"] == "釣る" and a["小さい方"] == "釣る"
    r = Two({"海": {"勉強": 0.9}, "魚": {"泳ぐ": 0.7}})
    a = X.answers(r, 2, ("海", "魚", None), C)
    assert a["掛ける"] is None and a["掛ける・無ければ足す"] == "勉強"
    r = Two({"海": {"釣る": 0.9}})
    a = X.answers(r, 2, ("海", "魚", None), C)
    assert a["掛ける"] == "釣る" and a["小さい方"] == "釣る"


def test_run_shapes():
    import i49x_and_run as X
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = X.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["白紙から"]["解く・属性（席あり）"]
    assert set(r) == set(X.VERS)
    for d in r.values():
        assert 0 <= d["正しい物"] <= d["合わせて"] <= 1


def test_i49_tools_have_no_dotted_keyword_names():
    """「・」を含む名前を引数の名前に書くと、Python 3.13 より前で読めない。"""
    for d in ("tools", "intervention"):
        for f in sorted(os.listdir(os.path.join(HERE, "..", d))):
            if f.startswith("i49") and f.endswith(".py"):
                tree = ast.parse(open(os.path.join(HERE, "..", d, f), encoding="utf-8").read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.keyword) and node.arg:
                        assert "・" not in node.arg, (f, node.arg)
                    if isinstance(node, ast.Name):
                        assert "・" not in node.id, (f, node.id)
