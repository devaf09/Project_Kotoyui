"""I49zzzc: 部品を通しでつないで文を出す（小さな作り物の話。本体は代わりの物）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools", os.path.join("intervention", "past")):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzy_form as Y
import i49zzzc_loop_run as C
from test_i49zzw import N, V, _attr, _talks
from test_i49zzx import PairBody
from test_i49zzy import SY


def _former():
    its = []
    for lem, stem in (("買う", "買い"), ("食べる", "食べ"), ("借りる", "借り"), ("読む", "読み")):
        its += [(SY, lem, ("ます",), stem, ("ます",))] * 3 + [(SY, lem, (), lem, ())] * 3
    return Y.Former([its])


def test_line_of_builds_plain_and_polite():
    f = _former()
    tri = (N("太郎"), N("りんご"), V("食べる"))
    assert C.line_of(None, tri) == "太郎がりんごを食べる。"
    assert C.lines_of(f, tri) == {"言い切り": "太郎がりんごを食べる。", "丁寧": "太郎がりんごを食べます。"}


def test_loop_outputs_replies_free_runs_and_counts():
    talks = _talks()
    d = C.run(talks[:12], talks[12:], _attr, PairBody(), _former(), talks=2, steps=4, sample=6, dims=512)
    assert len(d["相手に返す"]) == 2 * 3                                   # 話 2 個 × 続いた 2 行の組 3 つ
    r = d["相手に返す"][0]
    assert r["返事_広い候補"]["丁寧"].endswith("ます。") and r["前の行"].endswith("。")
    assert len(d["自分で続ける"]) == 2 * 2 and all(len(x["続き"]) == 4 for x in d["自分で続ける"])
    new = [x for x in d["自分で続ける"] if x["組み方"] == C.FORM_NEW]
    assert all(len({s["言い切り"].split("を")[1] for s in x["続き"]}) >= 3 for x in new)   # 出た動きは返さない形は、動きが替わり続ける
    verbs = ("買う", "食べる", "借りる", "読む", "買います", "食べます", "借ります", "読みます")
    assert all(s["言い切り"].rstrip("。").endswith(verbs) for x in d["自分で続ける"] for s in x["続き"])   # 動きの席に、物の語を置かない
    c = d["数え_広い候補"][C.FORM]
    assert c["問い"] >= 6 and c["だれが"] == c["問い"]                      # この世界では、人は話の中に 1 人だけ
