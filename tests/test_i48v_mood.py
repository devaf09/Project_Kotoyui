"""I48v（話し相手の気分の設定と書き出し）の小さな確かめ（LLM を使わない）。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48v_mood as V  # noqa: E402


def test_order_balanced_and_seeded():
    o = V.order(10, 0)
    assert sorted(o) == sorted(list(V.MOODS) * 2) and V.order(10, 0) == o and V.order(10, 1) != o


def test_parse_and_rate():
    ok = json.dumps({s: 0.5 for s in V.SIDES} | {"喜び": 1.4, "悲しみ": -1})
    r = V.parse("x" + ok)
    assert r["喜び"] == 1.0 and r["悲しみ"] == 0.0 and len(r) == 8        # 0〜1 に収める。両側を別々に持つ
    assert V.parse(json.dumps({"喜び": 0.1})) is None                       # 欠けた欄がある
    seen = []
    chat = lambda host, model, q, **k: (seen.append(q), ok)[1]
    msgs = [dict(role="system", content="s")]
    assert V.rate(chat, "h", "m", msgs)["信頼"] == 0.5 and len(msgs) == 1 and seen[0][-1]["content"] == V.RATE   # 会話の msgs には足さない
