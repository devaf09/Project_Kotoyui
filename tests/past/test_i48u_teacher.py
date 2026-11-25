"""I48u（促しの先生の型）の出力の読み方の確かめ（LLM を使わない）。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
import i48u_teacher as U  # noqa: E402


def j(**k):
    return "考え中…" + json.dumps(k, ensure_ascii=False)


def test_parse_verdicts():
    ev, bad = U.parse(j(verdict="そのまま", kind="言い方", odd_part="海", message=""), "海に行く。")
    assert bad == "" and ev["verdict"] == "そのまま" and not ev["has_error"] and ev["kind"] == ""     # そのまま は種類を持たない
    ev, _ = U.parse(j(verdict="良", message="受けていていいね"), "海に行く。")
    assert ev["verdict"] == "良" and not ev["has_error"]
    ev, bad = U.parse(j(verdict="直す", kind="文法", odd_part="海を行く", suggested_reply="海に行く。", message="〜と言うと伝わるよ"), "海を行く。")
    assert bad == "" and ev["has_error"] and ev["odd_part"] == "海を行く" and ev["kind"] == "文法"


def test_parse_rejects():
    assert U.parse("読めない", "x")[1] == "読めない"
    assert U.parse(j(verdict="?"), "x")[1] == "判定が無い"
    assert U.parse(j(verdict="直す", odd_part="山", suggested_reply="海。"), "海。")[1] == "返事に無い所"
    assert U.parse(j(verdict="直す", odd_part="海", suggested_reply="海。"), "海。")[1] == "お手本が同じ"
    assert set(U.PROMPTS) == {"p2a", "p2b", "p2c"} and all("JSON" in p for p in U.PROMPTS.values())
