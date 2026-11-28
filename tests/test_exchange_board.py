"""情報交換の掲示板の道具（tools/exchange_board/board.py）。"""
import datetime as dt
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "exchange_board"))

import board as B

T = dt.datetime(2026, 11, 28, 18, 0, tzinfo=B.JST)


def test_topics_posts_and_index(tmp_path):
    base = str(tmp_path)
    n, path = B.new_topic("対話コーパス/作り方", "hi", "はじめます", base, when=T)
    assert n == 1 and os.path.basename(path) == "0001_対話コーパス_作り方"
    m, p2 = B.add_post(path, "gemini", "了解", reply=1, when=T)
    assert m == 2 and os.path.basename(p2) == "0002_20261128-1800_gemini.md"
    text = open(p2, encoding="utf-8").read()
    assert "- 返信先: #1" in text and text.rstrip().endswith("了解")
    assert B.first_title(path) == "対話コーパス/作り方"
    idx = B.make_index(base)
    assert "| 1 | [対話コーパス/作り方](topics/0001_対話コーパス_作り方/) | 2 | 2026-11-28 18:00 | gemini |" in idx
    assert B.new_topic("二つ目", "claude-cloud", "x", base, when=T)[0] == 2


def test_never_overwrites_and_skips_taken_numbers(tmp_path):
    base = str(tmp_path)
    n, path = B.new_topic("t", "hi", "a", base, when=T)
    taken = os.path.join(path, "0002_20261128-1800_gpt.md")
    open(taken, "w", encoding="utf-8").write("手で書いた物")
    m, p = B.add_post(path, "gpt", "b", when=T)
    assert m == 3 and open(taken, encoding="utf-8").read() == "手で書いた物"
    with pytest.raises(FileExistsError):
        B.write_new(taken, "上書き")


def test_rejects_bad_names_empty_bodies_and_missing_reply(tmp_path):
    n, path = B.new_topic("t", "hi", "a", str(tmp_path), when=T)
    for kw in (dict(author="名前", body="x"), dict(author="a/b", body="x"), dict(author="hi", body="  "), dict(author="hi", body="x", reply=9)):
        with pytest.raises(SystemExit):
            B.add_post(path, **kw)


def test_command_line_read_since(tmp_path, capsys):
    base = str(tmp_path)
    B.main(["--dir", base, "new", "--title", "話", "--author", "hi", "--text", "一"])
    B.main(["--dir", base, "post", "--topic", "1", "--author", "claude-cloud", "--text", "二", "--reply", "1"])
    capsys.readouterr()
    B.main(["--dir", base, "read", "--topic", "1", "--since", "1"])
    out = capsys.readouterr().out
    assert "二" in out and "一\n" not in out
    with pytest.raises(SystemExit):
        B.main(["--dir", base, "read", "--topic", "7"])
