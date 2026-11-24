"""部品と道具の一覧（docs/parts_list.md）が、今のファイルと合っているか。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))


def test_parts_list_is_fresh():
    import make_parts_list as M
    have = open(os.path.join(HERE, "..", "docs", "parts_list.md"), encoding="utf-8").read()
    assert have == M.build(), "python tools/make_parts_list.py で作り直す"
