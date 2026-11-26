"""記録の文書が、途中で切れていない事（節の見出しが 1 から順に全部有る）。書き足す時に上書きして消した事が有ったので、確かめを置く。"""
import os
import re

DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "i49_inner_vessel.md")


def test_inner_vessel_doc_keeps_every_section():
    nums = [int(m.group(1)) for m in re.finditer(r"^## (\d+)\. ", open(DOC, encoding="utf-8").read(), flags=re.M)]
    assert len(nums) >= 78
    assert sorted(set(nums)) == list(range(min(nums), max(nums) + 1))      # 抜けた節が無い
