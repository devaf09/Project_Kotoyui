"""部品と道具の全部の一覧を作る（持ち主の頼み「過去の実験の概要も一覧があるといいかも」から）。

intervention/・experiments/・tools/ の .py を全部読み、説明の 1 行目を並べて docs/parts_list.md に書く。
手では書き換えない。部品や道具を足したら、これを回して作り直す（tests/test_parts_list.py が、古くなっていないかを確かめる）。
使い方: python tools/make_parts_list.py [出力.md]
"""
from __future__ import annotations

import ast
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DIRS = (("intervention", "介入の研究の部品（intervention/）"), ("experiments", "初めの頃の実験（experiments/）"), ("tools", "数える道具と図の道具（tools/）"))
GROUPS = (("i0", "I01〜I09: 小さい世界・逆引き・経験"), ("i1", "I10〜I19: 階層・融合・算数"), ("i2", "I20〜I29: 判断の板・文の板・違和感"),
          ("i3", "I32〜I38: 機能語の層・述語・会話の板"), ("i40", "I40〜I46: 記憶板・融合の流れ"), ("i42", "I40〜I46: 記憶板・融合の流れ"),
          ("memory", "I40〜I46: 記憶板・融合の流れ"), ("i47", "I47: 意味の波と共鳴"), ("i48", "I48: 波の板・語板・介入の板"),
          ("i49", "I49: 内面の器・関係の表・波で解く形"), ("fig_i49", "I49: 内面の器・関係の表・波で解く形"), ("fig_", "図の道具"))


def first_line(path):
    try:
        doc = ast.get_docstring(ast.parse(open(path, encoding="utf-8").read()))
    except SyntaxError:
        doc = None
    line = (doc or "（説明なし）").strip().splitlines()[0].strip()
    return re.sub(r"Sin-CNN\d*", "前のプロジェクト", line).replace("|", "／")


def group_of(name):
    for key, title in GROUPS:
        if name.startswith(key):
            return title
    m = re.match(r"(exp|est)", name)
    return "見積もりと実験 01〜" if m else "そのほか"


def build():
    out = ["# 部品と道具の全部の一覧", "",
           "`python tools/make_parts_list.py` で作り直す（手で書き換えない）。説明は、各ファイルの頭の説明の 1 行目（別のプロジェクトの名前だけ「前のプロジェクト」に置き換えている）。",
           "読み方の案内と、今使えそうな部品・外れた試しのまとめは [experiments_index.md](experiments_index.md)。", ""]
    total = 0
    for d, title in DIRS:
        files = sorted(f for f in os.listdir(os.path.join(ROOT, d)) if f.endswith(".py"))
        out += ["## " + title, ""]
        groups = {}
        for f in files:
            groups.setdefault(group_of(f), []).append(f)
        for g in sorted(groups, key=lambda g: min(groups[g])):
            out += ["### " + g, "", "| ファイル | 説明 |", "|---|---|"]
            for f in groups[g]:
                out.append("| `%s` | %s |" % (f, first_line(os.path.join(ROOT, d, f))))
                total += 1
            out.append("")
    out.insert(4, "全部で %d 個。" % total)
    out.insert(5, "")
    return "\n".join(out).rstrip() + "\n"


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "parts_list.md")
    open(path, "w", encoding="utf-8", newline="\n").write(build())
