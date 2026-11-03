"""目標への到達度（ポイント表）。合計 100 点。

各項目: 点 = 配点 × clip((今の値 − ゼロ点の基準) / (目標 − ゼロ点の基準), 0, 1)
小さいほど良い指標も同じ式（目標 < ゼロ点の基準になる）。桁で変わるもの（時間など）は log10 で比べる。
未測定の項目は 0 点。新しい結果が出たら ITEMS の「今の値」と「出典」を書き換えて
    python tools/scoreboard.py --md
で docs/scoreboard.md を作り直す。項目と配点は案（2026-10-23）。到達点の中身が決まったら見直す。
"""
from __future__ import annotations

import argparse
import math
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")

# (軸, 項目, 配点, ゼロ点の基準, 目標, 今の値, 対数か, 目標の根拠, 出典)
ITEMS = [
    ("M1 話者と隠し味", "2DWIG の正解率（12 類、%）", 8, 38.6, 52.7, None, False, "数えた表（語尾・2 つ組の袋、SN2 実験72）。ゼロ点 = いつも一番多い類", "未測定"),
    ("M1 話者と隠し味", "2DWIG の類ごとの正解率の平均（%）", 6, 8.3, 40, None, False, "少ない類も当たる（仮の目標）。ゼロ点 = 偶然（1/12）", "未測定"),
    ("M1 話者と隠し味", "隠し味の効き（次の単位の正解率の差、点）", 8, 0, 1.0, None, False, "数えた表の見積もり +0.8（SN2、150 作品）を上回る", "未測定"),
    ("M1 話者と隠し味", "隠し味つきの 2DTG と GRU の差（正解率、点。小さいほど良い）", 6, 10, 0, None, False, "GRU（話者つき）に並ぶ", "未測定"),
    ("M1 話者と隠し味", "口調の当たり（作った文の話者の組が合う割合、%）", 6, 25, 70, None, False, "仮の目標。ゼロ点 = 組 4 つの偶然", "未測定"),
    ("M2 短い返事", "2DWYG の正解率（相手との関係、%）", 6, 0, 1, None, False, "課題を決めてから", "未測定"),
    ("M2 短い返事", "返事が文として閉じる割合（%）", 6, 0, 90, None, False, "仮の目標", "未測定"),
    ("M2 短い返事", "繰り返しのない返事の割合（%）", 4, 0, 90, None, False, "仮の目標", "未測定"),
    ("M2 短い返事", "持ち主が読んで「会話になっている」と判断した割合（%）", 6, 0, 50, None, False, "docs/samples/ を持ち主が読む", "未測定"),
    ("M3 記憶と関連性", "会話の中で前に出た語を拾える（0/1）", 8, 0, 1, None, False, "課題を決めてから", "未測定"),
    ("M3 記憶と関連性", "関連度で内容語の当たりが上がる（点）", 6, 0, 2, None, False, "数えた表の見積もり +2〜11（SN2）", "未測定"),
    ("M4 意図・流れ・知識", "意図・流れ・知識・一貫性・アイディアの最初の形（5 つのうち、0〜5）", 10, 0, 5, None, False, "どれも最初の形が動く", "未測定"),
    ("M5 最小版", "kotoyui.py 1.0.0 が会話できる最小の形（0/1）", 8, 0, 1, None, False, "tests/ が通り、返事が出る", "未測定"),
    ("効率・アナログ", "光の部分の割合（学習する数のうち、%）", 6, 2, 50, None, False, "SN2 の単語は 2%（98% がデジタルの符号）", "未測定"),
    ("効率・アナログ", "位相だけの板での保ち（%、元に対する割合）", 6, 0, 95, None, False, "振幅 + 位相との比（SN2 実験63 は −2.2 点）", "未測定"),
]


def score(w, zero, goal, now, log):
    if now is None:
        return 0.0
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    r = (f(now) - f(zero)) / (f(goal) - f(zero))
    return w * min(max(r, 0.0), 1.0)


def to_md() -> str:
    """ポイント表を Markdown にする（合計・軸ごと・項目ごと）。"""
    axes, rows = {}, []
    total = 0.0
    for ax, name, w, zero, goal, now, log, why, src in ITEMS:
        s = score(w, zero, goal, now, log)
        total += s
        a = axes.setdefault(ax, [0.0, 0])
        a[0] += s
        a[1] += w
        rows.append(f"| {ax} | {name} | {w} | {zero} | {goal} | {'未測定' if now is None else now} | {s:.1f} | {why} | {src} |")
    out = ["# ポイント表（到達度、合計 100 点）", "",
           "`python tools/scoreboard.py --md` で作り直す（手で書き換えない）。項目と配点は案（2026-10-23）。", "",
           f"**合計: {total:.1f} / {sum(i[2] for i in ITEMS)} 点**", "",
           "| 軸 | 点 | 配点 |", "|---|---|---|"]
    out += [f"| {ax} | {s:.1f} | {w} |" for ax, (s, w) in axes.items()]
    out += ["", "| 軸 | 項目 | 配点 | ゼロ点の基準 | 目標 | 今の値 | 点 | 目標の根拠 | 出典 |", "|---|---|---|---|---|---|---|---|---|"]
    out += rows
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="docs/scoreboard.md に書き出す")
    args = ap.parse_args()
    assert sum(i[2] for i in ITEMS) == 100, "配点の合計は 100 点"
    md = to_md()
    if args.md:
        with open(os.path.join(ROOT, "docs", "scoreboard.md"), "w", encoding="utf-8") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    main()
