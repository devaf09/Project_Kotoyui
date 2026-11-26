"""I49zzz: 次の 3 つ組の語は、話の中でもう出た語か、初めての語か（席ごとに数えるだけ。持ち主の案 3-2 の前提の確かめ。未確定の試し）。

持ち主の案 3-2: 内面の器は、候補の語と直接比べるのではなく、「話題を広げるか、今の話題を深掘りするか」を決める所に使う。
その前に、材料の側を見る: 次の行の語は、どこから来るか。
  続く:                   前の行の同じ席の語と同じ。
  1 つ前の別の語へ戻る:   今の語が続き始める前に、その席に居た語（A, B, B の次の A）。
  もっと前に出た語へ戻る: その話の、その席に、もっと前に出た語。
  初めての語:             その話の、その席に、初めて出る語。
「続いた数」ごとにも数える（今の語が、その席で何行続いているか。4 以上はまとめる）。
材料は「A が B を C する」の文が並んだテキスト（tools/i49zp_text_talks_run.py と同じ切り方）。学びは無い。話を全部使う。語は出さず、数だけ出す。
手で置いた物: 分け方の 4 つ、続いた数を 4 でまとめる事。乱数は使わない。
引数: --text 材料（カンマ区切り） --out 記録.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49zp_text_talks_run as ZP  # noqa: E402

KINDS = ("続く", "1 つ前の別の語へ戻る", "もっと前に出た語へ戻る", "初めての語")
RUN_MAX = 4


def kind_of(hist, nxt):
    """hist: その席の、ここまでの語の並び（1 個以上）。→ (分け方, 続いた数)"""
    cur = hist[-1]
    j = len(hist) - 1
    while j >= 0 and hist[j] == cur:
        j -= 1
    run = min(len(hist) - 1 - j, RUN_MAX)
    if nxt == cur:
        return KINDS[0], run
    if j >= 0 and nxt == hist[j]:
        return KINDS[1], run
    return (KINDS[2] if nxt in hist else KINDS[3]), run


def count(talks):
    """talks: [3 つ組の並び]。→ 席 → dict(全部={分け方: 数}, 続いた数={数: {分け方: 数}})"""
    out = {s: dict(全部={k: 0 for k in KINDS}, 続いた数={r: {k: 0 for k in KINDS} for r in range(1, RUN_MAX + 1)}) for s in ZP.SEATS}
    for tris in talks:
        for si, seat in enumerate(ZP.SEATS):
            for i in range(1, len(tris)):
                k, run = kind_of([t[si] for t in tris[:i]], tris[i][si])
                out[seat]["全部"][k] += 1
                out[seat]["続いた数"][run][k] += 1
    return out


def read(path):
    for enc in ("utf-8", "cp932"):
        try:
            return [l.strip() for l in open(path, encoding=enc).read().splitlines()]
        except UnicodeDecodeError:
            continue
    raise ValueError("読めない文字の並び")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    talks = []
    for f in a.text.split(","):
        talks += [t[0] for t in ZP.talks_of(read(f))[0]]
    res = dict(話=len(talks), **count(talks))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for seat in ZP.SEATS:
        n = sum(res[seat]["全部"].values())
        print(seat, n, {k: round(v / max(n, 1), 3) for k, v in res[seat]["全部"].items()})


if __name__ == "__main__":
    main()
