"""結果の集計: results/expNN.txt（1 回 = 1 行の JSON）を、seed 以外の引数が同じ行でまとめ、平均 ± 標準誤差の表にする。

実行: python tools/summarize.py results/exp01.txt --keys method enc --metrics test_acc test_ce
--keys を省くと、行ごとに違う引数（seed・out・device を除く）を全部使ってまとめる。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

SKIP = {"seed", "out", "device"}


def mean_se(v):
    """平均と標準誤差（1 本なら標準誤差は None）。"""
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, None
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, s / math.sqrt(len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--metrics", nargs="*", default=["test_acc", "test_ce"])
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
    if not rows:
        return
    keys = args.keys
    if keys is None:
        keys = sorted(k for k in rows[0] if k not in SKIP and k not in args.metrics
                      and not isinstance(rows[0][k], (list, dict)) and len({str(r.get(k)) for r in rows}) > 1)
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(str(r.get(k)) for k in keys)].append(r)
    print("| " + " | ".join(keys + args.metrics + ["seed の数"]) + " |")
    print("|" + "---|" * (len(keys) + len(args.metrics) + 1))
    for g, rs in groups.items():
        cells = []
        for m in args.metrics:
            v = [r[m] for r in rs if r.get(m) is not None]
            if not v:
                cells.append("")
                continue
            mu, se = mean_se(v)
            cells.append(f"{mu:.2f}" if se is None else f"{mu:.2f} ± {se:.2f}")
        print("| " + " | ".join(list(g) + cells + [str(len(rs))]) + " |")


if __name__ == "__main__":
    main()
