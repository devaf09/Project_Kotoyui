"""実験05 の図: 4f 系の分担の 2DTG を「学習する符号」と「2 段で読む」で比べる（3 seed、docs/images/exp05_split_two.png）

results/exp02.txt・exp03.txt・exp05.txt から作る（同じテスト 3 万位置）。棒は平均、線は最小〜最大、右に学習するデジタルの数。
実行: python tools/fig_exp05.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def rows(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, "results", name), encoding="utf-8") if l.strip()]


def main():
    e2, e3, e5 = rows("exp02.txt"), [r for r in rows("exp03.txt") if not r["tag"]], rows("exp05.txt")
    items = [
        ("板の分担（実験02c）", [r["test_acc"] for r in e2 if r["method"] == "split" and r["steps"] == 9000], 2097696, "#9bd"),
        ("4f 系 光路 1 本\n（学習する符号）", [r["f4_single"]["acc"] for r in e3], e3[0]["f4_single"]["params_digital"], "#5aa0b5"),
        ("4f 系 分担 4\n（学習する符号）", [r["f4_split"]["acc"] for r in e3], e3[0]["f4_split"]["params_digital"], "#1f6f8b"),
        ("4f 系 分担 4\n（2 段で読む）", [r["test_acc"] for r in e5], e5[0]["params_digital_learned"], "#2b7a3d"),
        ("GRU", [r["test_acc"] for r in e2 if r["method"] == "gru"], 1873408, "#999"),
        ("3-gram", [r["tri"]["acc"] for r in e3][:1], None, "#c8a27a"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    y = np.arange(len(items))[::-1]
    for yi, (name, v, dig, c) in zip(y, items):
        m = float(np.mean(v))
        ax.barh(yi, m, color=c, xerr=[[m - min(v)], [max(v) - m]] if len(v) > 1 else None, capsize=4)
        tail = "" if dig is None else f"　学習するデジタル {dig:,}"
        ax.text(m + 0.25, yi, f"{m:.2f}%（{len(v)} 回）{tail}", va="center", fontsize=9)
    ax.set_yticks(y, [i[0] for i in items], fontsize=9)
    ax.set_xlim(20, 40)
    ax.set_xlabel("次のサブワードの正解率（%）")
    ax.set_title("実験05: 4f 系の分担の 2DTG — 学習する符号（デジタル 419 万）と、2 段で読む形（デジタル 520）", fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "images", "exp05_split_two.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
