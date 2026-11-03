"""実験03・04 の図: 4f 系 ＋ 平方根の光路で次の言葉を当てる（実験03）と、デジタルの符号を減らす 3 つの方法（実験04）（docs/images/exp03_04.png）

results/exp02.txt・exp03.txt・exp04.txt から作る（どれも同じテスト 3 万位置、seed 0）。
実行: python tools/fig_exp03_04.py
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
LIGHT, BASE = "#1f6f8b", "#999"


def rows(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, "results", name), encoding="utf-8") if l.strip()]


def main():
    e2, e3, e4 = rows("exp02.txt"), rows("exp03.txt"), rows("exp04.txt")
    g = lambda m, st=3000, sd=0: [r for r in e2 if r["method"] == m and r["steps"] == st and r["seed"] == sd][-1]
    n3 = [r for r in e3 if not r["tag"]][-1]
    t3 = [r for r in e3 if r["tag"]][-1]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    names = ["板の分担\n（実験02c）", "4f 系\n光路 1 本", "4f 系\n分担 4", "4f 系\n1 本 + 分担", "GRU", "3-gram", "4f 系 2 つ\n+ 3-gram"]
    vals = [g("split", 9000)["test_acc"], n3["f4_single"]["acc"], n3["f4_split"]["acc"], n3["fuse_light"]["acc"],
            g("gru")["test_acc"], n3["tri"]["acc"], n3["fuse_tri"]["acc"]]
    cols = ["#9bd", LIGHT, LIGHT, LIGHT, BASE, "#c8a27a", "#7a8f6a"]
    x = np.arange(len(names))
    ax[0].bar(x, vals, color=cols)
    for xi, v in zip(x, vals):
        ax[0].text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    ax[0].set_xticks(x, names, fontsize=8.5)
    ax[0].set_ylim(15, 33)
    ax[0].set_ylabel("次のサブワードの正解率（%）")
    ax[0].set_title("(a) 実験03: 4f 系 + 平方根で次の言葉（seed 0）", fontsize=10)

    learned = n3["f4_single"]
    pts = [("学習する符号\n（実験03）", learned["acc"], learned["params_digital"], LIGHT)]
    lab = {"two_clust": "方法 1: 2 段\n（似た単位で組）", "two_freq": "方法 1: 2 段\n（よく出る順で組）", "hartley": "方法 3: フーリエ\n（ハートレー）",
           "pixel": "方法 3: 画素", "fixed": "方法 2: 符号を固定", "proto": "方法 2: 典型に\nそろえる"}
    colr = {"two_clust": "#2b7a3d", "two_freq": "#2b7a3d", "hartley": "#d9822b", "pixel": "#d9822b", "fixed": "#8b5cf6", "proto": "#8b5cf6"}
    for r in e4:
        pts.append((lab[r["method"]], r["test_acc"], max(r["params_digital_learned"], 1), colr[r["method"]]))
    for name, acc, dig, c in pts:
        ax[1].scatter(dig, acc, s=70, color=c, zorder=3)
        off = (6, -18) if "よく出る" in name else ((6, 6) if "似た" in name else (6, -4))
        ax[1].annotate(name, (dig, acc), textcoords="offset points", xytext=off, fontsize=7.5)
    ax[1].set_xscale("log")
    ax[1].set_xlim(0.5, 1e8)
    ax[1].set_ylim(8, 27)
    ax[1].set_xlabel("学習するデジタルの数（対数、光路の数で合計）")
    ax[1].set_ylabel("次のサブワードの正解率（%）")
    ax[1].set_title("(b) 実験04: デジタルの符号を減らす（右ほどデジタルが多い）", fontsize=10)
    ax[1].grid(alpha=0.3)

    sty = [("板の分担\n+スイッチ（02c）", (lambda r: (r["style_per"]["丁寧"] + r["style_per"]["くだけた"]) / 2)(g("split_tag", 9000)), "#9bd"),
           ("4f 系 1 本\n+スイッチ", t3["style_f4_single"]["avg"], LIGHT),
           ("4f 系 2 つ\n+スイッチ", t3["style_fuse_light"]["avg"], LIGHT),
           ("GRU\n+スイッチ", g("gru_tag")["style_acc"], BASE),
           ("3-gram\n+口調の表", [r for r in e2 if r["method"] == "tri_tag"][-1]["style_acc"], "#c8a27a")]
    x = np.arange(len(sty))
    ax[2].bar(x, [s[1] for s in sty], color=[s[2] for s in sty])
    for xi, s in zip(x, sty):
        ax[2].text(xi, s[1] + 1, f"{s[1]:.0f}", ha="center", fontsize=9)
    ax[2].axhline(50, ls="--", c="k", lw=1)
    ax[2].text(len(sty) - 0.5, 51, "でたらめ 50%", ha="right", fontsize=8)
    ax[2].set_xticks(x, [s[0] for s in sty], fontsize=8.5)
    ax[2].set_ylim(0, 105)
    ax[2].set_ylabel("決めた口調の文になった割合（%）")
    ax[2].set_title("(c) 口調のスイッチ（丁寧・くだけた、100 文ずつ）", fontsize=10)
    fig.suptitle("実験03・04: 4f 系の光路で次の言葉を当てる・デジタルの符号を減らす（現代の雑談、同じテスト、seed 0）", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "images", "exp03_04.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
