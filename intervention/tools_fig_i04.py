"""実験I4 の図: 教えの文の数（横、対数）に対する、正しく答えた問いの数（縦）（intervention/images/i04_scale.png）

左: 正しく答えた問いの数（塗る + 熟す + 見たまま）。実線 = I3 の最終形、点線 = 相関だけ。色 = 世界（small / large）。
右: 相関だけでは間違え、介入（否定）で正しく答えた問いの数（= 介入で得た知見）。
点 = 6 seed の平均、帯 = 最小〜最大。
実行: python intervention/tools_fig_i04.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
COL = {"small": "#2a78d6", "large": "#eb6834"}
NAME = {"small": "小さい世界（5 語・9 意味・問い 108）", "large": "大きい世界（10 語・18 意味・問い 312）"}


def main():
    R = [json.loads(l) for l in open(os.path.join(HERE, "results", "i04.jsonl"), encoding="utf-8") if l.strip()]
    G = defaultdict(list)
    for r in R:
        G[(r["world"], r["n_train_per_sense"])].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for w in ("small", "large"):
        ns = sorted(n for ww, n in G if ww == w)
        x = [n * G[(w, n)][0]["n_senses"] for n in ns]
        for key, ls, lab in (("deny", "-", "介入（否定）"), ("corr", ":", "相関だけ")):
            v = [[sum(r["correct_pairs"][key]) for r in G[(w, n)]] for n in ns]
            m = [np.mean(a) for a in v]
            axes[0].plot(x, m, ls, color=COL[w], marker="o", ms=4, lw=2, label=f"{NAME[w]}: {lab}")
            axes[0].fill_between(x, [min(a) for a in v], [max(a) for a in v], color=COL[w], alpha=0.12 if key == "deny" else 0.05)
        gain = [[sum(1 for d, c in zip(r["correct_pairs"]["deny"], r["correct_pairs"]["corr"]) if d and not c) for r in G[(w, n)]] for n in ns]
        axes[1].plot(x, [np.mean(a) for a in gain], "-", color=COL[w], marker="o", ms=4, lw=2, label=NAME[w])
        axes[1].fill_between(x, [min(a) for a in gain], [max(a) for a in gain], color=COL[w], alpha=0.12)
        for xi, n in zip(x, ns):
            axes[1].annotate(f"{n}", (xi, np.mean([sum(1 for d, c in zip(r['correct_pairs']['deny'], r['correct_pairs']['corr']) if d and not c) for r in G[(w, n)]])),
                             textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color="#52514e")
    axes[0].set_title("正しく答えた問いの数", fontsize=11)
    axes[1].set_title("相関だけでは間違え、介入で正しく答えた問いの数\n（点の上の数字 = 意味 1 つあたりの文）", fontsize=11)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("教えの文の総数（対数）")
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("問いの数")
    fig.suptitle("実験I4: 文の数と世界の大きさに対する、答えられる問い（6 seed）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    out = os.path.join(HERE, "images", "i04_scale.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
