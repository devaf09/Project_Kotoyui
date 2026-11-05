"""実験I10 の図: 知らない品種の属性を、どこから想定するかで、本当の値に付けた確率がどう違うか（intervention/images/i10_hierarchy.png）

棒 = 6 seed × 隠した品種 20 の平均（意味 1 つあたり 10 文）。右: 言い方（きっと / たぶん / わからない）ごとの正解率（較正）。
実行: python intervention/tools_fig_i10.py
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
M = [("親", "#2a78d6"), ("親 + 祖先で薄める", "#1baf7a"), ("祖先（果物全体）", "#eda100"), ("誤った親", "#eb6834"), ("一様", "#8a8984")]


def main():
    R = [json.loads(l) for l in open(os.path.join(HERE, "results", "i10.jsonl"), encoding="utf-8") if l.strip()]
    rs = [r for r in R if r["n"] == 10]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw=dict(width_ratios=[1.4, 1]))
    V = ["色", "味", "形"]
    w = 0.8 / len(M)
    for i, (m, c) in enumerate(M):
        ax.bar(np.arange(3) + (i - 2) * w, [np.mean([r["summary"][m]["p_true"][v] for r in rs]) for v in V], w * 0.9, color=c, label=m)
    ax.set_xticks(range(3))
    ax.set_xticklabels(V)
    ax.set_ylabel("本当の値に付けた確率（平均）")
    ax.set_title("知らない品種の属性を、どこから想定するか", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    for i, (m, c) in enumerate(M[:4]):
        cal = defaultdict(lambda: [0.0, 0])
        for r in rs:
            for k, (acc, n) in r["summary"][m]["calib"].items():
                cal[k][0] += acc * n
                cal[k][1] += n
        ks = ["わからない", "たぶん", "きっと"]
        ax2.plot(range(3), [cal[k][0] / cal[k][1] if cal[k][1] else np.nan for k in ks], "-o", color=c, label=m, lw=2, ms=5)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["わからない\n（< 0.4）", "たぶん\n（0.4〜0.7）", "きっと\n（≥ 0.7）"])
    ax2.set_ylabel("その言い方で答えたときの正解率")
    ax2.set_ylim(0, 1)
    ax2.set_title("言い方と正解率（較正）", fontsize=11)
    for a_ in (ax, ax2):
        a_.grid(axis="y", color="#e6e5e0", lw=0.6)
        a_.set_axisbelow(True)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.suptitle("実験I10: 親（科）から知らない子（品種）を想定する（6 seed、品種 1 つあたり 10 文）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(HERE, "images", "i10_hierarchy.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
