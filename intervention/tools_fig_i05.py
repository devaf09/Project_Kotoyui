"""実験I5 の図: 反例（リンゴ）の文を固定し、交絡した果物の数 K（横）を増やしたときの、介入で得た知見（縦）（非公開の置き場の images/i05_more_words.png）

点 = 6 seed の平均、細い点 = seed ごと。点線 = 1 語あたり 4.5 問の比例（50 文の平均の傾き）。
実行: python intervention/tools_fig_i05.py
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
PRIV = os.environ.get("KOTOYUI_PRIVATE_INTERVENTION", os.path.join(HERE, "..", "..", "Project_Kotoyui_private", "intervention"))   # 記録・図・結果は非公開の置き場（持ち主 2026-11-05）
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
COL = {50: "#2a78d6", 10: "#eb6834"}


def main():
    R = [json.loads(l) for l in open(os.path.join(PRIV, "results", "i05.jsonl"), encoding="utf-8") if l.strip()]
    G = defaultdict(list)
    for r in R:
        G[(r["n_train_per_sense"], r["K"])].append(r)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    Ks = sorted({k for _, k in G})
    for n in (50, 10):
        m = [np.mean([r["gain"] for r in G[(n, k)]]) for k in Ks]
        for k in Ks:
            ax.scatter([k] * len(G[(n, k)]), [r["gain"] for r in G[(n, k)]], s=10, color=COL[n], alpha=0.35)
        ax.plot(Ks, m, "-o", color=COL[n], lw=2, ms=5, label=f"意味 1 つあたり {n} 文（反例のリンゴも {n} 文）")
    ax.plot(Ks, [4.5 * k for k in Ks], ":", color="#8a8984", lw=1.2, label="1 語あたり 4.5 問の比例")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("熟すと色と味が一緒に変わる果物の数 K（反例の文は固定）")
    ax.set_ylabel("介入で得た知見（問いの数）")
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("実験I5: 同じ反例から得る知見は、語の数に比例して増える（6 seed）", fontsize=11)
    fig.tight_layout()
    out = os.path.join(PRIV, "images", "i05_more_words.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
