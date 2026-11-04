"""実験I1 の図: 介入の問い・見たままの問いの正解率を、答え方ごとに並べる（intervention/images/i01_qa.png）

intervention/results/i01.jsonl（比べる相手は i01_baselines.jsonl）の、既定の設定（しきい値 0.5・分け直し・見直し・読み直しあり、光）の行から。
棒 = seed の平均、ひげ = 最小〜最大。灰色の点線 = いつも「いいえ」と答えたときの正解率。
実行: python intervention/tools_fig_i01.py
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
KINDS = ["塗る", "熟す", "見たまま"]
SERIES = [   # (名前, 値の取り出し, 色)
    ("語板 + 否定（1 語抜き）", lambda r: r["qa"]["deny_loo"], "#2a78d6"),
    ("語板 + 相関だけ", lambda r: r["qa"]["corr_loo"], "#eb6834"),
    ("GRU（介入の例あり）", lambda r: r["baselines"]["gru_int"], "#1baf7a"),
    ("トランスフォーマー（介入の例あり）", lambda r: r["baselines"]["tf_int"], "#eda100"),
    ("トランスフォーマー（見たままの例だけ）", lambda r: r["baselines"]["tf_obs"], "#e87ba4"),
]


def main():
    R = [json.loads(l) for l in open(os.path.join(HERE, "results", "i01.jsonl"), encoding="utf-8") if l.strip()]
    BL = {(b["seed"], b["n_train_per_sense"]): b["baselines"]
          for b in (json.loads(l) for l in open(os.path.join(HERE, "results", "i01_baselines.jsonl"), encoding="utf-8") if l.strip())}
    base = [dict(r, baselines=BL[(r["seed"], r["n_train_per_sense"])]) for r in R
            if r["mode"] == "light" and r["theta"] == 0.5 and r["split"] and r["consolidate"] and r["reread"]
            and (r["seed"], r["n_train_per_sense"]) in BL]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, n in zip(axes, (50, 10)):
        rs = [r for r in base if r["n_train_per_sense"] == n]
        w = 0.8 / len(SERIES)
        for i, (name, get, col) in enumerate(SERIES):
            m, lo, hi = [], [], []
            for k in KINDS:
                v = [get(r)[k] for r in rs]
                m.append(np.mean(v)); lo.append(max(0.0, np.mean(v) - np.min(v))); hi.append(max(0.0, np.max(v) - np.mean(v)))
            x = np.arange(len(KINDS)) + (i - (len(SERIES) - 1) / 2) * w
            ax.bar(x, m, w * 0.9, color=col, label=name, yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8, ecolor="#52514e"))
        no = rs[0]["always_no"]
        for j, k in enumerate(KINDS):
            ax.hlines(no[k], j - 0.42, j + 0.42, colors="#8a8984", linestyles="dashed", lw=1)
        ax.set_xticks(range(len(KINDS)))
        ax.set_xticklabels([f"{k}\n（{rs[0]['n_q'][k]} 問）" for k in KINDS])
        ax.set_title(f"教えの文: 意味 1 つあたり {n} 文（{len(rs)} seed）", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("正解率")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#8a8984", ls="dashed")); l.append("いつも「いいえ」")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("実験I1: 介入の問い（塗ったら・熟させたら）と見たままの問い", fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    out = os.path.join(HERE, "images", "i01_qa.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
