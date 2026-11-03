"""実験07〜09 の図: 「次の発言の共鳴」を光で作る形の比べ（10 択の返事選び、docs/images/exp08_09_resonance.png）

results/exp07.txt・exp08.txt・exp09.txt から作る（同じ seed なら同じ問 3,000）。
左: 単独の正解率（棒は平均、線は最小〜最大、右に学習する数）。右: kotoyui 0.1.0 の芯（重なり + PMI）に光を足したときの増え方（seed ごと）。
実行: python tools/fig_exp08_09.py
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


def acc(r, how, hard=False):
    return 100 * next(x["top1"] for x in r["runs"] if x["how"] == how and x["hard"] == hard)


def main():
    e7 = [r for r in rows("exp07.txt") if r["read"] == "hartley"]
    e8 = rows("exp08.txt")
    best = [r for r in e8 if (r["input"], r["plates"], r["read"], r["stages"]) == ("pixel", "xy", "image", 4)]
    big = [r for r in e8 if r["stages"] == 6]
    low = [r for r in e8 if r["read"] == "lowrank"]
    e9 = [r for r in rows("exp09.txt") if r["mode"] == "light" and r["n"] == 64]
    items = [
        ("実験07 の光\n（マス + 4f 系 + ハートレー）", [acc(r, "L") for r in e7], "光 13 万", "#c66"),
        ("実験08 直した光\n（+ 像の面の板・像の面で読む）", [acc(r, "L") for r in best], "光 15 万", "#5aa0b5"),
        ("実験08b 大きく・長く\n（6 段 × 幅 8、2 つの読み方）", [acc(r, "L") for r in big], "光 42 万", "#1f6f8b"),
        ("実験09 語板（sincnn3 の形）\n（1 語 1 台、上位 61 語）", [acc(r, "G") for r in e9], "光 1,644 万・対応表 28 万", "#2b7a3d"),
        ("比べる: デジタルの低ランク", [acc(r, "L") for r in low], "デジタル 420 万", "#999"),
        ("比べる: 数え上げの PMI", [acc(r, "B") for r in e7], "数えた表", "#c8a27a"),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    y = np.arange(len(items))[::-1]
    for yi, (name, v, size, c) in zip(y, items):
        m = float(np.mean(v))
        ax.barh(yi, m, color=c, xerr=[[m - min(v)], [max(v) - m]] if len(v) > 1 else None, capsize=4)
        ax.text(max(v) + 1.2, yi, f"{m:.1f}%（{len(v)} 回）{size}", va="center", fontsize=9)
    ax.axvline(10, color="#888", ls=":", lw=1)
    ax.text(10.5, y[-1] - 0.45, "でたらめ 10%", fontsize=8, color="#666")
    ax.set_yticks(y, [i[0] for i in items], fontsize=9)
    ax.set_xlim(0, 85)
    ax.set_xlabel("10 択の正解率（%、ほかの会話からの 9 つと比べる）")
    ax.set_title("光の共鳴、単独で")
    # 右: 0.1.0 の芯（重なり + PMI）に足したとき
    pairs = [("実験08 直した光", best, "L", "ABL", "#5aa0b5"), ("実験09 語板", e9, None, None, "#2b7a3d")]
    xs = []
    for k, (hard, lab) in enumerate(((False, "ほかの会話から"), (True, "同じ会話から"))):
        d8 = [acc(r, "ABL", hard) - acc(r, "AB", hard) for r in best]
        bx.scatter([k - 0.12] * len(d8), d8, color="#5aa0b5", s=40, label="実験08 直した光（3 seed）" if k == 0 else None)
        bx.plot([k - 0.22, k - 0.02], [np.mean(d8)] * 2, color="#5aa0b5")
        d9 = [acc(r, "AG", hard) - acc(r, "AB", hard) for r in e9]
        bx.scatter([k + 0.12] * len(d9), d9, color="#2b7a3d", s=40, marker="s", label="実験09 語板（置き換え、3 seed）" if k == 0 else None)
        bx.plot([k + 0.02, k + 0.22], [np.mean(d9)] * 2, color="#2b7a3d")
        xs.append(lab)
    bx.axhline(0, color="#888", lw=1)
    bx.set_xticks([0, 1], xs)
    bx.set_ylabel("重なり + PMI（kotoyui 0.1.0 の芯）からの差（点）")
    bx.set_title("0.1.0 の芯に足す（08）・置き換える（09）")
    bx.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "images", "exp08_09_resonance.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
