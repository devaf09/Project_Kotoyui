"""しくみの図: デジタルの符号の問題 — 光の部品の重みの大半が、答えを読むためのデジタルの符号（docs/images/codes_problem.png）

(a) 光の部品が答えを出すまでの流れ: 光の場 → 板と回折（光の部分）→ スクリーンの窓 256 個の明るさ → 符号の表（答え 4096 × 窓 256、デジタルの部分）→ 答えの点数。
(b) 学習する数を光の部分とデジタルの部分に分けて並べる（実験01・02 の実測と、実験03 の 4f 系の光路 1 本の設計の値）。
実行: python tools/fig_codes_problem.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
LIGHT, DIGI = "#1f6f8b", "#d9822b"


def first(name, **kw):
    for l in open(os.path.join(ROOT, "results", name), encoding="utf-8"):
        r = json.loads(l)
        if all(r.get(k) == v for k, v in kw.items()):
            return r


def main():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0.02, 0.47, 0.96, 0.46])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    # 光の部分
    ax.add_patch(Rectangle((1, 18), 22, 26, fc="#e3f1f5", ec=LIGHT, lw=1.5))
    ax.text(12, 40, "光の部分", ha="center", fontsize=12, color=LIGHT, )
    for i in range(5):
        ax.add_patch(Rectangle((4 + i * 3.6, 22), 1.6, 14, fc=LIGHT, alpha=0.35 + 0.1 * i, ec="none"))
    ax.text(12, 19.5, "板・回折・レンズ\n（学ぶのは板の模様）", ha="center", fontsize=9)
    # スクリーンと窓
    ax.add_patch(Rectangle((29, 18), 22, 26, fc="#f4f6fa", ec="#556", lw=1.2))
    rng = np.random.default_rng(0)
    for r in range(16):
        for c in range(16):
            v = rng.random()
            ax.add_patch(Rectangle((30.2 + c * 1.25, 19.6 + r * 1.25), 1.0, 1.0, fc=plt.cm.viridis(v), ec="none"))
    ax.text(40, 45.5, "スクリーンの窓 256 個の明るさ", ha="center", fontsize=10)
    # 符号の表
    ax.add_patch(Rectangle((57, 6), 20, 50, fc="#fdf1e4", ec=DIGI, lw=1.5))
    for r in range(40):
        for c in range(16):
            ax.add_patch(Rectangle((58 + c * 1.12, 8.5 + r * 1.03), 0.95, 0.85, fc=plt.cm.RdBu(rng.random()), ec="none", alpha=0.8))
    ax.text(67, 52.5, "デジタルの部分: 符号の表", ha="center", fontsize=11, color=DIGI, )
    ax.text(67, 2.0, "答え 4096 種類 × 窓 256 個\n= 104 万 8,576 個の数（光路 1 本）", ha="center", fontsize=9.5)
    # 答え
    ax.add_patch(Rectangle((83, 18), 15, 26, fc="#f4f6fa", ec="#556", lw=1.2))
    sc = np.sort(rng.random(12))[::-1]
    for i, v in enumerate(sc):
        ax.add_patch(Rectangle((84.5, 41 - i * 1.85), 12 * v, 1.4, fc="#8899aa", ec="none"))
    ax.text(90.5, 45.5, "答えの点数", ha="center", fontsize=10)
    ax.text(90.5, 14.5, "4096 種類のどれが\n次の言葉か", ha="center", fontsize=9)
    for x0, x1 in ((23.5, 28.5), (51.5, 56.5), (77.5, 82.5)):
        ax.add_patch(FancyArrowPatch((x0, 31), (x1, 31), arrowstyle="-|>", mutation_scale=16, color="#334"))
    ax.text(54, 34, "掛け算\n（窓の模様と\n各答えの符号の\n似ている度合い）", ha="center", fontsize=7.5)
    ax.set_title("(a) 光の部品が答えを出すまで（次の言葉を当てる場合）", fontsize=11, loc="left")

    bx = fig.add_axes([0.24, 0.05, 0.56, 0.33])
    e1 = first("exp01.txt", task="polite", method="plates")
    e2 = first("exp02.txt", method="split")
    rows = [("光の 2DWIG\n（丁寧さ、答え 2 種類）", e1["params_light"], e1["params_digital"]),
            ("板の分担の 2DTG\n（次の言葉、答え 4096）", e2["params_light"], e2["params_digital"]),
            ("4f 系の光路 1 本\n（実験03 の設計、答え 4096）", 2 * 6 * 8 * 4096 + 6 * 8 * 2 + 5, 4096 * 256 + 1)]
    y = np.arange(len(rows))[::-1]
    for yi, (name, l, d) in zip(y, rows):
        tot = l + d
        bx.barh(yi, l / tot * 100, color=LIGHT)
        bx.barh(yi, d / tot * 100, left=l / tot * 100, color=DIGI)
        bx.text(101, yi, f"光 {l:,}\nデジタル {d:,}\n（デジタル {d / tot * 100:.0f}%）", va="center", fontsize=8.5)
    bx.set_yticks(y, [r[0] for r in rows], fontsize=9)
    bx.set_xlim(0, 100)
    bx.set_xlabel("学習する数のうちの割合（%）")
    bx.set_title("(b) 学習する数の内訳（青 = 光の部分、橙 = デジタルの部分）", fontsize=11, loc="left")
    fig.suptitle("デジタルの符号の問題: 答えの種類が多いほど、答えを読むための表（デジタル）が大きくなる", fontsize=13)
    out = os.path.join(ROOT, "docs", "images", "codes_problem.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
