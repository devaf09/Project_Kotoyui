"""実験I2 の図: 融合した語板（口 2 つ）に属性を 1 つずつ逆から入れ、口 A に戻る光の割合を並べる（intervention/images/i02_reverse.png）

intervention/results/i02.jsonl の seed 0、学び方 mix・口 2 つの行から。棒の色 = 正解（片方の物だけの属性 / 共通の属性）。濃い部分 = 口 A に戻る割合、薄い部分 = 口 B。
実行: python intervention/tools_fig_i02.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
PICK = ["リンゴ: 果物 + 人", "トマト: 完熟 + 未熟", "比べ: リンゴ/果物を半分ずつ（赤と緑がある）", "比べ: トマト/完熟を半分ずつ"]
COL = {0: "#2a78d6", 1: "#eb6834", -1: "#8a8984"}


def main(seed=0):
    R = [json.loads(l) for l in open(os.path.join(HERE, "results", "i02.jsonl"), encoding="utf-8") if l.strip()]
    rows = {r["group"]: r for r in R if r["seed"] == seed and r["how"] == "mix" and r["M"] == 2}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, g in zip(axes, PICK):
        r = rows[g]
        at = r["attrs"]
        y = range(len(at))
        pa = [a["port"][0] for a in at]
        ax.barh(list(y), pa, color=[COL[a["own"]] for a in at], height=0.7)                          # 口 A に戻る分（濃い）
        ax.barh(list(y), [1 - x for x in pa], left=pa, color=[COL[a["own"]] for a in at], alpha=0.25, height=0.7)   # 口 B に戻る分（薄い）
        ax.set_yticks(list(y))
        ax.set_yticklabels([a["attr"] for a in at], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="#c3c2b7", lw=0.8, ls=":")
        dec = "分ける" if r["split"] else "分けない"
        ax.set_title(f"{g}\n競合: {'・'.join(r['compete']) or 'なし'} / 片方だけ: {len(r['exclusive'])} → {dec}", fontsize=9)
        ax.set_xlabel("逆から入れた光が戻る割合（濃い = 口 A、薄い = 口 B）", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in (0, 1, -1)]
    fig.legend(h, ["1 つ目の物だけの属性", "2 つ目の物だけの属性", "共通の属性（比べでは全部）"], loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"実験I2: 融合した語板に属性を 1 つずつ逆から入れる（口 2 つ、seed {seed}）", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    out = os.path.join(HERE, "images", "i02_reverse.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
