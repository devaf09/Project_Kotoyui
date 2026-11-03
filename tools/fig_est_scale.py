"""見積もり E5 の図: 青空文庫 1,000 作品と全 4,477 作品で、見積もり E1〜E4 を並べる（docs/images/est_scale.png）

results/est_*.txt の set（1000 / all）ごとの最後の実行から作る。
実行: python tools/fig_est_scale.py
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
SETS = [("1000", "1,000 作品"), ("all", "4,477 作品")]
COL = {"1000": "#9bd", "all": "#137"}


def rows(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, "results", name), encoding="utf-8") if l.strip()]


def last(rs, **kw):
    m = [r for r in rs if all(r.get(k, "1000" if k == "set" else None) == v for k, v in kw.items())]
    return m[-1] if m else None


def bars(ax, labels, vals, title, ylabel, ref=None):
    x = np.arange(len(labels))
    for i, (s, name) in enumerate(SETS):
        v = [vv[i] for vv in vals]
        ax.bar(x + (i - 0.5) * 0.38, v, 0.38, label=name, color=COL[s])
        for xi, yi in zip(x, v):
            ax.text(xi + (i - 0.5) * 0.38, yi + (0.01 * max(max(vv) for vv in vals)), f"{yi:.2f}" if abs(yi) < 1 else f"{yi:.1f}", ha="center", fontsize=7)
    if ref is not None:
        ax.axhline(ref[0], ls="--", c="r", lw=1)
        ax.text(len(labels) - 0.5, ref[0], ref[1], ha="right", va="bottom", color="r", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    wb = rows("est_wig_bag.txt")
    conds = [("nb", "end123+bi", 0, "real", "表 語尾+2つ組"), ("nb", "end123+bi+tri", 0, "spread", "表 +3つ組\n+広げたラベル"),
             ("nb", "end123+bi", 4096, "real", "表 散らして 4096"), ("lr", "end123+bi", 4096, "real", "回帰 散らして 4096")]
    labs, acc, mac = [], [], []
    for m, f, H, t, lab in conds:
        rr = [last(wb, set=s, method=m, feats=f, H=H, train=t) for s, _ in SETS]
        labs.append(lab)
        acc.append([r["test_acc"] for r in rr])
        mac.append([r["macro_recall"] for r in rr])
    bars(ax[0, 0], labs, acc, "(a) 2DWIG の正解率（12 類。いつも一番多い類は 38.6 / 36.3%）", "正解率（%）")
    bars(ax[0, 1], labs, mac, "(b) 2DWIG の類ごとの正解率の平均（少ない類が当たるか）", "類ごとの平均（%）")
    fc = rows("est_flavor_counts.txt")
    cues = [("oracle", "本当の類"), ("spread", "写した類"), ("prev", "2つ前の発言"), ("prefix", "書きかけ")]
    ga = [[last(fc, set=s, groups="g12", count="soft", cue=c)["gain_acc"] for s, _ in SETS] for c, _ in cues]
    bars(ax[1, 0], [n for _, n in cues], ga, "(c) 隠し味の効き（g12 soft、次のサブワードの正解率の差）", "隠し味あり − なし（点）", (0.5, "判断の基準 +0.5 点"))
    ax[1, 0].set_ylim(-0.2, 0.7)
    st = rows("est_flavor_style.txt")
    cs = [("none", "隠し味なし"), ("mix0.5", "λ=0.5"), ("mix0.8", "λ=0.8"), ("mix1.0", "λ=1.0")]
    sv = [[last(st, set=s, cond=c)["style_acc"] for s, _ in SETS] for c, _ in cs]
    bars(ax[1, 1], [n for _, n in cs], sv, "(d) 口調の当たり（5 組、3 seed の平均。偶然 20%）", "決めた組で当て直された割合（%）")
    fig.suptitle("見積もり E5: データを 4.5 倍にすると（青空文庫、数えた表、GPU なし。テストの作品は組ごとに別）", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "images", "est_scale.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
