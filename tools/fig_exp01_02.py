"""実験01・02 の図: 光の 2DWIG と、光の分担の 2DTG ＋ 口調のスイッチを、比較相手と並べる（docs/images/exp01_wig.png・exp02_style.png）

results/exp01.txt・exp02.txt から作る。複数 seed は平均と幅（最小〜最大）。
実行: python tools/fig_exp01_02.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def rows(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, "results", name), encoding="utf-8") if l.strip()]


def agg(rs, key, val):
    d = defaultdict(list)
    for r in rs:
        d[key(r)].append(val(r))
    return {k: (float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v)) for k, v in d.items()}


def bar(ax, names, stats, colors, ylabel, title, ref=None, fmt="{:.1f}"):
    x = np.arange(len(names))
    m = [s[0] for s in stats]
    err = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
    ax.bar(x, m, color=colors, yerr=err, capsize=4)
    for xi, s in zip(x, stats):
        ax.text(xi, s[2] + 0.8, fmt.format(s[0]) + ("" if s[3] == 1 else f"\n（{s[3]} 回）"), ha="center", fontsize=8)
    if ref:
        ax.axhline(ref[0], ls="--", c="k", lw=1)
        ax.text(len(names) - 0.5, ref[0] + 0.8, ref[1], ha="right", fontsize=8)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def main():
    e1 = [r for r in rows("exp01.txt") if not r["spread"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i, (task, title) in enumerate((("polite", "(a) 丁寧か、くだけているか（現代の雑談、2 択）"), ("aozora12", "(b) 一人称の 12 類（青空文庫の小説の会話文）"))):
        rs = [r for r in e1 if r["task"] == task]
        st = agg(rs, lambda r: r["method"], lambda r: r["test_acc"])
        names = [("plates", "光の 2DWIG"), ("lr", "回帰\n（ふつうの計算）"), ("nb", "数えた表")]
        bar(ax[i], [n for _, n in names], [st[k] for k, _ in names], ["#137", "#999", "#bbb"], "正解率（%）", title,
            (rs[0]["majority"], f"いつも同じ答え {rs[0]['majority']:.1f}%"))
        ax[i].set_ylim(0, 105)
        lp = [r for r in rs if r["method"] == "plates"][0]
        lr = [r for r in rs if r["method"] == "lr"][0]
        ax[i].set_title(title + f"\n学習する数: 光 = 光の部分 {lp['params_light']:,}・デジタル {lp['params_digital']:,} / 回帰 = デジタル {lr['params_digital']:,}", fontsize=10)
    fig.suptitle("実験01: 光の 2DWIG — 会話文の語尾・言い回しから、話し方を見分ける（3 seed、幅は最小〜最大）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "images", "exp01_wig.png"), dpi=130)

    e2 = rows("exp02.txt")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    names = [("split", "光の 2DTG"), ("split_tag", "光の 2DTG\n＋スイッチ"), ("gru", "GRU"), ("gru_tag", "GRU\n＋スイッチ"), ("tri", "3-gram"), ("tri_tag", "3-gram\n＋口調の表")]
    cols = ["#9bd", "#137", "#ccc", "#777", "#e8c49a", "#c07a2a"]
    st = agg(e2, lambda r: r["method"], lambda r: r["test_acc"])
    bar(ax[0], [n for _, n in names], [st[k] for k, _ in names], cols, "次のサブワードの正解率（%）", "(a) 次の言葉を当てる（知らない話者の発言）")
    ax[0].set_ylim(0, 40)
    st = agg(e2, lambda r: r["method"], lambda r: r["style_acc"])
    bar(ax[1], [n for _, n in names], [st[k] for k, _ in names], cols, "決めた口調で文が作れた割合（%）",
        "(b) 口調のスイッチ（丁寧・くだけたを決めて 100 文ずつ作る）", (50, "でたらめ 50%"))
    ax[1].set_ylim(0, 110)
    fig.suptitle("実験02: 現代の雑談で、光の分担の 2DTG と口調のスイッチ（光は 3 seed、GRU は 3 seed、3-gram は 1 回）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "images", "exp02_style.png"), dpi=130)
    print("docs/images/exp01_wig.png docs/images/exp02_style.png")


if __name__ == "__main__":
    main()
