"""算数の仕組みの板の中身をビットマップで描く（非公開の置き場の images/mech_10_*.png 〜）: I17・I21〜I23

  図 10  一般常識の数の板 0〜99: 光の器の出口の明るさ（一の位・十の位・偶奇）
  図 11  0〜9 の板の演算の帯（I22）: 7 の板の（役, 演算, 相手）→ 光る出口（一の位・桁上り）と、表の半分を教えたときの気付きの埋まり方
  図 12  バッファ板（I21〜I23）: 47 ＋ 35 → 82。値の帯 = 一般常識の 82 の板の明るさ、参照の帯 = 47・＋・35
  図 13  文法解析グレインの経験の器（I23、命令ごと）: 命令 × 前後の位置 × 板の種類 の画素。縮める / 待つ の明るさの比
持ち主（2026-11-06）: 「算数の仕組みの板の中身を見てみたいんだけど、ビットマップみたいに表示してくれる？」
実行: python intervention/tools_fig_boards.py
"""
from __future__ import annotations

import math
import os
import random
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools_fig_mechanism as F  # noqa: E402
from tools_fig_mechanism import INK, INK2, plt  # noqa: E402
import i17_arithmetic as M  # noqa: E402
import i19_grain_arith as G  # noqa: E402
import i23_digits as D  # noqa: E402
from i21_board_alu import OpBands  # noqa: E402
from i22_carry import CarryALU, digit_fact, D10  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "IPAGothic", "sans-serif"]
OUT = os.path.join(F.PRIV, "images")
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cfe0f6", "#6fa3e6", "#2a78d6", "#123a6b"])       # 1 色の濃淡（明るさ）
DIV = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f2b394", "#e9e8e4", "#9cc0ec", "#1d5fb3"])       # 待つ ← 灰 → 縮める
ATTR_LABELS = [str(d) for d in range(10)] + [str(d) for d in range(10)] + ["偶", "奇"]


def title(fig, s):
    fig.suptitle(s, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")


def band_ticks(ax):
    ax.set_xticks(range(22))
    ax.set_xticklabels(ATTR_LABELS, fontsize=7, color=INK2)
    for x in (9.5, 19.5):
        ax.axvline(x, color="#ffffff", lw=2)
    for x, t in ((4.5, "一の位"), (14.5, "十の位"), (20.5, "偶奇")):
        ax.text(x, 1.01, t, ha="center", va="bottom", fontsize=9, color=INK, transform=ax.get_xaxis_transform())


def fig_general(general):
    fig, ax = plt.subplots(figsize=(10, 11))
    title(fig, "図 10  一般常識の数の板 0〜99（I17 の光の器）: 行 = 数の板、列 = 出口（属性）、色 = 明るさ")
    L = np.array([general[x] for x in range(100)])
    im = ax.imshow(L, aspect="auto", cmap=SEQ, vmin=0, interpolation="nearest")
    band_ticks(ax)
    ax.set_yticks(range(0, 100, 5))
    ax.set_yticklabels(range(0, 100, 5), fontsize=7, color=INK2)
    ax.set_ylabel("数の板", color=INK2)
    ax.set_xlabel("出口（属性）", color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("明るさ（属性の割合）", color=INK2)
    fig.text(0.01, 0.01, "どの板も、一の位・十の位・偶奇の 3 つの観点に 1 つずつ明るい出口がある（例: 47 は 一7・十4・奇）。足りない明るさは暗い出口に逃げている。",
             fontsize=9, color=INK2)
    return fig


def fig_opband(ob, half):
    fig = plt.figure(figsize=(13, 8.5))
    title(fig, "図 11  0〜9 の板の演算の帯（I22）: 7 の板が「相手と演算」を受けたときに光る出口と、表の半分を教えたときの気付き")
    ax = fig.add_axes([0.05, 0.1, 0.42, 0.78])
    rows, labels = [], []
    for op in ("+", "-", "*"):
        for role in ("左", "右"):
            for y in D10:
                c = ob.band[7].get((role, op, y), Counter())
                v = np.zeros(20)
                for (d, carry), n in c.items():
                    v[d] += n
                    v[10 + carry] += n
                rows.append(v / max(1, sum(c.values())))
                labels.append(f"{role} {G.OPS_SYM.get(op, op) if hasattr(G, 'OPS_SYM') else op} {y}")
    R = np.array(rows)
    ax.imshow(R, aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(20))
    ax.set_xticklabels([str(d) for d in range(10)] * 2, fontsize=7, color=INK2)
    ax.axvline(9.5, color="#ffffff", lw=2)
    for x, t in ((4.5, "出口: 一の位"), (14.5, "出口: 桁上り・桁下がり")):
        ax.text(x, 1.005, t, ha="center", va="bottom", fontsize=9, color=INK, transform=ax.get_xaxis_transform())
    ax.set_yticks(range(60))
    ax.set_yticklabels(labels, fontsize=5.5, color=INK2)
    for k in range(1, 6):
        ax.axhline(k * 10 - 0.5, color="#ffffff", lw=2)
    ax.set_title("7 の板の演算の帯（行 = 役・演算・相手、列 = 光る出口）\n例: 左 + 5 → 一の位 2・桁上り 1（7 + 5 = 12）", fontsize=9, color=INK2, loc="left", pad=20)
    # 右: 半分を教えたとき
    ax2 = fig.add_axes([0.55, 0.1, 0.42, 0.78])
    S = np.zeros((10, 30))
    for k, op in enumerate(("+", "-", "*")):
        for x in D10:
            for y in D10:
                taught = bool(half.band[x].get(("左", op, y)))
                swap = (not taught) and half.swap.get(op) and bool(half.band[x].get(("右", op, y)) or half.band[y].get(("左", op, x)))
                S[x, k * 10 + y] = 2 if taught else (1 if swap else 0)
    cmap = LinearSegmentedColormap.from_list("st", ["#eeede9", "#9cc0ec", "#2a78d6"], N=3)
    ax2.imshow(S, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax2.set_xticks(range(30))
    ax2.set_xticklabels([str(y) for _ in range(3) for y in D10], fontsize=7, color=INK2)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f"{x} の板" for x in D10], fontsize=8, color=INK2)
    for x in (9.5, 19.5):
        ax2.axvline(x, color="#ffffff", lw=3)
    for k, op in enumerate(("＋", "−", "×")):
        ax2.text(k * 10 + 4.5, 1.01, op, ha="center", va="bottom", fontsize=12, color=INK, transform=ax2.get_xaxis_transform())
    ax2.set_title("表の半分だけ教えたとき（列 = 演算ごとの相手 0〜9）\n濃い青 = 教えた、薄い青 = 入れ替えの気付きで埋まった、灰 = わからない", fontsize=9, color=INK2, loc="left", pad=22)
    fig.text(0.05, 0.02, "＋ と × は左右の結果が一致するので気付きで埋まる。− は一致しないので埋まらない（I21: 0.52 → 0.75）。", fontsize=9, color=INK2)
    return fig


def fig_buffer(general):
    fig = plt.figure(figsize=(12, 5.2))
    title(fig, "図 12  バッファ板（I21〜I23）: 47 ＋ 35 → 82。値の帯 = 一般常識の 82 の板の明るさ、参照の帯 = 元の 3 枚")
    ax = fig.add_axes([0.08, 0.32, 0.6, 0.48])
    L = np.array([general[47], general[35], general[82]])
    ax.imshow(L, aspect="auto", cmap=SEQ, vmin=0, interpolation="nearest")
    band_ticks(ax)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["47 の板", "35 の板", "バッファ板 82\n（値の帯）"], fontsize=9, color=INK2)
    ax.axhline(1.5, color="#ffffff", lw=4)
    ax2 = fig.add_axes([0.72, 0.32, 0.25, 0.48])
    ax2.axis("off")
    for k, (t, c) in enumerate((("参照の帯（ネスト）", INK), ("左: 47 の板", INK2), ("演算: ＋ の板", INK2), ("右: 35 の板", INK2))):
        ax2.text(0.02, 0.9 - k * 0.22, t, fontsize=10 if k else 11, color=c, fontweight="bold" if k == 0 else "normal", transform=ax2.transAxes)
    fig.text(0.08, 0.08, "値の帯は 82 の板の明るさそのもの（一2・十8・偶）。計算の結果を「数」として読むのはこの帯から。\n"
             "参照の帯は、どの板からできたかを指す。これをたどると 82 → 47 ＋ 35 に戻る（劣化なしで 1.00）。z（全バッファ板の和）だけでは戻らない。",
             fontsize=9, color=INK2, linespacing=1.6)
    return fig


def train_decoder(seed=0, n=300):
    general = M.learn_boards(list(range(100)), 10, seed)
    ob = OpBands()
    for op in G.OPS:
        for x in D10:
            for y in D10:
                ob.teach(x, op, y, digit_fact(x, op, y))
    ob.notice()
    D.DigitArr.alu, D.DigitArr.general = CarryALU(ob), general
    rng = random.Random(seed)
    train = D.make(rng, 2, 3, n)
    dec = D.Decoder()
    st, log = dict(solves=0, found=0, given_up=0), Counter()
    for t, toks, v in train:
        D.LEARN_REFUSAL = True
        D.learn_one(dec, toks, v, st, log)
        D.LEARN_REFUSAL = False
    return dec


def fig_decoder(dec):
    rows = dec.e.ep["命令"]
    kinds = ["字", "数", "+", "-", "*", "(", ")", "端"]
    ins_list = ["まとめる", "読む", "縮める", "外す"]
    offs = [-2, -1, 0, 1, 2]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    title(fig, "図 13  文法解析グレインの経験の器（I23、命令ごと、式 300 で学んだあと）: 画素 = 命令 × 位置 × 板の種類、色 = 縮める / 待つ の明るさの比")
    for ax, ins in zip(axes, ins_list):
        rr = [(r, s) for r, s in rows if f"操作:{ins}" in r]
        pos = [r for r, s in rr if s == "縮める"]
        neg = [r for r, s in rr if s == "待つ"]
        Mx = np.full((len(offs), len(kinds)), np.nan)
        for i, d in enumerate(offs):
            for j, k in enumerate(kinds):
                f = f"操作:{ins}&{d}:{k}"
                a = sum(f in r for r in pos)
                b = sum(f in r for r in neg)
                if a + b:
                    Mx[i, j] = math.log((a + 0.1) / (len(pos) + 0.2)) - math.log((b + 0.1) / (len(neg) + 0.2))
        im = ax.imshow(np.clip(Mx, -5, 5), cmap=DIV, vmin=-5, vmax=5, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(["字", "数", "＋", "−", "×", "（", "）", "端"], fontsize=9, color=INK2)
        ax.set_yticks(range(len(offs)))
        ax.set_yticklabels(["2 つ前", "1 つ前", "その場", "1 つ後", "2 つ後"], fontsize=9, color=INK2)
        ax.set_title(f"{ins}（経験 {len(pos)} / {len(neg)}）", fontsize=10, color=INK)
        ax.set_facecolor("#ffffff")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("← 待つ ｜ 縮める →（対数の比）", color=INK2)
    fig.text(0.01, -0.08, "白 = 経験に無い画素。経験の数は（縮める / 待つ）。まとめる: 1 つ後が字なら青（2 桁にまとめる）、数・＋・−・端なら赤（断られて待つ）。\n"
             "縮める: 2 つ後（次の演算）が ＋・− なら青（先に縮めてよい）、1 つ前・1 つ後が字なら赤（まだ数になっていない）。", fontsize=9, color=INK2, linespacing=1.6)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    general = M.learn_boards(list(range(100)), 10, 0)
    ob = OpBands()
    for op in G.OPS:
        for x in D10:
            for y in D10:
                ob.teach(x, op, y, digit_fact(x, op, y))
    half = OpBands()
    r2 = random.Random(5)
    for op in G.OPS:
        for x in D10:
            for y in D10:
                if r2.random() < 0.5:
                    half.teach(x, op, y, digit_fact(x, op, y))
    half.notice()
    dec = train_decoder()
    for name, fig in (("mech_10_number_boards", fig_general(general)), ("mech_11_op_bands", fig_opband(ob, half)),
                      ("mech_12_buffer", fig_buffer(general)), ("mech_13_decoder", fig_decoder(dec))):
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(os.path.join(OUT, name + ".png"))


if __name__ == "__main__":
    main()
