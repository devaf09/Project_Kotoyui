"""説明ページ（README・docs/overview.md）の図をまとめて作る（docs/images/overview_*.png）

overview_pipeline.png   返事ができるまでの流れ（グレインと指揮役）
overview_light.png      光の器の 2 つの形（4f 系の光路・語板）の模式図
overview_scoreboard.png 光の部品とふつうの方法の成績の比べ（数字は NOTES.md の 3 seed の平均）
overview_position.png   似た技術との位置づけ（模式図。軸は定性的）
実行: python tools/fig_overview.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "images")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984", "#0b0b0b", "#52514e"
SURF = "#fcfcfb"


def box(ax, x, y, w, h, text, fc, ec=None, fs=10, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec or fc, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, a, b, color=INK2, style="-|>", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls))


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 5.6), facecolor=SURF)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    box(ax, 0.2, 2.4, 1.8, 0.9, "相手の発言\n「犬の散歩が\n大変で」", "#e9e8e4", fs=10)
    # 材料
    box(ax, 2.6, 4.45, 3.3, 0.8, "会話の材料（本物の雑談の発言 33 万）\nここから返事の候補を探す", "#f3f2ee", ec="#c9c8c2", fs=9)
    # グレイン
    gx, gw = 2.6, 3.3
    grains = [("語の重なり（数え上げ）", "#e9e8e4", INK), ("次の発言の共鳴\n語板の光 ✦（0.2.0）", "#cfe1f7", INK),
              ("話の形の移り変わり（数え上げ）", "#e9e8e4", INK), ("口調・キャラクタ・時計・相手の記憶", "#e9e8e4", INK)]
    for i, (t, c, tc) in enumerate(grains):
        y = 3.35 - i * 0.95
        box(ax, gx, y, gw, 0.78, t, c, fs=9.5, color=tc)
        arrow(ax, (2.0, 2.85), (gx, y + 0.39))
        arrow(ax, (gx + gw, y + 0.39), (6.6, 2.85))
    arrow(ax, (4.25, 4.45), (4.25, 4.15), ls="--")
    ax.text(4.25, 0.05, "グレイン = 小さな学習器。点を出して足し合わせる", ha="center", fontsize=9, color=INK2)
    box(ax, 6.6, 2.2, 2.3, 1.3, "指揮役\n共感 →  内容 → 質問\n（候補を点で選ぶ）", "#fde3d6", fs=10)
    box(ax, 9.5, 2.2, 2.3, 1.3, "返事\n「散歩が大変なんですね。\nどんなところが\n一番大変ですか？」", "#d3efe3", fs=9)
    arrow(ax, (8.9, 2.85), (9.5, 2.85))
    ax.text(6.0, 5.4, "返事ができるまで（kotoyui.py 0.2.0）", ha="center", fontsize=13, color=INK)
    ax.text(7.75, 1.7, "✦ = 光のグレインに置き換えた所。\nほかは順に置き換えていく", ha="center", fontsize=9, color=BLUE)
    fig.savefig(os.path.join(OUT, "overview_pipeline.png"), dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def light():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.4), facecolor=SURF)
    for ax in (a1, a2):
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 3.2)
        ax.axis("off")
    # 4f 系の光路
    a1.text(0.1, 2.95, "① 4f 系の光路（実験03〜08）: 1 本の光路にいろいろな入力を通す", fontsize=12, color=INK)
    box(a1, 0.1, 0.9, 1.5, 1.4, "入力の光\n（語のマスが光る）", "#e9e8e4", fs=9)
    xs = [2.2, 3.9, 5.6, 7.3]
    for k, x in enumerate(xs):
        lens = matplotlib.patches.Ellipse((x, 1.6), 0.28, 1.8, fc="#cfe1f7", ec=BLUE)
        a1.add_patch(lens)
        a1.text(x, 0.45, "レンズ", ha="center", fontsize=8.5, color=INK2)
        if k < 3:
            a1.add_patch(Rectangle((x + 0.75, 0.75), 0.12, 1.7, fc="#fde3d6", ec=ORANGE))
            a1.text(x + 0.81, 0.45, "板（学習）", ha="center", fontsize=8.5, color=INK2)
    for x0, x1 in zip([1.6] + xs, xs + [8.9]):
        arrow(a1, (x0 + 0.15, 1.6), (x1 - 0.15, 1.6), color=YELLOW, lw=2)
    box(a1, 8.9, 0.9, 1.9, 1.4, "明るさを読む\n= 答えの点", "#d3efe3", fs=9)
    a1.text(5.0, 2.6, "レンズ → 周波数の面の板 → レンズ を何段も（間で明るさを測って出し直す）", ha="center", fontsize=9, color=INK2)
    # 語板
    a2.text(0.1, 2.95, "② 語板（実験09、Sin-CNN3 のライブラリ sincnn3 の形）: 1 語に 1 台の小さな器", fontsize=12, color=INK)
    words = ["犬", "散歩", "映画"]
    for i, w in enumerate(words):
        y = 2.2 - i * 0.75
        box(a2, 0.1, y, 0.9, 0.55, w, "#e9e8e4", fs=10)
        for j in range(5):
            a2.add_patch(Rectangle((1.4 + j * 0.55, y), 0.1, 0.55, fc="#fde3d6", ec=ORANGE))
        arrow(a2, (1.0, y + 0.27), (1.4, y + 0.27), color=YELLOW, lw=2)
        arrow(a2, (4.1, y + 0.27), (4.6, y + 0.27), color=YELLOW, lw=2)
        bars = np.array([0.5, 0.25, 0.12, 0.08, 0.05]) if i == 0 else np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        for j, b in enumerate(bars):
            a2.add_patch(Rectangle((4.7 + j * 0.25, y), 0.18, 0.55 * b / bars.max(), fc=BLUE, ec=BLUE))
    a2.text(2.6, 0.1, "板（学習）+ 光が広がる（回折）", ha="center", fontsize=8.5, color=INK2)
    a2.text(5.3, 0.1, "出口の明るさ = 次に出やすい語の割合", ha="center", fontsize=8.5, color=INK2)
    box(a2, 7.1, 0.9, 3.7, 1.3, "前の発言の語の器の明るさを足す（融合）\n→ 返事の候補の語に当たった光が多いほど\n「つながる返事」", "#d3efe3", fs=9)
    arrow(a2, (6.0, 1.55), (7.1, 1.55))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overview_light.png"), dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def scoreboard():
    # (課題, 光の部品, ふつうの方法, ふつうの方法の名前, 出典)
    rows = [
        ("話し方が丁寧か（2DWIG、実験01）", 96.5, 96.7, "回帰"),
        ("一人称の類 12 通り（2DWIG、実験01）", 47.7, 49.2, "回帰"),
        ("次のサブワード 4096 通り（2DTG、実験05）", 27.2, 28.3, "GRU"),
        ("口調の切り替え（2DTG、実験03、seed 0）", 92.5, 91.0, "3-gram"),
        ("返事選び 10 択: 共鳴（語板、実験09）", 57.9, 59.0, "数え上げ"),
        ("返事選び 10 択: 共鳴（1 本の光路、実験08）", 48.0, 59.0, "数え上げ"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    y = np.arange(len(rows))[::-1]
    h = 0.36
    for yi, (name, lv, bv, bn) in zip(y, rows):
        ax.barh(yi + h / 2 + 0.01, lv, h, color=BLUE)
        ax.barh(yi - h / 2 - 0.01, bv, h, color=GRAY)
        ax.text(lv + 0.8, yi + h / 2, f"光 {lv:.1f}%", va="center", fontsize=9, color=INK)
        ax.text(bv + 0.8, yi - h / 2, f"{bn} {bv:.1f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("正解率（%）", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=GRAY)],
              labels=["光の部品", "ふつうの方法（比べる相手）"], loc="lower right", fontsize=9, frameon=False)
    ax.set_title("光の部品は、ふつうの方法とどれくらい並ぶか（実験03 以外は 3 seed の平均）", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overview_scoreboard.png"), dpi=130, facecolor=SURF)
    plt.close(fig)


def position():
    fig, ax = plt.subplots(figsize=(10, 6.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(5, color="#d6d5d0", lw=1)
    ax.axvline(5, color="#d6d5d0", lw=1)
    ax.text(0.1, 4.6, "ふつうの計算機（デジタル）", fontsize=9, color=INK2)
    ax.text(0.1, 9.6, "光の計算（干渉・回折）", fontsize=9, color=INK2)
    ax.text(9.9, 0.15, "会話・文を扱う →", fontsize=9, color=INK2, ha="right")
    ax.text(0.1, 0.15, "← 画像・分類・記憶", fontsize=9, color=INK2)
    pts = [
        (8.6, 2.0, "大規模言語モデル\n（LLM、ChatGPT など）", GRAY),
        (6.8, 3.4, "検索型の会話ボット\n（材料から返事を選ぶ）", GRAY),
        (6.0, 1.6, "数え上げ（n-gram・PMI）", GRAY),
        (2.0, 2.4, "ふつうのニューラルネット\n（画像の分類など）", GRAY),
        (1.8, 8.0, "回折の深層ネットワーク\n（D²NN、2018〜）", YELLOW),
        (3.4, 6.4, "光学相関器\n（4f 系、1964〜）", YELLOW),
        (4.0, 8.9, "光のトランスフォーマー\n（小規模の実験、2023）", YELLOW),
        (7.6, 7.6, "ことゆい（このプロジェクト）\n光の小さなグレインで会話", BLUE),
    ]
    for x, y, t, c in pts:
        ax.scatter([x], [y], s=260 if c == BLUE else 140, color=c, edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(x, y - 0.45, t, ha="center", va="top", fontsize=9.5 if c == BLUE else 9, color=INK)
    ax.set_title("似た技術との位置づけ（模式図。位置は大まかな目安）", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overview_position.png"), dpi=130, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline()
    light()
    scoreboard()
    position()
    print(OUT)
