"""仕組みの図（intervention/images/mech_*.png）: 語板と口・逆から入れる／経験の作り方／判断のときの光の通し方

実行: python intervention/tools_fig_mechanism.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
LIGHT = {BLUE: "#dce9f9", ORANGE: "#fbe2d6", AQUA: "#d3f0e5", YELLOW: "#fbecc8", VIOLET: "#e2def7", MUTED: "#eeede9"}


def box(ax, x, y, w, h, text, col=MUTED, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=LIGHT.get(col, "#f5f5f2"), ec=col, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, col=INK2, text=None, fs=8, rad=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.3, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color=INK2)


def setup(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def fig_board():
    """図 1: 語板（光の器）と口、順と逆。I2・I3。"""
    fig, ax = setup(13, 6.2)
    ax.text(0.2, 5.85, "図 1  語板（光の器）と「口」: 順に通すと属性の明るさ、逆から入れると口への戻り（I2・I3）", fontsize=12, color=INK, fontweight="bold")
    # 口
    box(ax, 0.3, 3.7, 1.7, 0.8, "口 A\n（周波数 1 の波）", BLUE)
    box(ax, 0.3, 2.5, 1.7, 0.8, "口 B\n（周波数 2 の波）", ORANGE)
    # 板
    for i in range(6):
        ax.add_patch(plt.Rectangle((2.9 + i * 0.42, 1.9), 0.12, 3.2, fc="#c3c2b7", ec="none"))
    ax.text(3.95, 5.25, "位相の板 8 枚 + 回折\n（学ぶのは板の位相だけ）", ha="center", fontsize=8.5, color=INK2)
    # レンズ
    ax.add_patch(matplotlib.patches.Ellipse((5.9, 3.5), 0.35, 3.0, fc="#dce9f9", ec=BLUE, lw=1.2))
    ax.text(5.9, 1.75, "レンズ\n（フーリエ変換）", ha="center", fontsize=8.5, color=INK2, va="top")
    arrow(ax, 2.0, 4.1, 2.85, 3.8, BLUE)
    arrow(ax, 2.0, 2.9, 2.85, 3.2, ORANGE)
    arrow(ax, 5.4, 3.5, 5.7, 3.5)
    # 出口の属性
    attrs = [("種類:果物", BLUE, 0.95), ("味:甘い", BLUE, 0.9), ("色:赤", BLUE, 0.7), ("種類:人", ORANGE, 0.95), ("する:手を振る", ORANGE, 0.8), ("（拡散: 対応なし）", MUTED, 0.15)]
    for i, (t, c, b) in enumerate(attrs):
        y = 5.0 - i * 0.62
        ax.add_patch(plt.Rectangle((6.5, y - 0.18), 1.9 * b, 0.36, fc=c, alpha=0.85 if c != MUTED else 0.4, ec="none"))
        ax.text(8.55, y, t, va="center", fontsize=9, color=INK)
    arrow(ax, 6.1, 3.5, 6.45, 3.5)
    ax.text(7.4, 5.45, "出口: 属性ごとの明るさ = 割合", ha="center", fontsize=9, color=INK2)
    # 説明
    box(ax, 10.1, 3.6, 2.7, 1.9,
        "順: 口 A を通すと\n「果物・甘い・赤」が明るい\n口 B を通すと\n「人・手を振る」が明るい\n（文ごとの組で学ぶと、\n境目を教えなくても分かれる）", BLUE, fs=8.5)
    box(ax, 10.1, 1.2, 2.7, 2.1,
        "逆: 「甘い」の点に光を入れ、\n板を逆にたどる\n→ 口 A にほぼ全部戻る\n「野菜」（共通）は\nA と B に半分ずつ戻る\n= 2 つの経路にまとまる", ORANGE, fs=8.5)
    arrow(ax, 9.95, 2.3, 8.45, 4.4, ORANGE, rad=0.25)
    ax.text(0.3, 0.55, "分けるかの判断: 両方の口に光がある観点で値が食い違う（競合）数と、片方の口にしか光がない観点の数を数える。"
            "競合 2 つ以上、または競合 1 つ + 片方だけ 1 つ以上なら「分ける」（I3。I8b で光の判断に置き換え）。", fontsize=8.5, color=INK2, wrap=True)
    save(fig, "mech_1_board.png")


def fig_experience():
    """図 2: 経験の作り方（I7〜I9）。"""
    fig, ax = setup(13, 6.4)
    ax.text(0.2, 6.05, "図 2  経験の作り方: 世界で判断し、その場面と結論を記号の組にして、結論の種類ごとの口に学ばせる（I7〜I9）", fontsize=12, color=INK, fontweight="bold")
    box(ax, 0.3, 4.2, 2.3, 1.3, "世界 A を読む\n（果物・リンゴ・ウメ…）\n語板を分ける（図 1）", MUTED)
    box(ax, 3.1, 4.2, 2.6, 1.3, "比べる組を作る\n同じ語の板どうし\n1 枚の中（色違いのリンゴ）\n口どうし", MUTED)
    box(ax, 6.2, 4.2, 2.9, 1.3, "観点ごとに判断\n色は味の原因か？\n熟し具合は？\n（規則 or 採点で決める）", YELLOW)
    arrow(ax, 2.6, 4.85, 3.05, 4.85)
    arrow(ax, 5.7, 4.85, 6.15, 4.85)
    box(ax, 9.6, 3.9, 3.2, 1.9,
        "経験 1 つ = 記号の組\n場面: 観点:色 / 支持:多い /\n反例:多い / 共に変わる:あり /\n相手の反例:なし / 割合:中\n結論: 見かけ", AQUA, fs=8.5)
    arrow(ax, 9.1, 4.85, 9.55, 4.85)
    ax.text(0.3, 3.45, "結論の付け方: I8 = 規則（I6 の見直し）が付ける ／ I9 = 27 通りを試して答えの採点がいちばん良いもの（失敗・成功から）",
            fontsize=8.5, color=INK2)
    # 口
    labs = [("本物", BLUE), ("例外あり", YELLOW), ("見かけ", ORANGE), ("不明", MUTED), ("分ける", AQUA), ("分けない", VIOLET)]
    for i, (t, c) in enumerate(labs):
        x = 0.5 + i * 2.1
        box(ax, x, 1.2, 1.8, 1.0, f"口: {t}", c, fs=9.5, bold=True)
        arrow(ax, 11.2, 3.2, x + 0.9, 2.25, MUTED, rad=0.0)
    ax.plot([11.2, 11.2], [3.85, 3.2], color=MUTED, lw=1.3)
    ax.text(0.3, 0.95, "経験の器（光の器 1 台、口 = 結論の種類）: その種類の経験だけを、その口に学ばせる（I8b。口を隠して学ぶと経験が混ざった）",
            fontsize=9, color=INK)
    ax.text(0.3, 0.5, "同じ観点・同じ結論の経験は同じ口に融合する（例: 口「見かけ」には「観点:色・反例:多い」が明るく、口「本物」には「観点:状態・支持:多い・反例:なし」が明るい）。"
            "経験が増えても、器は 1 台・口は結論の種類の数のまま。", fontsize=8.5, color=INK2)
    save(fig, "mech_2_experience.png")


def fig_judge():
    """図 3: 判断のときの光の通し方と比べ方（I7〜I9）。"""
    fig, ax = setup(13, 6.6)
    ax.text(0.2, 6.25, "図 3  判断のとき: いまの場面を記号にし、経験の器の口ごとの明るさで重ね、結論の明るさを比べる（光 1 回）", fontsize=12, color=INK, fontweight="bold")
    box(ax, 0.3, 4.4, 2.8, 1.4, "いまの場面（世界 B）\n「色」は味の原因か？\n支持:多い 反例:なし\n共に変わる:あり（状態）", YELLOW, fs=8.5)
    box(ax, 3.6, 4.4, 2.6, 1.4, "場面の記号を\n器の口ごとに照らす\n（口ごとの明るさの積\n= 起こりやすさ）", MUTED, fs=8.5)
    arrow(ax, 3.1, 5.1, 3.55, 5.1)
    # ports bars: likelihood
    ports = [("本物", BLUE, 0.20), ("例外あり", YELLOW, 0.05), ("見かけ", ORANGE, 0.70), ("不明", MUTED, 0.05)]
    ax.text(7.9, 5.95, "口ごとの起こりやすさ（重みに使う）", fontsize=9, color=INK2, ha="center")
    for i, (t, c, v) in enumerate(ports):
        y = 5.55 - i * 0.42
        ax.add_patch(plt.Rectangle((7.4, y - 0.14), 1.8 * v, 0.28, fc=c, ec="none", alpha=0.85))
        ax.text(7.3, y, t, ha="right", va="center", fontsize=9)
    arrow(ax, 6.2, 5.1, 6.5, 5.1)
    box(ax, 9.8, 4.4, 3.0, 1.4, "重ねて通す\n口の明るさを、起こりやすさの\n割合で足し合わせる\n（光を重ねるのと同じ）", MUTED, fs=8.5)
    arrow(ax, 9.3, 5.1, 9.75, 5.1)
    # conclusions compare
    concl = [("本物", BLUE, 0.15), ("例外あり", YELLOW, 0.04), ("見かけ", ORANGE, 0.62), ("不明", MUTED, 0.03)]
    ax.text(3.0, 3.75, "結論の記号の明るさを比べる", fontsize=9.5, color=INK, ha="center")
    for i, (t, c, v) in enumerate(concl):
        y = 3.35 - i * 0.45
        ax.add_patch(plt.Rectangle((1.6, y - 0.15), 3.2 * v, 0.3, fc=c, ec="none", alpha=0.85))
        ax.text(1.5, y, t, ha="right", va="center", fontsize=9)
    arrow(ax, 11.3, 4.35, 5.1, 3.8, MUTED, rad=0.0)
    box(ax, 6.0, 1.75, 3.2, 1.5, "いちばん明るい結論で効き目を決める\n本物 → 1　例外あり → 支持の割合\n見かけ → 0\n不明 → 見たまま（支持の割合, I9）", AQUA, fs=8.5)
    arrow(ax, 4.9, 2.5, 5.95, 2.5)
    box(ax, 9.8, 1.75, 3.0, 1.5, "答える\n「赤いトマトを緑に塗ったら\n渋くなる？」\n色の効き目 0 → いいえ", BLUE, fs=8.5)
    arrow(ax, 9.2, 2.5, 9.75, 2.5)
    ax.text(0.3, 0.7, "デジタルに残る処理: 場面を記号にする（数を 3 段階に丸める）、口ごとの明るさの積と重み付きの和、いちばん明るい結論を選ぶ。"
            "規則の連なり（否定 → 見直しの順番 → 取り消し）は要らない。", fontsize=8.5, color=INK2)
    ax.text(0.3, 0.3, "経験が場面を区別できないと（例: 世界 B で色と熟し具合の場面が同じ）、観点の記号（観点:色 / 観点:状態）の明るさが決め手になる（I8b でウメのある世界に規則より強かった理由）。",
            fontsize=8.5, color=INK2)
    save(fig, "mech_3_judge.png")


def save(fig, name):
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    out = os.path.join(HERE, "images", name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    fig_board()
    fig_experience()
    fig_judge()
