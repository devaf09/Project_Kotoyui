"""記憶の仕組みの全体図（I1〜I10 の時点）: どこに何を覚え、いつ書き、どう思い出すか

図は生成物なので、非公開の置き場 ../Project_Kotoyui_private/intervention/images/ に書き出す（持ち主 2026-11-05。tools_fig_mechanism.py と同じ）。
実行: python intervention/tools_fig_memory.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.environ.get("KOTOYUI_PRIVATE_INTERVENTION", os.path.join(HERE, "..", "..", "Project_Kotoyui_private", "intervention"))
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "IPAGothic", "sans-serif"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
LIGHT = {BLUE: "#dce9f9", ORANGE: "#fbe2d6", AQUA: "#d3f0e5", YELLOW: "#fbecc8", VIOLET: "#e2def7", MUTED: "#eeede9"}


def box(ax, x, y, w, h, text, col=MUTED, fs=9, bold=False, ha="center", dash=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=LIGHT.get(col, "#f5f5f2"), ec=col, lw=1.4,
                                ls="--" if dash else "-"))
    tx = x + w / 2 if ha == "center" else x + 0.15
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.45)


def title(ax, x, y, text, col, fs=10.5):
    ax.text(x, y, text, fontsize=fs, color=col, fontweight="bold", va="bottom")


def arrow(ax, x0, y0, x1, y1, col=INK2, text=None, fs=8, rad=0.0, dx=0.0, dy=0.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.3, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color=INK2, linespacing=1.3)


def fig_memory(out):
    W, H = 16, 10.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(0.2, 9.75, "図 4  記憶の全体図: 光に覚えるもの 3 つ（語板・親の語板・経験の器）と、手で与えるデジタルの表（I1〜I10 の時点）",
            fontsize=13, color=INK, fontweight="bold")

    # ---- 上: 手で与えるもの（デジタル）
    box(ax, 0.3, 8.45, 15.4, 0.95,
        "手で与える（デジタル、最初の成長段階）:  語 → 属性の対応表（「赤い」= 色:赤、「手を振った」= する:手を振る + 種類:人）  ／  "
        "動詞が変える観点（塗る = 色、熟させる = 状態）\n"
        "「X の Y」なら Y の親は X（I10）  ／  分ける・否定のしきい値（違い 0.5、反例の強さ 5 など）  ／  場面を記号に丸める区切り（なし・少し・多い）",
        MUTED, fs=8.8, ha="left")

    # ---- 入口: 文
    box(ax, 0.3, 6.3, 2.0, 1.3, "文を 1 つずつ読む\n「赤い リンゴ は\n甘い 。」", MUTED, fs=9)
    arrow(ax, 1.3, 8.45, 1.3, 7.62, MUTED, "対応表で\n属性に", fs=7.5, dx=0.55, dy=-0.15)

    # ---- 記憶 1: 語板
    title(ax, 3.0, 7.75, "記憶 1  語板（語の意味の記憶）", BLUE)
    box(ax, 3.0, 5.2, 4.3, 2.45,
        "1 つの意味 = 1 台の光の器（位相の板 8 枚 + レンズ）\n"
        "出口の明るさ = 属性の割合\n"
        "（リンゴ/果物: 果物 0.9・甘い 0.9・赤 0.7 …）\n"
        "対応のない属性には光が届かない（拡散）\n"
        "器の入口に「口」を持てる（I2〜）:\n"
        "1 語の中の別の物・別の状態を口ごとに持つ",
        BLUE, fs=8.6)
    arrow(ax, 2.3, 6.95, 2.98, 6.95, INK2)
    box(ax, 3.0, 3.55, 4.3, 1.35,
        "書く: 文を読むたびに板の位相を学ぶ\n"
        "分ける: 既存と違う（I1）／口 2 つで競合する観点が\n"
        "2 つ以上（I3〜）→ 同じ文字列で新しい語板",
        BLUE, fs=8.3, dash=True)
    box(ax, 3.0, 2.1, 4.3, 1.15,
        "読む: 順に通す → 属性の明るさ（見たままの問い）\n"
        "逆から入れる → どの口に戻るか（分けるかの自覚）\n"
        "違いの大きさ → 違和感（おかしな文）",
        BLUE, fs=8.3, dash=True)

    # ---- 記憶 2: 親の語板
    title(ax, 8.0, 7.75, "記憶 2  親の語板（親子の記憶、I10）", VIOLET)
    box(ax, 8.0, 5.2, 3.6, 2.45,
        "親 = 子の語板の明るさの和\n（光の融合。親の器は別に学ばない）\n\n"
        "リンゴ科 = フジ + 紅玉 + 王林 + …\n→ 赤・緑・黄、甘い・酸っぱい の幅\n"
        "祖先（果物全体）= 全部の科の和",
        VIOLET, fs=8.6)
    arrow(ax, 7.3, 6.4, 7.98, 6.4, VIOLET, "子の明るさ\nを足す", fs=7.5, dy=0.05)
    box(ax, 8.0, 3.55, 3.6, 1.35,
        "書く: 子の語板が育つと、和も自動で変わる\n"
        "（住所: 日本 → 宮崎 → 日南 は、深さごとに\n0.5 倍で受け継ぐ形だけ確かめた）",
        VIOLET, fs=8.1, dash=True)
    box(ax, 8.0, 2.1, 3.6, 1.15,
        "読む: 知らない子（新しい品種）の属性を\n親の明るさから確率で想定\n"
        "≥0.7 きっと ／ 0.4〜0.7 たぶん ／ <0.4 わからない",
        VIOLET, fs=8.1, dash=True)

    # ---- 記憶 3: 経験の器
    title(ax, 12.3, 7.75, "記憶 3  経験の器（判断の記憶、I7〜I9）", AQUA)
    box(ax, 12.3, 5.2, 3.4, 2.45,
        "光の器 1 台、口 = 結論の種類\n（本物・例外あり・見かけ・不明・\n分ける・分けない）\n\n"
        "経験 1 つ = 記号の組\n{観点:色, 支持:多い, 反例:多い,\n共に変わる:あり, …, 結論:見かけ}",
        AQUA, fs=8.4)
    box(ax, 12.3, 3.55, 3.4, 1.35,
        "書く: 世界 A で判断するたびに 1 つ足す\n"
        "結論を付けるのは 規則（I7・I8）か\n答えの採点（I9）。口ごとに 1 種類（I8b）",
        AQUA, fs=8.1, dash=True)
    box(ax, 12.3, 2.1, 3.4, 1.15,
        "読む: いまの場面の記号を 1 回通し、\n口ごとの起こりやすさで重ねて、\nいちばん明るい結論を選ぶ",
        AQUA, fs=8.1, dash=True)

    # 語板 → 判断 → 経験
    box(ax, 8.0, 0.35, 3.6, 1.35,
        "判断（原因の比べ）\n同じ語の語板どうし・口どうし・1 枚の中で\n観点ごとの差を比べる（色違いのリンゴ）",
        YELLOW, fs=8.2)
    arrow(ax, 7.3, 2.6, 8.6, 1.72, YELLOW)
    ax.text(7.65, 1.85, "語板の組を\n比べる", ha="center", fontsize=7.5, color=INK2)
    arrow(ax, 11.62, 1.4, 12.28, 4.0, YELLOW)
    ax.text(11.95, 4.55, "場面 + 結論\nを経験に", ha="center", fontsize=7.5, color=INK2, va="bottom")
    arrow(ax, 13.6, 2.08, 11.62, 0.75, AQUA, rad=-0.25)
    ax.text(13.9, 0.95, "世界 B では\n判断を光 1 回に", ha="center", fontsize=7.5, color=INK2)

    # 答え
    box(ax, 0.3, 0.35, 6.9, 1.35,
        "答える: 「赤いリンゴを緑に塗ったら渋くなる？」\n"
        "動詞の観点（塗る = 色）を変えた姿 → 判断で残った原因（状態）の効き目 → いいえ\n"
        "「シナノは甘い？」（知らない品種）→ 親の語板から確率つきで「たぶん 甘い」",
        MUTED, fs=8.4)
    arrow(ax, 7.98, 0.75, 7.22, 0.75, YELLOW)

    out_path = os.path.join(out, "mech_4_memory.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def fig_life(out):
    """図 5: 記憶がいつ書かれ、いつ固定されるか（時間の流れ）。"""
    W, H = 16, 5.6
    
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(0.2, 5.2, "図 5  記憶の時間の流れ: 世界 A で語板と経験を作り、世界 B では語板だけ新しく学んで経験を思い出す",
            fontsize=13, color=INK, fontweight="bold")
    ax.add_patch(FancyArrowPatch((0.4, 4.45), (15.6, 4.45), arrowstyle="-|>", mutation_scale=16, color=INK2, lw=1.6))
    for x, t in [(1.0, "世界 A の文を読む"), (5.3, "世界 A で判断・採点"), (9.4, "世界 B の文を読む（少ない）"), (13.2, "世界 B の問いに答える")]:
        ax.text(x, 4.6, t, fontsize=9.5, color=INK2)
    box(ax, 0.4, 2.7, 3.9, 1.4, "語板 A を学ぶ・分ける\n（文ごとに書き足す）", BLUE, fs=9)
    box(ax, 4.7, 2.7, 3.9, 1.4, "経験を集める（約 375 個 / seed）\n→ 経験の器をまとめて学ぶ（ここで固定）", AQUA, fs=9)
    box(ax, 8.9, 2.7, 3.4, 1.4, "語板 B を新しく学ぶ\n（語板 A は使わない）", BLUE, fs=9)
    box(ax, 12.6, 2.7, 3.0, 1.4, "判断を経験の器に\n1 回通して答える\n（器は書き換えない）", AQUA, fs=9)
    arrow(ax, 6.65, 2.68, 14.1, 2.68, AQUA, rad=0.08)
    ax.text(10.6, 1.95, "経験の器だけ持ち越す", ha="center", fontsize=8.5, color=INK2)
    ax.text(0.4, 1.55, "いまの限界（次の実験の候補）", fontsize=10, color=ORANGE, fontweight="bold")
    ax.text(0.4, 1.4,
            "・経験の器は世界 A の後にまとめて学び、その後は増えない（持ち主の「失敗や反映で増える」はまだ。口を足して育てる形は未実装）。\n"
            "・結論は 1 つだけ選ぶ（いちばん明るい口）。結論の分布・確率で持つのは I11 の予定。\n"
            "・語板は増えるだけで、まとめ直し・忘れることはない。世界 A と B の語板は別々で、同じ語が出ても引き継がない。\n"
            "・口の割り当て（結論の種類 = 口）と、場面の記号への丸めはデジタル。光の中で口が自然に分かれたわけではない（I8 の mix では混ざった）。\n"
            "・親子の向きは「X の Y」の手の規則で決める。親の語板は和を取るだけなので、知っている子が少ないと言い切りすぎる（I10 の較正）。",
            fontsize=8.8, color=INK, linespacing=1.6, va="top")
    out_path = os.path.join(out, "mech_5_memory_time.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PRIV, "images"), help="書き出す先（非公開のフォルダ）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(fig_memory(a.out))
    print(fig_life(a.out))


if __name__ == "__main__":
    main()
