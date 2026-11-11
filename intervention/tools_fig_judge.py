"""気付き・介入を判断する板の図（非公開の置き場の images/mech_6_*.png 〜）: I19〜I23

  図 6  判断の板（I20）: 答えが違ったら一手ずつ替えて解き直し、道が分かれた一手を覚える
  図 7  気付き（I21）: 同じ板の左と右の結果の差分から「入れ替えても同じ」を見つける。− は ＋ の逆回し
  図 8  桁上りの照射（I22）: 条件分岐なし。いつも桁上りの板（0 か 1）を十の位に足す
  図 9  算術の流れ（I19・I23）: 字のアレイ → まとめる → 縮める → バッファ板（一般常識の板 + ネスト）→ 復元
実行: python intervention/tools_fig_judge.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools_fig_mechanism as F  # noqa: E402
from tools_fig_mechanism import AQUA, BLUE, INK, INK2, MUTED, ORANGE, VIOLET, YELLOW, arrow, box, plt, setup  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "IPAGothic", "sans-serif"]
OUT = os.path.join(F.PRIV, "images")


def title(ax, h, s):
    ax.text(0.2, h - 0.4, s, fontsize=12, color=INK, fontweight="bold")


def fig_judge():
    fig, ax = setup(13, 7)
    title(ax, 7, "図 6  判断の板（I20）: 総当たりの代わりに、答えが違ったときだけ「道が分かれた一手」を探して覚える")
    box(ax, 0.3, 4.9, 2.2, 0.9, "式のアレイ\n2 × 3 ＋ 4 − 1", MUTED)
    box(ax, 3.1, 4.9, 2.6, 0.9, "文法解析グレイン\n（命令デコーダ）で解く", BLUE, bold=True)
    box(ax, 6.3, 4.9, 1.8, 0.9, "答え\n（最後の 1 枚）", MUTED)
    box(ax, 8.7, 4.9, 1.6, 0.9, "合った？", YELLOW, bold=True)
    arrow(ax, 2.5, 5.35, 3.1, 5.35)
    arrow(ax, 5.7, 5.35, 6.3, 5.35)
    arrow(ax, 8.1, 5.35, 8.7, 5.35)
    box(ax, 10.9, 4.9, 1.9, 0.9, "道の一手ずつを\n「縮める」に足す", AQUA)
    arrow(ax, 10.3, 5.35, 10.9, 5.35, text="はい")
    # いいえ
    box(ax, 7.6, 2.6, 3.4, 1.3, "判断の板（介入）\n自分の道を前から一手ずつ、\nその一手だけ別の手に替えて解き直す", ORANGE, bold=True)
    arrow(ax, 9.3, 4.9, 9.3, 3.9)
    ax.text(9.45, 4.3, "いいえ", fontsize=8, color=INK2)
    box(ax, 3.6, 2.6, 3.4, 1.3, "答えが合った所 = 道が分かれた一手\n例: 2 × 3 ＋ 4 − 1 で\n「4 − 1 を先にした」", ORANGE)
    arrow(ax, 7.6, 3.25, 7.0, 3.25)
    box(ax, 0.3, 2.6, 2.7, 1.3, "注意つきの経験\n選んだ手 → 待つ\n替えた手 → 縮める", VIOLET)
    arrow(ax, 3.6, 3.25, 3.0, 3.25)
    box(ax, 0.3, 0.5, 12.5, 1.3, "経験の器（口: 縮める / 待つ）に融合 → 次の式からは、前後 2 つの板の種類を照らすだけで「ここを縮める」が光る\n"
        "600 式のうち、解き直しが要ったのは 9 式。数 6 個の式で 0.99（I19 の総当たりは 0.96〜0.97、しかも長い式ほど倍々に重い）", MUTED, fs=9.5)
    arrow(ax, 1.6, 2.6, 1.6, 1.8)
    arrow(ax, 11.8, 4.9, 11.8, 1.8)
    ax.add_patch(F.FancyArrowPatch((3.3, 1.8), (3.3, 4.9), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.0, ls="--"))
    ax.text(3.4, 4.3, "次の式で使う", fontsize=8, color=INK2)
    return fig


def fig_notice():
    fig, ax = setup(13, 6.6)
    title(ax, 6.6, "図 7  気付き（I21）: 同じ板の「左に来たとき」と「右に来たとき」の結果の差分で、入れ替えても同じかを見つける")
    for k, (op, l, r, same, col, agree) in enumerate((("＋", "4 ＋ 3 = 7", "3 ＋ 4 = 7", True, AQUA, "左右の一致 1.00"), ("×", "4 × 3 = 12", "3 × 4 = 12", True, AQUA, "左右の一致 1.00"),
                                                       ("−", "4 − 3 = 1", "3 − 4 = −1", False, ORANGE, "左右の一致 0.05〜0.15（3 − 3 = 0 のような偶然）"))):
        y = 4.6 - k * 1.55
        box(ax, 0.3, y, 1.6, 1.0, f"4 の板\n演算の帯（{op}）", BLUE)
        box(ax, 2.5, y + 0.52, 2.6, 0.48, f"（左, {op}, 3）→ {l.split('= ')[1]}", MUTED, fs=8.5)
        box(ax, 2.5, y, 2.6, 0.48, f"（右, {op}, 3）→ {r.split('= ')[1]}", MUTED, fs=8.5)
        arrow(ax, 1.9, y + 0.5, 2.5, y + 0.76)
        arrow(ax, 1.9, y + 0.5, 2.5, y + 0.24)
        box(ax, 5.7, y + 0.1, 1.7, 0.8, "差分\n（左 − 右）", YELLOW)
        arrow(ax, 5.1, y + 0.5, 5.7, y + 0.5)
        box(ax, 8.0, y + 0.1, 2.3, 0.8, "0 → 入れ替えても同じ" if same else "0 でない → 入れ替えない", col, bold=True, fs=9)
        arrow(ax, 7.4, y + 0.5, 8.0, y + 0.5)
        ax.text(5.7, y - 0.2, agree, fontsize=8.5, color=INK2)
    box(ax, 10.8, 3.0, 2.0, 2.6, "気付くと、教えて\nいない組も照らせる\n\n表の半分を教えて\n0.52 → 0.75", AQUA, fs=9)
    box(ax, 10.8, 0.25, 2.0, 2.0, "− は教えなくても\n＋ の逆回し\n「z ＋ 3 = 4 の z」\nで 0.75", VIOLET, fs=9)
    return fig


def fig_carry():
    fig, ax = setup(13, 6.6)
    title(ax, 6.6, "図 8  桁上りの照射（I22）: 「繰り上がったら足す」という判断を立てず、いつも桁上りの板（暗ければ 0）を足す")
    steps = [("①", "7 の板に（5, ＋）", "2 と 桁上り 1", BLUE), ("②", "4 の板に（3, ＋）", "7 と 桁上り 0", BLUE),
             ("③", "7 の板に（桁上り 1, ＋）", "8 と 桁上り 0", ORANGE), ("④", "桁上り 0 の板に（0, ＋）", "百の位 0", ORANGE)]
    box(ax, 0.3, 4.45, 2.2, 1.3, "47 ＋ 35\n十の位 4・一の位 7\n十の位 3・一の位 5", MUTED)
    for k, (n, a, b, col) in enumerate(steps):
        y = 4.75 - k * 1.1
        box(ax, 3.1, y, 3.3, 0.8, f"{n} {a}", col, fs=9.5)
        box(ax, 7.0, y, 2.4, 0.8, f"光る出口: {b}", MUTED, fs=9.5)
        arrow(ax, 6.4, y + 0.4, 7.0, y + 0.4)
    arrow(ax, 2.5, 5.1, 3.1, 5.15)
    pts = lambda y0, y1, xr: ([9.4, xr, xr, 2.85, 2.85, 3.1], [y0 + 0.4, y0 + 0.4, y0 - 0.15, y0 - 0.15, y1 + 0.4, y1 + 0.4])
    for y0, y1, xr in ((4.75, 2.55, 9.75), (3.65, 1.45, 9.95)):
        xs_, ys_ = pts(y0, y1, xr)
        ax.plot(xs_[:-1], ys_[:-1], color=ORANGE, lw=1.1)
        ax.add_patch(F.FancyArrowPatch((2.85, ys_[-1]), (xs_[-1], ys_[-1]), arrowstyle="-|>", mutation_scale=12, color=ORANGE, lw=1.1))
    ax.text(10.05, 4.3, "桁上りの板を\n次の照射の入力に", fontsize=8.5, color=ORANGE)
    box(ax, 10.2, 1.6, 2.6, 1.0, "答え 82\n（一般常識の 82 の板）", AQUA, bold=True)
    arrow(ax, 9.4, 1.85, 10.2, 2.0)
    ax.text(0.3, 0.75, "桁上りが 0 なら ③ は「＋0」で何も変えない → 条件分岐が要らない。判断は「足すかどうか」ではなく「何を足すか（明るさ）」。\n"
            "0〜9 の板の表 300 文だけで、0〜99 の ＋ − と 2 桁 × 1 桁の × が 1.00（数そのものの表に同じ 300 文では 0.02）。その代わり「x ＋ 0」「x ＋ 1」の文が要になる。",
            fontsize=9, color=INK2, linespacing=1.6)
    return fig


def fig_flow():
    fig, ax = setup(13, 6.8)
    title(ax, 6.8, "図 9  算術の流れ（I19・I23）: 字の板のアレイを縮め、答えのバッファ板からネストで式を戻す")
    xs = [0.3, 3.0, 5.9, 9.2]
    box(ax, xs[0], 4.6, 2.2, 1.0, "字のアレイ\n4 7 ＋ 3 5", MUTED)
    box(ax, xs[1], 4.6, 2.4, 1.0, "まとめる・読む\n→ [47] ＋ [35]", BLUE)
    box(ax, xs[2], 4.6, 2.8, 1.0, "縮める（演算の帯の照射）\n→ [82]", BLUE)
    box(ax, xs[3], 4.4, 3.5, 1.4, "バッファ板 82\n値の帯: 一般常識の 82 の板の明るさ\n参照の帯（ネスト）: 47・＋・35", AQUA, bold=True, fs=9)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        arrow(ax, xs[a] + (2.2, 2.4, 2.8)[a], 5.1, xs[b], 5.1)
    box(ax, 3.0, 2.3, 5.7, 1.3, "文法解析グレイン（命令デコーダ）\n前後 2 つの板の種類 + 命令の名前を照らし、命令ごとの器で「縮める / 待つ」\n答えだけから学ぶ（図 6 の判断の板）。実行できない手は断られたら覚える", ORANGE, fs=9)
    arrow(ax, 4.2, 3.6, 4.2, 4.6, col=ORANGE)
    arrow(ax, 7.2, 3.6, 7.2, 4.6, col=ORANGE)
    box(ax, 9.2, 2.3, 3.5, 1.3, "復元: 参照の帯をたどる\n82 → 47 ＋ 35（劣化なしで 1.00）", VIOLET, fs=9)
    arrow(ax, 10.95, 4.4, 10.95, 3.6, col=VIOLET)
    box(ax, 9.2, 0.4, 3.5, 1.3, "総合板 z（全バッファ板の和）\n演算の数は読める（1.00）\n式・答えは戻らない（0.16〜0.25）", MUTED, fs=9)
    ax.text(0.3, 0.9, "z は「何が入っているか」の索引、組み方はネストが持つ。\n案 2 の最後の語板（浦島太郎）も、z とネストの両方を持たせる。", fontsize=9, color=INK2, linespacing=1.6)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, f in (("mech_6_judge", fig_judge), ("mech_7_notice", fig_notice), ("mech_8_carry", fig_carry), ("mech_9_arith_flow", fig_flow)):
        fig = f()
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(os.path.join(OUT, name + ".png"))


if __name__ == "__main__":
    main()
