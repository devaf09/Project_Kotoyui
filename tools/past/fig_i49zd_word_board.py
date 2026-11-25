"""I49zd の図: 仮の板に語そのものの波を持たせた時の、正しい物の割合（属性の板、出た語と比べる）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zd_word_board.py 出力.png
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = os.environ.get("KOTOYUI_FONT", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

X = ["1 組で半分", "2 組", "4 組", "薄れない\n（話ごとに空）"]
# 始め方: (板無し, 出た語・上位 3, 語の波の板の 4 版, 属性の板・九割以内・席ごと（I49z）)
D = {
    "白紙から": (.4084, .4307, {
        "九割以内・1 枚の板": [.4112, .4047, .3982, .4061],
        "九割以内・席ごとの板": [.4169, .4100, .4084, .4090],
        "上位 3・1 枚の板": [.3465, .3274, .3053, .3493],
        "上位 3・席ごとの板": [.3819, .3646, .3555, .3805]},
        [.4168, .4100, .4032, .4043]),
    "青空の 3 つ組を先に": (.3983, .4242, {
        "九割以内・1 枚の板": [.4004, .3931, .3857, .3923],
        "九割以内・席ごとの板": [.4076, .3970, .3942, .3987],
        "上位 3・1 枚の板": [.3506, .3359, .3160, .3620],
        "上位 3・席ごとの板": [.3757, .3596, .3530, .3784]},
        [.4117, .4038, .3982, .3986]),
}
STY = {"九割以内・1 枚の板": ("#4c72b0", "-", "o"), "九割以内・席ごとの板": ("#dd8452", "-", "s"),
       "上位 3・1 枚の板": ("#4c72b0", ":", "o"), "上位 3・席ごとの板": ("#dd8452", ":", "s")}
# 席ごとの、板無しからの差（ポイント）。九割以内・席ごと・1 組で半分の版と、出た語・上位 3
SEAT = {
    "白紙から": {"属性の板": [2.3, -0.6, 0.8], "語の波の板": [2.9, 0.5, -0.9], "出た語・上位 3": [4.9, 0.7, 1.1]},
    "青空の 3 つ組を先に": {"属性の板": [5.1, -1.1, 0.0], "語の波の板": [5.8, -0.6, -2.4], "出た語・上位 3": [7.5, 0.1, 0.2]},
}
COL = {"属性の板": "#8c8c8c", "語の波の板": "#dd8452", "出た語・上位 3": "#55a868"}


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.1]))
    for j, start in enumerate(D):
        a = ax[j]
        base, said, rows, attr = D[start]
        a.axhline(said, color="#55a868", lw=1.5, ls="-.", label="出た語・上位 3（決まりで書いた形） " + f"{said:.3f}"[1:])
        a.axhline(base, color="k", lw=1, ls="--", label="板無し " + f"{base:.3f}"[1:])
        a.plot(range(3), attr[:3], color="#8c8c8c", lw=1, marker="^", label="属性の板・九割以内・席ごと（前の試し）")
        a.plot([3], attr[3:], color="#8c8c8c", marker="^", ls="none", mfc="white")
        for name, ys in rows.items():
            c, ls, m = STY[name]
            a.plot(range(3), ys[:3], color=c, ls=ls, marker=m, label="語の波の板・" + name)
            a.plot([3], ys[3:], color=c, marker=m, ls="none", mfc="white")
        a.set_xticks(range(4))
        a.set_xticklabels(X, fontsize=9)
        a.set_ylim(.24, .44)
        a.set_ylabel("正しい物の割合")
        a.set_title(f"{start}: 出た語の線には届かない")
        a.grid(alpha=.3)
        a.legend(fontsize=7.5, loc="lower left")
    a = ax[2]
    seats = ["だれが", "何を", "どうした"]
    w = .13
    k = 0
    for start, rows in SEAT.items():
        for name, v in rows.items():
            xs = [i + (k - 2.5) * w for i in range(3)]
            a.bar(xs, v, w * .92, color=COL[name], hatch="//" if start != "白紙から" else None, edgecolor="white",
                  label=f"{name}（{start}）")
            for x, p in zip(xs, v):
                a.text(x, p + (.15 if p >= 0 else -.15), f"{p:+.1f}", ha="center", va="bottom" if p >= 0 else "top", fontsize=7)
            k += 1
    a.axhline(0, color="k", lw=.8)
    a.set_xticks(range(3))
    a.set_xticklabels(seats)
    a.set_ylim(-3.5, 10.5)
    a.set_ylabel("板無しからの差（ポイント）")
    a.set_title("席ごと: 語の波の板は「どうした」を下げる")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("仮の板に語そのものの波を持たせる: 属性の板とほぼ同じ高さで、決まりで書いた「出た語」の代わりにはならなかった", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zd_word_board.png")
