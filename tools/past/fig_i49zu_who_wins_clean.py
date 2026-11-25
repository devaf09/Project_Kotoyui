"""I49zu の図: 「正解が上位に居るのに、属性の似た語に負ける」の数えを、古い材料ときれいな材料で並べる。

数は PC で数えた集計値をそのまま置いている。古い材料は I49zn の白紙からの数。きれいな材料は 448 話・13,392 問、2 通りの並べ方の平均。
使い方: python tools/past/fig_i49zu_who_wins_clean.py 出力.png
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

OLD_C, NEW_C = "#b07aa1", "#4c72b0"
FORMS = ["属性だけ", "属性＋固有の位相", "＋振幅"]
LOST_OLD = [.270, .263, .256]
LOST_NEW = [(.2551 + .2525) / 2, (.2501 + .2501) / 2, (.2508 + .2370) / 2]
# 属性だけ・元の順、席ごと（一位と正解の近さ, 正解とほかの候補の近さ, 負けた問い）
NEAR = [("全体", .3709, .1213, 3416), ("だれが", .4963, .0903, 1581), ("何を", .2352, .1057, 835), ("どうした", .3292, .1743, 1000)]
# 属性だけ・元の順、負けた問いで（一位がこの話でもう出た, 正解がこの話でもう出た）
SAID = [("全体", .683, .4789), ("だれが", .7369, .7875), ("何を", .6814, .3246), ("どうした", .599, .12)]


def bars(a, labels, xs, ys, lx, ly, cx, cy):
    w = .38
    for i, (x, y) in enumerate(zip(xs, ys)):
        a.bar(i - w / 2, x, w * .94, color=cx, label=lx if i == 0 else None)
        a.bar(i + w / 2, y, w * .94, color=cy, label=ly if i == 0 else None)
        a.text(i - w / 2, x + .008, ("%.3f" % x)[1:], ha="center", fontsize=10)
        a.text(i + w / 2, y + .008, ("%.3f" % y)[1:], ha="center", fontsize=10)
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper right")


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
    bars(ax[0], FORMS, LOST_OLD, LOST_NEW, "古い材料（I49zn）", "きれいな材料", OLD_C, NEW_C)
    ax[0].set_ylim(0, .36)
    ax[0].set_ylabel("正解が 2〜3 位で負けた問いの割合")
    ax[0].set_title("負ける問いの割合は、材料を替えてもほぼ同じ（25% 前後）", fontsize=11)
    bars(ax[1], ["%s\n%s 問" % (n[0], format(n[3], ",")) for n in NEAR], [n[1] for n in NEAR], [n[2] for n in NEAR],
         "1 位の語と正解の近さ", "正解とほかの候補の近さ", "#dd8452", "#8c8c8c")
    ax[1].set_ylim(0, .62)
    ax[1].set_ylabel("属性の波の近さ（きれいな材料、属性だけ、元の順）")
    ax[1].set_title("1 位を取るのは、正解に属性が似た語（古い材料では .37 対 .13）", fontsize=11)
    bars(ax[2], [s[0] for s in SAID], [s[1] for s in SAID], [s[2] for s in SAID],
         "1 位の語が、この話でもう出た", "正解が、この話でもう出た", "#dd8452", "#55a868")
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel("負けた問いの中での割合（同じ条件）")
    ax[2].set_title("何を・どうした では、もう出た語が、まだ出ていない正解に勝っている", fontsize=11)
    fig.suptitle("「属性が近い語に負ける」は、きれいな材料でも変わらない（材料のせいではなかった）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zu_who_wins_clean.png")
