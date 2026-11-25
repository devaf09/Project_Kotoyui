"""I49zp の図: 「A が B を C する」のテキスト（切り出し無し）で数え直した結果と、材料ごとの自然さの点。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zp_text_talks.py 出力.png
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

FORMS = ["関係の表", "属性だけ", "属性＋固有の位相", "＋振幅", "数えて選ぶ"]
ORDERS = ["元の順", "並べ替え 1", "並べ替え 2", "並べ替え 3"]
ALL = [[.2761, .2994, .2855, .2978, .3185], [.2761, .2921, .2855, .2989, .3126], [.2759, .2952, .2789, .2963, .3143], [.2780, .3001, .2813, .3013, .3211]]
TAIL = [[.2650, .3039, .2968, .3192, .3216], [.3486, .3628, .3651, .3840, .3887], [.3369, .3722, .3369, .3651, .3757], [.3804, .3734, .3722, .4028, .4287]]
# 並べ替え 3、段ごと（表の当たり、解く形・属性だけの当たり、問い）
LEVELS = [("聞いた組\nそのもの", 134, 124, 312), ("片方だけ\n借りる", 418, 357, 708), ("述語側", 145, 125, 202), ("もう片方", 177, 249, 1368), ("借りる", 306, 321, 1064), ("答えられ\nない", 0, 98, 591)]
LOOK = [("青空の 3 つ組\n（切り出し）", 1.47), ("自作の話\n（切り出し）", 2.74), ("AI が書く\n1 回目", 3.15), ("AI が書く\n2 回目・12B", 3.65), ("AI が書く\n2 回目・31B", 3.90), ("いただいた\nテキスト", 4.35)]
COL = ["#4c72b0", "#8c8c8c", "#9ecae1", "#dd8452", "#55a868"]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    a = ax[0]
    w = .16
    for k, name in enumerate(FORMS):
        xs = [i + (k - 2) * w for i in range(4)]
        a.bar(xs, [ALL[i][k] for i in range(4)], w * .94, color=COL[k], label=name)
        a.plot(xs, [TAIL[i][k] for i in range(4)], "v", color="k", ms=4, label="後ろ 2 割の問いだけ" if k == 0 else None)
    a.set_xticks(range(4))
    a.set_xticklabels(ORDERS)
    a.set_ylim(.2, .45)
    a.set_ylabel("正しい物（隠した語を当てた割合）")
    a.set_title("どの並べ方でも 数えて選ぶ が上。解く形（属性だけ）が表を上回る")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, ncol=2, loc="upper left")
    a = ax[1]
    w = .38
    for i, (name, t, s, n) in enumerate(LEVELS):
        a.bar(i - w / 2, t / n, w * .94, color=COL[0], label="関係の表" if i == 0 else None)
        a.bar(i + w / 2, s / n, w * .94, color=COL[1], label="解く形・属性だけ" if i == 0 else None)
        a.text(i, max(t, s) / n + .02, "%d 問" % n, ha="center", fontsize=9)
    a.axvline(2.5, color="k", lw=.8, ls=":")
    a.text(1, .78, "聞いた事の有る組\n1,222 問（29%）", ha="center", fontsize=10)
    a.text(4, .78, "聞いた事の無い組\n3,023 問（71%）", ha="center", fontsize=10)
    a.set_xticks(range(len(LEVELS)))
    a.set_xticklabels([l[0] for l in LEVELS], fontsize=9)
    a.set_ylim(0, .9)
    a.set_ylabel("当たった割合（並べ替え 3）")
    a.set_title("この材料は、聞いた事の無い組の問いが 7 割。そこは解く形の方が強い")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="center right")
    a = ax[2]
    cols = ["#b0b0b0", "#b0b0b0", "#9ecae1", "#9ecae1", "#9ecae1", "#55a868"]
    a.bar(range(len(LOOK)), [v for _, v in LOOK], .7, color=cols)
    for i, (_, v) in enumerate(LOOK):
        a.text(i, v + .06, "%.2f" % v, ha="center", fontsize=10)
    a.set_xticks(range(len(LOOK)))
    a.set_xticklabels([n for n, _ in LOOK], fontsize=8)
    a.set_ylim(1, 5)
    a.set_ylabel("ローカル AI が付けた自然さ（1〜5）")
    a.set_title("材料の 3 つ組の自然さ")
    a.grid(axis="y", alpha=.3)
    fig.suptitle("切り出しを通さない材料（194 話、3 つ組 1,415 個）で数え直す: 3 つ組は自然になり、形の順位が変わった", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zp_text_talks.png")
