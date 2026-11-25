"""I49zo の図: 答えが違った問いで、ローカル AI が付けた自然さの点（古い答え・新しい答え・正解）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zo_look.py 出力.png
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

GROUPS = ["全体", "新しい方が\n当たり", "古い方が\n当たり", "どちらも\n外れ"]
# (古い答え、新しい答え、正解) の平均と、問いの数
DATA = {"白紙から": ([(2.375, 2.36, 2.8), (2.618, 2.99, 2.99), (2.812, 2.287, 2.812), (2.025, 2.071, 2.695)], [400, 102, 101, 197]),
        "青空を先に": ([(2.3, 2.243, 2.683), (2.198, 2.629, 2.629), (3.023, 2.36, 3.023), (2.045, 1.965, 2.566)], [400, 116, 86, 198])}
WIN = {"白紙から": (94, 99, 207), "青空を先に": (77, 93, 230)}
NAMES = ["属性だけ の答え", "属性＋固有の位相 の答え", "正解"]
COL = ["#8c8c8c", "#9ecae1", "#55a868"]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1.4, 1.4, 1]))
    w = .27
    for a, start in zip(ax, DATA):
        vals, ns = DATA[start]
        for k in range(3):
            xs = [i + (k - 1) * w for i in range(len(GROUPS))]
            ys = [v[k] for v in vals]
            a.bar(xs, ys, w * .94, color=COL[k], label=NAMES[k])
            for x, y in zip(xs, ys):
                a.text(x, y + .04, "%.2f" % y, ha="center", fontsize=8)
        a.axhline(3, color="k", lw=.8, ls=":")
        a.text(3.45, 3.03, "3 = どちらとも言えない", ha="right", fontsize=8)
        a.set_xticks(range(len(GROUPS)))
        a.set_xticklabels(["%s\n（%d 問）" % (g, n) for g, n in zip(GROUPS, ns)], fontsize=9)
        a.set_ylim(1, 4.2)
        a.set_ylabel("自然さの点の平均（1〜5）")
        a.set_title(start + ": 外れどうしでは差が無い。正解でも 3 に届かない")
        a.grid(axis="y", alpha=.3)
        a.legend(fontsize=9, loc="upper left")
    a = ax[2]
    labs = ["新しい方が自然", "古い方が自然", "同じ点"]
    for k, start in enumerate(WIN):
        xs = [i + (k - .5) * .38 for i in range(3)]
        a.bar(xs, WIN[start], .36, color=["#dd8452", "#4c72b0"][k], label=start)
        for x, y in zip(xs, WIN[start]):
            a.text(x, y + 3, str(y), ha="center", fontsize=9)
    a.set_xticks(range(3))
    a.set_xticklabels(labs)
    a.set_ylim(0, 270)
    a.set_ylabel("問いの数（400 問中）")
    a.set_title("どちらの答えが自然か: 並び（z −0.4 / −1.2）")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9)
    fig.suptitle("ローカル AI に違和感を見てもらう: 固有の位相を足しても答えの自然さは変わらない。正解の組そのものが低い点（問いの材料が崩れている物が有る）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zo_look.png")
