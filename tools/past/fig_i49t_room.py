"""I49t の図: 共振の揺らぎの余地（上位に正解が入る割合と、一位と二位の近さ）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49t_room.py 出力.png
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

RANKS = [1, 2, 3, 5]
IN = {
    "白紙から": {"関係の表": [.4321, .5391, .5984, .6704], "解く・属性": [.4084, .5900, .6785, .7556]},
    "青空の 3 つ組を先に": {"関係の表": [.4533, .5645, .6292, .7091], "解く・属性": [.3983, .5963, .6987, .8150]},
}
GAP = {
    "白紙から": {"関係の表": [.0925, .0275, .0274, .0362, .0429, .0656, .0472, .0381, .0325, .5902],
             "解く・属性": [.4007, .1930, .1108, .0708, .0496, .0344, .0304, .0340, .0297, .0467]},
    "青空の 3 つ組を先に": {"関係の表": [.0848, .0308, .0307, .0408, .0429, .0674, .0459, .0399, .0434, .5735],
                    "解く・属性": [.5026, .2243, .1164, .0641, .0417, .0198, .0133, .0096, .0049, .0033]},
}
COL = {"関係の表": "#4c72b0", "解く・属性": "#dd8452"}


def main(out):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
    for j, start in enumerate(IN):
        a = ax[0][j]
        for name, ys in IN[start].items():
            a.plot(RANKS, ys, marker="o", color=COL[name], label=name)
            for x, y in zip(RANKS, ys):
                up = 9 if (name == "解く・属性") == (x > 1) else -15
                a.annotate(f"{y:.2f}"[1:], (x, y), textcoords="offset points", xytext=(0, up),
                           ha="center", fontsize=9, color=COL[name])
        a.set_xticks(RANKS)
        a.set_xticklabels([f"{r} 位まで" for r in RANKS])
        a.set_ylim(.3, .9)
        a.set_ylabel("正解が入っている割合")
        a.set_title(f"{start}: 上位に正解が入る割合")
        a.grid(alpha=.3)
        a.legend(loc="lower right")
        b = ax[1][j]
        xs = range(10)
        w = .4
        for k, (name, ys) in enumerate(GAP[start].items()):
            b.bar([x + (k - .5) * w for x in xs], ys, w, color=COL[name], label=name)
        b.set_xticks(list(xs))
        b.set_xticklabels([f"{i * 10}" for i in xs])
        b.set_xlabel("一位と二位の点の開き（一位の点に対する %、左ほど並んでいる）")
        b.set_ylabel("問いの割合")
        b.set_ylim(0, .65)
        b.set_title(f"{start}: 一位と二位の近さ")
        b.grid(axis="y", alpha=.3)
        b.legend(loc="upper center")
    fig.suptitle("共振の揺らぎの余地: 解く形は一位では負けるが、二位までで逆転し、候補が並んでいる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49t_room.png")
