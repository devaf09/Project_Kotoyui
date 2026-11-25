"""I49zb の図: いいとこ取りの形の、正しい物と、正しいと言える物を合わせた割合。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zb_mix.py 出力.png
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

NAMES = ["解く・属性だけ", "解く＋出た語→板（表なし）", "関係の表だけ", "表 → 解く", "表 → 解く＋板",
         "表 → 解く＋出た語（九割以内）", "表 → 解く＋出た語", "表 → 解く＋出た語→板"]
# (正しい物, 合わせて)
D = {
    "白紙から": [(.4084, .5421), (.4303, .5512), (.4321, .5893), (.4454, .6026), (.4437, .6007), (.4477, .6061), (.4491, .6082), (.4506, .6089)],
    "青空の 3 つ組を先に": [(.3983, .5155), (.4237, .5368), (.4533, .6148), (.4651, .6278), (.4601, .6219), (.4654, .6283), (.4659, .6284), (.4649, .6269)],
}


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    ys = list(range(len(NAMES)))[::-1]
    for j, start in enumerate(D):
        a = ax[j]
        rows = D[start]
        col = ["#dd8452", "#dd8452", "#4c72b0"] + ["#55a868"] * 5
        light = ["#f2c9ae", "#f2c9ae", "#a9bfe0"] + ["#b5dcc0"] * 5
        a.barh(ys, [r[1] for r in rows], .7, color=light)
        a.barh(ys, [r[0] for r in rows], .7, color=col)
        for y, r in zip(ys, rows):
            a.text(r[0] - .005, y, f"{r[0]:.3f}"[1:], va="center", ha="right", color="white", fontsize=9)
            a.text(r[1] + .004, y, f"{r[1]:.3f}"[1:], va="center", fontsize=9)
        a.axvline(rows[2][0], color="#4c72b0", ls=":", lw=1.2)
        a.set_yticks(ys)
        a.set_yticklabels(NAMES)
        a.set_xlim(.3, .68)
        a.set_xlabel("濃い色 = 正しい物、薄い色まで = 正しいと言える物も合わせて（点線は表だけ）")
        a.set_title(start)
        a.grid(axis="x", alpha=.3)
    fig.suptitle("いいとこ取りの形: 表が強い問いは表、ほかは波で解く形。初めて表だけを上回った", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zb_mix.png")
