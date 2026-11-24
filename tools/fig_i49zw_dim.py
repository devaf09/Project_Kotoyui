"""I49zw の図（2 枚目）: 広げる本数だけを 8,192 から 32,768 に変えた時（448 話・4,464 組）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49zw_dim.py 出力.png
"""
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

FORMS = [("組だけ・広げる", [.2816, .2861, .2801, .2811], [.2856, .2866, .2838, .2836]),
         ("解く形＋組・広げる", [.3465, .3507, .3485, .3488], [.3485, .3516, .3493, .3501]),
         ("数えて選ぶ・組と", [.3934, .3952, .3927, .3936], [.3932, .3959, .3929, .3949])]
BASE = {1: ("解く形 .342", .3423), 2: ("数えて選ぶ .390", .3902)}


def main(out):
    fig, a = plt.subplots(figsize=(10, 5.4))
    w = .36
    for i, (name, lo, hi) in enumerate(FORMS):
        for k, (ys, col, lab) in enumerate(((lo, "#f2c6a8", "8,192 本"), (hi, "#dd8452", "32,768 本"))):
            x = i + (k - .5) * w
            a.bar(x, sum(ys) / 4, w * .94, color=col, label=lab if i == 0 else None)
            a.plot([x] * 4, ys, ".", color="k", ms=3)
            a.text(x, max(ys) + .004, ("%.3f" % (sum(ys) / 4))[1:], ha="center", fontsize=10)
        if i in BASE:
            a.plot([i - .45, i + .45], [BASE[i][1]] * 2, "--", color="#555555", lw=1)
            a.text(i + .46, BASE[i][1], "組を足す前\n" + BASE[i][0], va="center", fontsize=8.5)
    a.set_xticks(range(3))
    a.set_xticklabels([f[0] for f in FORMS])
    a.set_xlim(-.6, 2.95)
    a.set_ylim(.25, .43)
    a.set_ylabel("正しい物（棒は 4 通りの並べ方の平均、点は並べ方ごと）")
    a.set_title("本数を 4 倍にしても、ほぼ変わらない（+0.1〜0.3 ポイント）。\n違う組の重なりは、上積みが小さい事の主な理由ではない", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zw_dim.png")
