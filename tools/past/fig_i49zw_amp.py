"""I49zw の図（3 枚目）: 広げる時の大きさのそろえ方だけを変えた比べ（448 話・4,464 組、8,192 本）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zw_amp.py 出力.png
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

AMPS = [("属性を 1\n（割合を捨てる）", "#b07aa1"), ("そのまま\n（属性の割合）", "#f2c6a8"), ("位相だけ\n（どの本も同じ大きさ）", "#dd8452")]
DATA = {"組だけ・広げる": ([.2351, .2360, .2394, .2386], [.2816, .2861, .2801, .2811], [.2885, .2891, .2861, .2870], ("z −11.7〜−14.2", "z +1.6〜+3.7")),
        "解く形＋組・広げる": ([.3447, .3457, .3450, .3468], [.3465, .3507, .3485, .3488], [.3507, .3538, .3519, .3504], ("z −1.0〜−2.8", "z +1.2〜+3.1"))}


def main(out):
    fig, a = plt.subplots(figsize=(11, 5.6))
    w = .27
    for i, (name, (one, raw, ph, zs)) in enumerate(DATA.items()):
        for k, ys in enumerate((one, raw, ph)):
            x = i + (k - 1) * w
            a.bar(x, sum(ys) / 4, w * .94, color=AMPS[k][1], label=AMPS[k][0].replace("\n", "") if i == 0 else None)
            a.plot([x] * 4, ys, ".", color="k", ms=3)
            a.text(x, max(ys) + .004, ("%.3f" % (sum(ys) / 4))[1:], ha="center", fontsize=10)
        a.text(i - w, max(one) + .02, zs[0], ha="center", fontsize=8.5, color="#c44e52")
        a.text(i + w, max(ph) + .02, zs[1], ha="center", fontsize=8.5, color="#2a7a3a")
    a.plot([.55, 1.45], [.3423] * 2, "--", color="#555555", lw=1)
    a.text(1.47, .3423, "組を足す前\n解く形 .342", va="center", fontsize=8.5)
    a.set_xticks(range(2))
    a.set_xticklabels(list(DATA))
    a.set_xlim(-.55, 1.85)
    a.set_ylim(.2, .4)
    a.set_ylabel("正しい物（棒は 4 通りの並べ方の平均、点は並べ方ごと）")
    a.set_title("大きさをそろえる（位相だけ）と少し上がる。属性の割合を捨てると下がる。\nz は「そのまま」との直の比べ", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zw_amp.png")
