"""I49v の図: 2 つの席を同時に隠した時の、選び方ごとの「2 席とも当たり」と、上位に 2 席とも正解が入る割合（上限）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49v_two_hidden.py 出力.png
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

NAMES = ["別々", "順に・前から", "順に・後ろから", "掛け算・上位 3", "足し算・上位 3", "掛け算・上位 5", "足し算・上位 5"]
D = {
    "白紙から": {"関係の表": ([.1243, .1399, .1503, .1394, .1301, .1389, .1294], (.2331, .2559)),
             "解く・属性": ([.1299, .1412, .1455, .1484, .1531, .1452, .1507], (.3862, .4924))},
    "青空の 3 つ組を先に": {"関係の表": ([.1274, .1452, .1645, .1517, .1393, .1526, .1370], (.2638, .2982)),
                    "解く・属性": ([.1135, .1266, .1300, .1370, .1394, .1372, .1406], (.3870, .5537))},
}
COL = {"関係の表": "#4c72b0", "解く・属性": "#dd8452"}


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True)
    ys = list(range(len(NAMES)))[::-1]
    h = .38
    for j, start in enumerate(D):
        a = ax[j]
        for k, (form, (vals, room)) in enumerate(D[start].items()):
            pos = [y + (.5 - k) * h for y in ys]
            a.barh(pos, vals, h, color=COL[form], label=form)
            for p, v in zip(pos, vals):
                a.text(v + .004, p, f"{v:.3f}"[1:], va="center", fontsize=8)
            a.axvline(vals[0], color=COL[form], ls=":", lw=1)
            for r, ls, name, ha, dx in zip(room, ("--", "-."), ("上位 3", "上位 5"), ("right", "left"), (-.006, .006)):
                a.axvline(r, color=COL[form], ls=ls, lw=1.4)
                a.text(r + dx, len(NAMES) - .3 - .32 * k, f"{name}: {r:.2f}"[:6] + f"{r:.2f}"[1:], color=COL[form], fontsize=8, ha=ha,
                       bbox=dict(fc="white", ec="none", pad=.5))
        a.set_yticks(ys)
        a.set_yticklabels(NAMES)
        a.set_ylim(-.6, len(NAMES) - .05)
        a.set_xlim(0, .66)
        a.set_xlabel("2 席とも当たった割合（点線は別々、破線は上位に 2 席とも正解が入る割合）")
        a.set_title(start)
        a.legend(loc="lower right", fontsize=9)
        a.grid(axis="x", alpha=.3)
    fig.suptitle("2 つの席を同時に隠す問い: 一緒に選ぶと少し上がるが、上限（破線）にはまだ遠い", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49v_two_hidden.png")
