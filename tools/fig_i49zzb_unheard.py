"""I49zzb の図: 本体が聞いていない事を教える場面での、介入の板の数え（448 話の半分を本体に聞かせ、残り半分で問う）。

数は PC で数えた集計値をそのまま置いている（並べ替え 1 と 2）。
使い方: python tools/fig_i49zzb_unheard.py 出力.png
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

FORMS = ["本体だけ\n（後半を聞かない）", "介入の板を付ける\n介入 10%", "介入 30%", "介入 100%", "板が答えを持てば\nいつも板（介入 100%）", "本体が後半も聞く\n（比べ役、板なし）"]
ALL = [(.4057, .4057, .4057, .4057, .3033, .4542), (.3912, .3912, .3911, .3909, .3022, .4438)]
# 介入 100% の時の（問いの数, 本体の当たり, 板の当たり）: 教えた組そのもの、似た組
PART = [((1173, .4049, .3581), (3951, .4207, .2622)), ((1218, .3966, .3645), (3961, .3989, .2585))]


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw=dict(width_ratios=(1.5, 1)))
    a = ax[0]
    for k, (vals, col) in enumerate(zip(ALL, ("#2f6fb0", "#8fb4d9"))):
        xs = [i + (k - .5) * .38 for i in range(len(FORMS))]
        a.bar(xs, vals, .36, color=col, label="並べ替え %d（%s 問）" % (k + 1, ("6,651", "6,687")[k]))
        for x, v in zip(xs, vals):
            a.text(x, v + .006, ("%.3f" % v)[1:], ha="center", fontsize=9)
    a.set_xticks(range(len(FORMS)))
    a.set_xticklabels(FORMS, fontsize=9.5)
    a.set_ylim(0, .56)
    a.set_ylabel("当たった割合")
    a.set_title("全部の問い: 板を付けても、本体だけと同じ", fontsize=12.5, loc="left")
    a.legend(loc="upper left", fontsize=10)
    a.grid(axis="y", alpha=.3)
    a.annotate("本体が聞けば +.05 上がる分を、\n板は運べていない", xy=(5, .47), xytext=(2.6, .50), fontsize=10.5, arrowprops=dict(arrowstyle="->", color="#333"))
    b = ax[1]
    names = ["教えた組そのものが\nまた問われた時", "教えていない似た組"]
    for k in range(2):
        for j in range(2):
            n, body, board = PART[k][j]
            x = j * 2.4 + k
            b.bar(x - .2, body, .38, color="#9aa3ad", label="本体の当たり" if (k, j) == (0, 0) else None)
            b.bar(x + .2, board, .38, color="#d98c2b", label="板の当たり" if (k, j) == (0, 0) else None)
            b.text(x - .2, body + .006, ("%.2f" % body)[1:], ha="center", fontsize=9)
            b.text(x + .2, board + .006, ("%.2f" % board)[1:], ha="center", fontsize=9)
            b.text(x, -.035, "並べ替え %d\n%s 問" % (k + 1, format(n, ",")), ha="center", va="top", fontsize=9)
    b.set_xticks([.5, 2.9])
    b.set_xticklabels(names, fontsize=10.5)
    b.tick_params(axis="x", pad=34)
    b.set_ylim(0, .56)
    b.set_title("板が答えを持った問い（介入 100%）: 板だけの当たりは本体より下", fontsize=12.5, loc="left")
    b.legend(loc="upper right", fontsize=10)
    b.grid(axis="y", alpha=.3)
    fig.suptitle("本体が聞いていない事を教えても、介入の板の効き目は出なかった（448 話の半分を聞かせ、残り半分で問う。鍵の形は今まで）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
