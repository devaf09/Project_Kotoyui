"""I49y の図: 正解が 2〜3 位だった問いで、1 位を取った語はどんな語か（解く・属性、席あり）。

数は PC で数えた集計値をそのまま置いている（白紙から）。
使い方: python tools/fig_i49y_who_wins.py 出力.png
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

G = ["全体", "だれが", "何を", "どうした"]
NEAR_TOP = [.3676, .5138, .1660, .3060]
NEAR_OTHER = [.1325, .1274, .1042, .1604]
ALSO = [.2439, .1505, .3734, .2530]
MORE = [.4382, .5043, .3910, .3901]
SAID_TOP = [.5518, .5602, .6186, .4774]
SAID_ANS = [.5487, .6247, .5577, .4337]


def bars(a, v1, v2, l1, l2, c1, c2):
    xs = range(len(G))
    a.bar([x - .2 for x in xs], v1, .4, color=c1, label=l1)
    a.bar([x + .2 for x in xs], v2, .4, color=c2, label=l2)
    for x, (p, q) in enumerate(zip(v1, v2)):
        a.text(x - .2, p + .01, f"{p:.2f}"[1:], ha="center", fontsize=9)
        a.text(x + .2, q + .01, f"{q:.2f}"[1:], ha="center", fontsize=9)
    a.set_xticks(list(xs))
    a.set_xticklabels(G)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="upper right")


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))
    bars(ax[0], NEAR_TOP, NEAR_OTHER, "1 位の語と正解", "ほかの候補と正解", "#dd8452", "#c9c9c9")
    ax[0].set_title("① 属性の近さ: 1 位は正解に似た語")
    ax[0].set_ylim(0, .7)
    a = ax[1]
    xs = range(len(G))
    a.bar(xs, ALSO, .5, color="#55a868")
    for x, v in enumerate(ALSO):
        a.text(x, v + .01, f"{v:.2f}"[1:], ha="center", fontsize=9)
    a.set_xticks(list(xs))
    a.set_xticklabels(G)
    a.set_ylim(0, .7)
    a.grid(axis="y", alpha=.3)
    a.set_title("② 1 位が「正しいと言える物」だった割合")
    bars(ax[2], SAID_TOP, SAID_ANS, "1 位の語がこの話でもう出た", "正解がこの話でもう出た", "#4c72b0", "#a9bfe0")
    ax[2].plot(range(len(G)), MORE, "k_", ms=22, mew=2, label="1 位の方が多く聞いた語")
    ax[2].legend(fontsize=8, loc="upper right")
    ax[2].set_ylim(0, .9)
    ax[2].set_title("③ 聞いた回数や、話で出たかは差が無い")
    fig.suptitle("正解が 2〜3 位だった 2,218 問（白紙から）: 1 位を取るのは、正解に属性が似た別の語", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49y_who_wins.png")
