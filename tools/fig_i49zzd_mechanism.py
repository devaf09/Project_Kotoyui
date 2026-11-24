"""I49zzd の図: 介入の土台（本体と同じ形の外付け）の仕組み。説明用の絵で、数えの図ではない。

使い方: python tools/fig_i49zzd_mechanism.py 出力.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

FONT = os.environ.get("KOTOYUI_FONT", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, color, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, a, b, text=None, color="#333", ls="-", off=(0, .12)):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=color, lw=1.6, ls=ls))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", fontsize=9.5, color=color)


def main(out):
    fig, ax = plt.subplots(figsize=(14, 6.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    box(ax, .3, 2.8, 2.3, 1.1, "問い\n「学生が 本を （？）」", "#f2f2f2")
    box(ax, 3.7, 4.3, 3.4, 1.5, "本体\n今までに聞いた話の覚え\n（介入では書き換えない）", "#cfe0f3")
    box(ax, 3.7, .9, 3.4, 1.5, "外付け（本体と同じ形）\n教わった 3 つ組だけの覚え\n（外せば元に戻る）", "#cfe9d6")
    box(ax, 8.3, 2.6, 2.6, 1.5, "選ぶ\n答えた段の組ごとに\n当たりが多かった方\n（同じなら本体）", "#fbe6c2", 10.5)
    box(ax, 11.9, 2.8, 1.8, 1.1, "最後の答え", "#f2f2f2")
    box(ax, 8.3, .2, 2.6, 1.0, "教え（介入）\n「正しくは 読む」", "#f6d5d3")
    arrow(ax, (2.6, 3.6), (3.7, 4.9))
    arrow(ax, (2.6, 3.1), (3.7, 1.8))
    arrow(ax, (7.1, 5.0), (8.3, 3.8), "答えと、答えた段", off=(.55, .2))
    arrow(ax, (7.1, 1.7), (8.3, 2.9), "答えと、答えた段", off=(.6, -.38))
    arrow(ax, (10.9, 3.35), (11.9, 3.35))
    arrow(ax, (8.3, .7), (7.1, 1.3), "3 つ組として聞かせる\n（その 3 つ組の問いが\n全部終わった後）", "#a33", off=(-.2, -.95))
    arrow(ax, (9.6, 1.2), (9.6, 2.6), "どちらが当たったかを数える\n（教えが届いた問いだけ）", "#a33", off=(1.75, -.1))
    ax.text(5.4, 6.15, "教えが届かない時は、本体だけの時と全く同じ答えになる", ha="center", fontsize=10.5, color="#245")
    ax.set_title("介入の土台の仕組み: 教わった事は本体に混ぜず、本体と同じ形の外付けに持ち、数えて選ぶ", fontsize=13, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
