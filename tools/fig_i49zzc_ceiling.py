"""I49zzc の図: 「2 語の組で覚えて引く」事の上限（448 話の半分を本体に聞かせ、残り半分で問う。並べ替え 1）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49zzc_ceiling.py 出力.png
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

NAMES = ("本体（後半を聞かない）", "組で覚えて、一番多く出た語で答える", "組で覚える事の上限\n（正解が、その組で前に出た語のどれか）", "本体（後半も聞く）")
COL = ("#9aa3ad", "#d98c2b", "#f0c890", "#2f6fb0")
GROUPS = [("同じ 2 語の組が、後半でもう出ていた問い\n1,383 問（全体の 21%）", (.389, .2227, .2755, .4714)),
          ("同じ組がまだ出ていない問い\n5,268 問（全体の 79%）", (.41, 0, 0, .4497)),
          ("全部\n6,651 問", (.4057, .0463, .0573, .4542))]


def main(out):
    fig, ax = plt.subplots(figsize=(13.5, 6))
    for g, (name, vals) in enumerate(GROUPS):
        for k, v in enumerate(vals):
            x = g + (k - 1.5) * .2
            ax.bar(x, v, .19, color=COL[k], label=NAMES[k] if g == 0 else None)
            ax.text(x, v + .008, ("%.3f" % v)[1:] if v else "0", ha="center", fontsize=9.5)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g[0] for g in GROUPS], fontsize=10.5)
    ax.set_ylim(0, .78)
    ax.set_ylabel("当たった割合")
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.annotate("同じ 2 語の組でも、次の語は同じとは限らない。\n前に出た語を全部知っていても .28 までで、\n後半を聞いていない本体（.39）より下", xy=(.1, .29), xytext=(.25, .5), fontsize=10.5, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("組が出ていない問いでも、\n本体は聞けば +.04 上がる\n（語ごとの覚えの分）", xy=(1.3, .46), xytext=(1.15, .5), fontsize=10.5)
    ax.set_title("介入の板が効かなかった理由: この材料では「2 語の組 → 語」の覚えは、上限が低い", fontsize=13, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
