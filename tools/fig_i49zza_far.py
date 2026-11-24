"""I49zza の図: 「学生が 歌を は遠い（板の出番ではない）」と 1 回教えた前後の、板の点（鍵の形は今まで、同じ 19 問のうち 8 問）。

数は PC で数えた集計値をそのまま置いている。問いは、動作試験のために自分で作った物。
使い方: python tools/fig_i49zza_far.py 出力.png
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

# （問い, 種類, 読む: 前, 後, 書く: 前, 後）
DATA = [("学生が 本を", "教えた", 1.81, 1.51, .81, .66), ("母が 手紙を", "教えた", 1.35, 1.21, 1.17, 1.10), ("父が 本を", "教えていない\n似た問い", 1.17, 1.05, .75, .69),
        ("学生が 雑誌を", "教えていない\n似た問い", .84, .81, .37, .36), ("学生が 手紙を", "教えていない\n似た問い", 1.05, .73, .67, .50),
        ("学生が 歌を", "「遠い」\nと教えた", .79, 0, .41, 0), ("先生が 歌を", "教えていない\n遠い問い", .77, .27, .54, .28), ("父が 歌を", "教えていない\n遠い問い", .70, .37, .55, .38)]
COL = ("#9aa3ad", "#2f6fb0", "#d9b3b1", "#c0504d")
NAMES = ("読む（教える前）", "読む（「遠い」を教えた後）", "書く（教える前）", "書く（「遠い」を教えた後）")


def main(out):
    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    for i, d in enumerate(DATA):
        for k in range(4):
            v = d[2 + k]
            x = i + (k - 1.5) * .21
            ax.bar(x, v, .2, color=COL[k], label=NAMES[k] if i == 0 else None)
            ax.text(x, v + .03, ("%.2f" % v).replace("0.", ".") if v else "0", ha="center", fontsize=8.5)
    ax.axvline(4.5, color="#888", lw=.8, ls=":")
    ax.set_xticks(range(len(DATA)))
    ax.set_xticklabels(["%s\n%s" % (d[0], d[1]) for d in DATA], fontsize=10)
    ax.set_ylim(0, 2.3)
    ax.set_ylabel("板の点")
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    ax.text(4.75, 1.2, "教えた問いは、どの語の点も 0 になり、板は答えなくなる。\n教えていない 歌 の問いは、半分ほどに下がるが 0 にはならない\n（読む と 書く がほぼ同じ点で残る）", fontsize=11)
    ax.annotate("手紙 は 読む が上のまま\n（.73 対 .50）", xy=(3.9, .78), xytext=(3.9, 1.6), ha="center", fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("「遠い」を 1 回教えた前後: 教えた問いでは板が黙る。似た遠い問いは弱まるだけで、まだ答える", fontsize=13, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
