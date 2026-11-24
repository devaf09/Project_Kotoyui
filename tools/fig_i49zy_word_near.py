"""I49zy の図: 語どうしの近さ（属性の波の cos）を、3 つの形で並べる。「学生が 歌を」が直しても近いままだった理由を見る。

数は PC で数えた集計値をそのまま置いている。語は、動作試験のために自分で作った問いの語。
使い方: python tools/fig_i49zy_word_near.py 出力.png
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

FORMS = ("そのまま", "いつもの属性を引く", "めずらしさを掛ける")
COL = ("#9aa3ad", "#2f6fb0", "#d98c2b")
GA = [("父-母", .9716, .9548, .9638), ("客-店員", .8733, .6952, .6487), ("猫-犬", .6197, .6102, .5665), ("学生-先生", .7179, .3416, .4049),
      ("学生-父", .5302, .0629, .2377), ("先生-父", .5066, .1079, .1607), ("漁師-子供", .4302, .0643, .1094), ("猫-客", .1383, -.3138, .0372),
      ("学生-猫", .1432, -.3282, .0652), ("ふつうの組の\n上から 10% の所", .5981, .2775, .2694)]
WO = [("本-手紙", .5423, .4131, .4361), ("本-歌", .4927, .4008, .4472), ("歌-手紙", .4930, .4139, .4492), ("魚-パン", .2639, .1500, .1631),
      ("本-魚", .1504, -.0401, .0266), ("本-水", .1512, -.0498, .0236), ("魚-水", .1308, -.0044, .0130), ("歌-水", .1212, .0151, .0128),
      ("ふつうの組の\n上から 10% の所", .3197, .1333, .1705)]


def main(out):
    fig, ax = plt.subplots(2, 1, figsize=(13, 9.6))
    for a, data, title in ((ax[0], GA, "「が」の席の語どうし"), (ax[1], WO, "「を」の席の語どうし")):
        for k in range(3):
            xs = [i + (k - 1) * .27 for i in range(len(data))]
            a.bar(xs, [d[1 + k] for d in data], .26, color=COL[k], label=FORMS[k])
            for x, d in zip(xs, data):
                v = d[1 + k]
                a.text(x, v + (.02 if v >= 0 else -.06), ("%.2f" % v).replace("0.", "."), ha="center", fontsize=8)
        a.axvline(len(data) - 1.5, color="#888", lw=.8, ls=":")
        a.axhline(0, color="#333", lw=.8)
        a.set_xticks(range(len(data)))
        a.set_xticklabels([d[0] for d in data], fontsize=11)
        a.set_ylim(-.45, 1.12)
        a.set_ylabel("近さ（cos）")
        a.set_title(title, fontsize=13, loc="left")
        a.grid(axis="y", alpha=.3)
    ax[0].legend(loc="upper right", fontsize=10, ncol=3)
    i = [d[0] for d in WO].index("本-歌")
    ax[1].annotate("直しても .40〜.45 で、本-手紙 と同じ高さ。\n属性の上で、本 と 歌 がもともと近い", xy=(i + .27, .52), xytext=(1.5, .82), fontsize=11,
                   arrowprops=dict(arrowstyle="->", color="#333"))
    ax[0].annotate("学生-父 は、直すと離れる", xy=(4, .26), xytext=(4.6, .62), fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    fig.suptitle("語どうしの近さ: どの語も持つ属性を除くと「が」の席は離れるが、本 と 歌 は近いまま（材料 3 つ、属性 1,997 個）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
