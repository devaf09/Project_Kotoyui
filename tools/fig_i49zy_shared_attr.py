"""I49zy の図: 本-歌、本-手紙 の近さが、どの属性から来ているか（共通の属性の、割合の積の上位）。

数は PC で数えた集計値をそのまま置いている。属性の名前は内容語の板の物。
使い方: python tools/fig_i49zy_shared_attr.py 出力.png
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

# （属性, 割合の積, 「を」の席の語のうち持つ割合）
UTA = [("の部分: 著作", .4170, .019), ("格: を", .0371, .940), ("役: 目的語", .0371, .939), ("役: 主語", .0157, .917), ("話題: 個人", .0123, .996),
       ("性質: 好き", .0111, .136), ("取り立て: は", .0056, .835), ("役: 連体", .0049, .944)]
TEGAMI = [("の部分: 著作", .2085, .019), ("クラス: アーチファクト", .1052, .480), ("クラス: 創造物", .1052, .044), ("格: を", .0542, .940),
          ("役: 目的語", .0540, .939), ("読む/される", .0196, .052), ("クラス: 著作", .0131, .031), ("話題: 個人", .0112, .996)]


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    for a, data, title in ((ax[0], UTA, "本 と 歌（共通の属性 195 個）"), (ax[1], TEGAMI, "本 と 手紙（共通の属性 226 個）")):
        ys = list(range(len(data)))[::-1]
        a.barh(ys, [d[1] for d in data], color=["#c0504d" if d[2] < .1 else "#9aa3ad" for d in data])
        for y, d in zip(ys, data):
            a.text(d[1] + .008, y, "%.3f（持つ語 %.1f%%）" % (d[1], d[2] * 100), va="center", fontsize=10)
        a.set_yticks(ys)
        a.set_yticklabels([d[0] for d in data], fontsize=11)
        a.set_xlim(0, .62)
        a.set_xlabel("2 語の割合の積（大きいほど、近さに効く）")
        a.set_title(title, fontsize=13, loc="left")
        a.grid(axis="x", alpha=.3)
    ax[0].text(.2, 3.2, "近さのほとんどが、この 1 つの属性から来る。\n歌 はこの属性を割合 1.0 で持つ。\n持つ語が少ない属性なので、\n「どの語も持つ属性」を除いても残る", fontsize=11)
    fig.suptitle("本-歌 が近い理由: 内容語の板の 1 つの属性（赤 = 「を」の席の語の 1 割未満しか持たない属性、灰 = 多くの語が持つ属性）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
