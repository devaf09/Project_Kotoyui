"""I49zw の図: 2 語の組を「広げてから掛ける」束ね方（448 話・4,464 組、dim 8,192）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49zw_dense_pair.py 出力.png
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

GRAY, GREEN, ORANGE, PURPLE = "#8c8c8c", "#55a868", "#dd8452", "#b07aa1"
FORMS = [("組だけ\nまばら", PURPLE, [.1924, .1982, .1959, .1968]), ("組だけ\n広げる", ORANGE, [.2816, .2861, .2801, .2811]),
         ("解く形", GRAY, [.3415, .3421, .3436, .3418]), ("解く形＋\n今までの束ね", PURPLE, [.3321, .3326, .3339, .3336]),
         ("解く形＋\n組・広げる", ORANGE, [.3465, .3507, .3485, .3488]), ("数えて選ぶ", GREEN, [.3894, .3910, .3901, .3903]),
         ("数えて選ぶ\n・組と", ORANGE, [.3934, .3952, .3927, .3936])]
NOTES = {1: "まばら → 広げる\nz +22〜+23", 3: "z −3.7〜−4.3", 4: "z +2.7〜+4.8", 6: "z +2.0〜+3.5"}
PAIRS = [("だれが×何を", .0479), ("だれが×どうした", .0205), ("何を×どうした", .0160)]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[.8, .8, 1.6]))
    a = ax[0]
    vals = [1997, 152.9, 102.4]
    labs = ["属性の総数\n（周波数の数）", "1 語が持つ\n属性の数（平均）", "2 語に共通の\n属性の数（平均）"]
    a.bar(range(3), vals, .6, color=[GRAY, "#4c72b0", PURPLE])
    for i, v in enumerate(vals):
        a.text(i, v + 30, "{:,.0f}".format(v), ha="center", fontsize=11)
    a.set_xticks(range(3))
    a.set_xticklabels(labs, fontsize=9)
    a.set_ylim(0, 2300)
    a.set_title("語の波は、周波数の 7.7% にだけ力が有る。\nしかも、どの 2 語も 100 個ほどの属性を共に持つ", fontsize=10.5)
    a.grid(axis="y", alpha=.3)
    a = ax[1]
    a.bar(range(3), [p[1] for p in PAIRS], .6, color=PURPLE)
    for i, p in enumerate(PAIRS):
        a.text(i, p[1] + .001, ("%.3f" % p[1])[1:], ha="center", fontsize=11)
    a.set_xticks(range(3))
    a.set_xticklabels([p[0] for p in PAIRS], fontsize=9)
    a.set_ylim(0, .07)
    a.set_ylabel("今までの掛け算で残る長さ（元の 2 語は長さ 1）")
    a.set_title("そのまま掛けると、長さの 2〜5% しか残らない\n（残るのは、どの語も持つ属性）", fontsize=10.5)
    a.grid(axis="y", alpha=.3)
    a = ax[2]
    for i, (name, col, ys) in enumerate(FORMS):
        a.bar(i, sum(ys) / 4, .7, color=col)
        a.plot([i] * 4, ys, ".", color="k", ms=3)
        a.text(i, max(ys) + .005, ("%.3f" % (sum(ys) / 4))[1:], ha="center", fontsize=10)
        if i in NOTES:
            a.text(i, max(ys) + .024, NOTES[i], ha="center", fontsize=8.5, color="#c44e52" if "−" in NOTES[i] else "#2a7a3a")
    a.set_xticks(range(len(FORMS)))
    a.set_xticklabels([f[0] for f in FORMS], fontsize=9)
    a.set_ylim(.15, .46)
    a.set_ylabel("正しい物（棒は 4 通りの並べ方の平均、点は並べ方ごと）")
    a.set_title("広げてから掛けると、組の印として働く（.196 → .282）。解く形に足した上積みは小さい（+0.6 ポイント）", fontsize=10.5)
    a.grid(axis="y", alpha=.3)
    fig.suptitle("助言「広げて密にしてから掛ける」を試す: 前提は合っていた。束ね方としては良くなる。全体の点への上積みは小さい", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zw_dense_pair.png")
