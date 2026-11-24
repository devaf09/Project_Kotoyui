"""I49zx の図: 介入の板（1 つ目の段階）の最初の数え（448 話・13,392 問、元の順）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49zx_teach_board.py 出力.png
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

RATES = ["介入 10%\n教えた数 1,028", "介入 30%\n教えた数 3,148", "介入 100%\n教えた数 10,513"]
# （教えた組そのもの: 問い, 本体, 板）（似た組: 問い, 本体, 板）
DATA = [((640, .4828, .3563), (8743, .4063, .2212)), ((1586, .4735, .3455), (8606, .3990, .2482)), ((3301, .4717, .3441), (7148, .3798, .2745))]
TOTAL = [(.3894, .3893, 0, 2), (.3894, .3893, 4, 6), (.3894, .3894, 0, 0)]


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    w = .38
    for a, k, title in ((ax[0], 0, "教えた組そのものが、また問われた時"), (ax[1], 1, "教えていない似た組（板が答えを持った問い）")):
        for i, d in enumerate(DATA):
            n, body, board = d[k]
            a.bar(i - w / 2, body, w * .94, color="#55a868", label="本体（数えて選ぶ）の当たり" if i == 0 else None)
            a.bar(i + w / 2, board, w * .94, color="#dd8452", label="介入の板だけの当たり" if i == 0 else None)
            a.text(i - w / 2, body + .008, ("%.3f" % body)[1:], ha="center", fontsize=10)
            a.text(i + w / 2, board + .008, ("%.3f" % board)[1:], ha="center", fontsize=10)
            a.text(i, body + .04, "{:,} 問".format(n), ha="center", fontsize=9, color="#555555")
        a.set_xticks(range(3))
        a.set_xticklabels(RATES, fontsize=9)
        a.set_title(title, fontsize=11)
        a.grid(axis="y", alpha=.3)
        a.legend(fontsize=9, loc="upper right")
    ax[0].set_ylim(0, .62)
    ax[0].set_ylabel("当たった割合（元の順）")
    fig.suptitle("介入の板だけの答えは、本体より当たらない。だから数えて選ぶ形は板を使わず、全体の点は変わらない（.3894 のまま、害は 0〜6 問）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zx_teach_board.png")
