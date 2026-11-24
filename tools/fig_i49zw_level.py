"""I49zw の図（4 枚目）: 組の印（位相だけ）を解く形に足した時、どの種類の問いが当たりに変わったか（448 話・13,392 問、元の順）。

数は PC で数えた集計値をそのまま置いている。問いの種類 = 関係の表がその問いに答える段。
使い方: python tools/fig_i49zw_level.py 出力.png
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

# 段, 問い, 当たりに変わった, 外れに変わった（元の順）, 差し引き（並べ替え 1）
LV = [("聞いた組\nそのもの", 2035, 77, 25, 61), ("片方だけ\n借りる", 3045, 115, 82, 50), ("述語側", 1022, 18, 20, 5),
      ("もう片方", 4183, 99, 67, 20), ("借りる", 2212, 34, 30, 16), ("答えられ\nない", 895, 7, 2, 5)]


def main(out):
    fig, a = plt.subplots(figsize=(11, 5.4))
    w = .38
    for i, (name, n, up, dn, net1) in enumerate(LV):
        a.bar(i - w / 2, up, w * .94, color="#55a868", label="当たりに変わった" if i == 0 else None)
        a.bar(i + w / 2, dn, w * .94, color="#c44e52", label="外れに変わった" if i == 0 else None)
        a.text(i, max(up, dn) + 5, "差し引き %+d\n（並べ替え 1 では %+d）" % (up - dn, net1), ha="center", fontsize=9)
        a.text(i, -13, "{:,} 問".format(n), ha="center", fontsize=8.5, color="#555555")
    a.axvline(2.5, color="k", lw=.8, ls=":")
    a.text(1, 150, "聞いた事の有る組", ha="center", fontsize=10)
    a.text(4, 150, "聞いた事の無い組", ha="center", fontsize=10)
    a.set_xticks(range(len(LV)))
    a.set_xticklabels([l[0] for l in LV], fontsize=9)
    a.tick_params(axis="x", pad=16)
    a.set_ylim(-20, 165)
    a.set_ylabel("問いの数")
    a.set_title("組の印がはっきり効くのは「聞いた組そのもの」（77 対 25）。聞いた事の無い似た組へは、あまり広がらない", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zw_level.png")
