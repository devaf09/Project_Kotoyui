"""I49zp の図（2 枚目）: 切り出しを通さない材料 3 通り（前の版・修正版・合わせた物）での、形ごとの当たり。

数は PC で数えた集計値をそのまま置いている。棒は 4 通りの並べ方の平均、点は並べ方ごと。
使い方: python tools/fig_i49zp_three_materials.py 出力.png
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

FORMS = ["関係の表", "属性だけ", "属性＋固有の位相", "＋振幅", "数えて選ぶ"]
COL = ["#4c72b0", "#8c8c8c", "#9ecae1", "#dd8452", "#55a868"]
MAT = {"前の版\n194 話・1,415 組": [[.2761, .2994, .2855, .2978, .3185], [.2761, .2921, .2855, .2989, .3126], [.2759, .2952, .2789, .2963, .3143], [.2780, .3001, .2813, .3013, .3211]],
       "修正版\n195 話・1,926 組": [[.2786, .2833, .2707, .2953, .3075], [.2778, .2819, .2703, .3070, .3129], [.2771, .2838, .2717, .3017, .3089], [.2741, .2819, .2705, .2956, .3067]],
       "合わせた物\n389 話・3,341 組": [[.3362, .3343, .3219, .3515, .3709], [.3379, .3347, .3231, .3609, .3721], [.3392, .3384, .3263, .3615, .3728], [.3402, .3410, .3263, .3611, .3751]]}
# 合わせた物・並べ替え 3、段ごと（表の当たり、解く形・属性だけ、＋振幅、問い）
LEVELS = [("聞いた組\nそのもの", 655, 593, 627, 1259), ("片方だけ\n借りる", 1090, 899, 998, 1819), ("述語側", 447, 358, 393, 764),
          ("もう片方", 566, 680, 698, 3426), ("借りる", 652, 662, 686, 1847), ("答えられ\nない", 0, 226, 217, 908)]
NOATTR = [("前の版", 32.0), ("修正版", 35.9), ("合わせた物", 34.2)]


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]
    w = .16
    for mi, (name, rows) in enumerate(MAT.items()):
        for k in range(5):
            x = mi + (k - 2) * w
            ys = [r[k] for r in rows]
            a.bar(x, sum(ys) / 4, w * .94, color=COL[k], label=FORMS[k] if mi == 0 else None)
            a.plot([x] * 4, ys, ".", color="k", ms=3)
            a.text(x, max(ys) + .004, ("%.3f" % (sum(ys) / 4))[1:], ha="center", fontsize=8)
        a.text(mi, .215, "属性の無い語を含む行 %.0f%%" % NOATTR[mi][1], ha="center", fontsize=9)
    a.set_xticks(range(3))
    a.set_xticklabels(list(MAT))
    a.set_ylim(.2, .4)
    a.set_ylabel("正しい物（棒は 4 通りの並べ方の平均、点は並べ方ごと）")
    a.set_title("どの材料でも 数えて選ぶ が一番上。固有の位相は属性だけより下")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper left")
    a = ax[1]
    w = .27
    for i, (name, t, s, p, n) in enumerate(LEVELS):
        a.bar(i - w, t / n, w * .94, color=COL[0], label="関係の表" if i == 0 else None)
        a.bar(i, s / n, w * .94, color=COL[1], label="解く形・属性だけ" if i == 0 else None)
        a.bar(i + w, p / n, w * .94, color=COL[3], label="属性＋固有の位相＋振幅" if i == 0 else None)
        a.text(i, max(t, s, p) / n + .02, "%d 問" % n, ha="center", fontsize=9)
    a.axvline(2.5, color="k", lw=.8, ls=":")
    a.text(1, .78, "聞いた事の有る組\n3,842 問（38%）", ha="center", fontsize=10)
    a.text(4, .78, "聞いた事の無い組\n6,181 問（62%）", ha="center", fontsize=10)
    a.set_xticks(range(len(LEVELS)))
    a.set_xticklabels([l[0] for l in LEVELS], fontsize=9)
    a.set_ylim(0, .9)
    a.set_ylabel("当たった割合（合わせた物、並べ替え 3）")
    a.set_title("段ごと: 聞いた組は表、聞いた事の無い組は解く形")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1, .74))
    fig.suptitle("切り出しを通さない材料 3 通りで数え直す: 材料が倍になると全体が上がる。表と解く形（属性だけ）は並び", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zp_three_materials.png")
