"""I49zt の図: 選び直し（もう出た語・仮の板）を、切り出しを通さない材料で数え直した結果。

数は PC で数えた集計値をそのまま置いている（448 話・4,464 組、13,392 問）。古い材料の数は I49u・I49z・I49zb の白紙からの数。
使い方: python tools/fig_i49zt_repick_clean.py 出力.png
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

GRAY, GREEN, RED = "#8c8c8c", "#55a868", "#c44e52"
FORMS = [("解く形", GRAY, [.3415, .3421, .3436, .3418], ""),
         ("＋出た語", GRAY, [.2325, .2331, .2323, .2330], "z −26"),
         ("＋席ごと\nの板", GRAY, [.3176, .3169, .3191, .3198], "z −8〜−9"),
         ("数えて選ぶ", GREEN, [.3894, .3910, .3901, .3903], ""),
         ("＋出た語", GREEN, [.3855, .3852, .3835, .3821], "z −1.5〜−3.2")]
OLD = [("解く形", .408), ("＋出た語", .431), ("＋席ごとの板", .417)]
SEEN = [("だれが", .6183, .7000), ("何を", .4105, .2039), ("どうした", .3941, .0726)]
# 並べ替え 3、解く形と ＋出た語 の、正解がもう出た問い（4,359）／まだ出ていない問い（9,033）での当たり
SPLIT = [("正解がもう出た問い\n4,359 問", .4815, .5380), ("正解がまだ出ていない問い\n9,033 問", .2743, .0859)]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    a = ax[0]
    for i, (name, col, ys, z) in enumerate(FORMS):
        a.bar(i, sum(ys) / 4, .7, color=col, alpha=1 if i in (0, 3) else .55)
        a.plot([i] * 4, ys, ".", color="k", ms=3)
        a.text(i, max(ys) + .006, ("%.3f" % (sum(ys) / 4))[1:], ha="center", fontsize=10)
        if z:
            a.text(i, max(ys) + .026, z, ha="center", fontsize=9, color=RED)
    a.text(1, .455, "古い材料では: 解く形 .408、＋出た語 .431、＋席ごとの板 .417\n（どちらも上がっていた）", ha="center", fontsize=9.5)
    a.set_xticks(range(len(FORMS)))
    a.set_xticklabels([f[0] for f in FORMS], fontsize=9)
    a.set_ylim(.2, .49)
    a.set_ylabel("正しい物（棒は 4 通りの並べ方の平均、点は並べ方ごと）")
    a.set_title("きれいな材料では、選び直しはどれも下がる", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a = ax[1]
    w = .38
    for i, (s, o, c) in enumerate(SEEN):
        a.bar(i - w / 2, o, w * .94, color="#b07aa1", label="古い材料（自作の 600 話）" if i == 0 else None)
        a.bar(i + w / 2, c, w * .94, color="#4c72b0", label="きれいな材料（448 話）" if i == 0 else None)
        a.text(i - w / 2, o + .012, ("%.2f" % o)[1:], ha="center", fontsize=10)
        a.text(i + w / 2, c + .012, ("%.2f" % c)[1:], ha="center", fontsize=10)
    a.set_xticks(range(3))
    a.set_xticklabels([s[0] for s in SEEN])
    a.set_ylim(0, .85)
    a.set_ylabel("正解が、その話でもう出た語だった問いの割合")
    a.set_title("理由: 古い材料は、同じ語が何度も出ていた", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper right")
    a = ax[2]
    for i, (lab, base, said) in enumerate(SPLIT):
        a.bar(i - w / 2, base, w * .94, color=GRAY, label="解く形" if i == 0 else None)
        a.bar(i + w / 2, said, w * .94, color=GRAY, alpha=.55, label="解く形＋出た語" if i == 0 else None)
        a.text(i - w / 2, base + .012, ("%.3f" % base)[1:], ha="center", fontsize=10)
        a.text(i + w / 2, said + .012, ("%.3f" % said)[1:], ha="center", fontsize=10)
    a.set_xticks(range(2))
    a.set_xticklabels([s[0] for s in SPLIT], fontsize=9)
    a.set_ylim(0, .7)
    a.set_ylabel("当たった割合（きれいな材料、並べ替え 3）")
    a.set_title("出た語は、正解がまだ出ていない問いを壊す", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper right")
    fig.suptitle("「もう出た語で選び直す」は、古い材料の崩れ（同じ語のくり返し）で良く見えていた。きれいな材料では逆に下がる", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zt_repick_clean.png")
