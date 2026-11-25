"""I49zzd の図: 土台の確かめ。外付けに持つ形と、本体に直に混ぜる形の比べ（並べ替え 1〜3 の平均）と、外付けを使って外した問いの段の組。

数は PC で数えた集計値（run23・run24）から。平均は並べ替え 3 通りの単純な平均。
使い方: python tools/fig_i49zzd_checks.py 出力.png
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

BASE, ALL = .3990, .4494
GROUPS = ["介入 10%", "介入 30%", "介入 100%", "外れた時だけ\n30%", "外れた時だけ\n100%"]
EXT = [.4060, .4312, .4416, .4056, .4230]
DIRECT = [.4282, .4445, .4494, .4375, .4446]
# 介入 100%・並べ替え 1〜3 の合計（段の組: 外付けを使った問い, 当たりに変わった, 外れに変わった）
CELLS = [("借りる｜解く", 2406, 381, 223), ("借りる｜借りる", 838, 130, 69), ("解く｜解く", 525, 69, 59), ("借りる｜片方だけ借りる", 479, 187, 26),
         ("もう片方｜片方だけ借りる", 352, 139, 25), ("もう片方｜2語", 332, 113, 19), ("借りる｜2語", 247, 103, 13)]


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw=dict(width_ratios=(1.25, 1)))
    a = ax[0]
    for k, (vals, col, name) in enumerate(((EXT, "#2e8b57", "外付けに持つ（外せば元に戻る）"), (DIRECT, "#7b5ea7", "本体に直に混ぜる（外せない）"))):
        xs = [i + (k - .5) * .38 for i in range(len(GROUPS))]
        a.bar(xs, vals, .36, color=col, label=name)
        for x, v in zip(xs, vals):
            a.text(x, v + .002, ("%.3f" % v)[1:], ha="center", fontsize=9.5)
    a.axhline(BASE, color="#666", ls="--", lw=1.2)
    a.text(4.72, BASE + .002, "本体だけ .399", ha="right", fontsize=10, color="#444")
    a.axhline(ALL, color="#2f6fb0", ls="--", lw=1.2)
    a.text(4.72, ALL + .002, "本体が後半も聞く .449", ha="right", fontsize=10, color="#2f6fb0")
    a.axvline(2.5, color="#888", lw=.8, ls=":")
    a.set_xticks(range(len(GROUPS)))
    a.set_xticklabels(GROUPS, fontsize=10.5)
    a.set_ylim(.38, .47)
    a.set_ylabel("当たった割合（並べ替え 3 通りの平均。縦の軸は .38 から）")
    a.set_title("分けて持つ事の代金: 教えが少ない時と、外れた時だけ教える時に大きい", fontsize=12.5, loc="left")
    a.legend(loc="upper left", fontsize=10)
    a.grid(axis="y", alpha=.3)
    b = ax[1]
    ys = list(range(len(CELLS)))[::-1]
    b.barh(ys, [c[2] for c in CELLS], color="#2e8b57", label="当たりに変わった")
    b.barh(ys, [-c[3] for c in CELLS], color="#c0504d", label="外れに変わった（害）")
    for y, c in zip(ys, CELLS):
        b.text(c[2] + 6, y, str(c[2]), va="center", fontsize=9.5)
        b.text(-c[3] - 6, y, str(c[3]), va="center", ha="right", fontsize=9.5)
    b.axvline(0, color="#333", lw=.8)
    b.set_yticks(ys)
    b.set_yticklabels(["本体: %s\n外付け: %s" % tuple(c[0].split("｜")) for c in CELLS], fontsize=9.5)
    b.set_xlim(-300, 460)
    b.set_xlabel("問いの数（介入 100%、並べ替え 3 通りの合計）")
    b.set_title("害が出る場面: どちらも弱い手がかりで答えた時に多い", fontsize=12.5, loc="left")
    b.legend(loc="lower right", fontsize=10)
    b.grid(axis="x", alpha=.3)
    fig.suptitle("土台の確かめ（448 話の半分を本体に聞かせ、残り半分で問う）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
