"""I49ze の図: 仮の板の語を並べて一斉に走査した時の、正しい物の割合と、答えの変わり方。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49ze_row_board.py 出力.png
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

X = ["1 組で半分", "2 組", "4 組", "薄れない\n（話ごとに空）"]
# 始め方: (板無し, 出た語・上位 3, 足した板・九割以内・席ごと（I49zd）, 並べた板・九割以内・席ごと, 並べた板・九割以内・1 枚)
D = {
    "白紙から": (.4084, .4307, [.4169, .4100, .4084, .4090], [.4169, .4172, .4172, .4010], [.4167, .4172, .4175, .3994]),
    "青空の 3 つ組を先に": (.3983, .4242, [.4076, .3970, .3942, .3987], [.4071, .4078, .4081, .3918], [.4079, .4083, .4058, .3878]),
}
# 板無しから答えが変わった問い（白紙から）: (当たりに変わった, 外れに変わった, 当たり外れは同じまま)
CH = [("出た語・上位 3\n（決まりで書いた形）", 874, 691, 2703),
      ("並べた板・九割以内\n席ごと・1 組で半分", 532, 462, 1724),
      ("並べた板・上位 3\n席ごと・1 組で半分", 981, 1196, 3973)]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.1]))
    for j, start in enumerate(D):
        a = ax[j]
        base, said, add, row, row1 = D[start]
        a.axhline(said, color="#55a868", lw=1.5, ls="-.", label="出た語・上位 3（決まりで書いた形） " + f"{said:.3f}"[1:])
        a.axhline(base, color="k", lw=1, ls="--", label="板無し " + f"{base:.3f}"[1:])
        for ys, c, m, name in ((add, "#8c8c8c", "^", "足した板・席ごと（前の試し）"), (row, "#dd8452", "s", "並べた板・席ごと"), (row1, "#4c72b0", "o", "並べた板・1 枚")):
            a.plot(range(3), ys[:3], color=c, marker=m, label=name)
            a.plot([3], ys[3:], color=c, marker=m, ls="none", mfc="white")
        a.set_xticks(range(4))
        a.set_xticklabels(X, fontsize=9)
        a.set_ylim(.375, .44)
        a.set_ylabel("正しい物の割合（九割以内の候補から選び直す）")
        a.set_title(f"{start}: 並べると、薄れる速さが効かなくなる")
        a.grid(alpha=.3)
        a.legend(fontsize=8, loc="lower left")
    a = ax[2]
    xs = range(len(CH))
    up = [c[1] for c in CH]
    dn = [c[2] for c in CH]
    same = [c[3] - c[1] - c[2] for c in CH]
    a.bar(xs, up, .6, color="#55a868", label="当たりに変わった")
    a.bar(xs, dn, .6, bottom=up, color="#c44e52", label="外れに変わった")
    a.bar(xs, same, .6, bottom=[u + d for u, d in zip(up, dn)], color="#d9d9d9", label="変わったが、当たり外れは同じ")
    for x, u, d in zip(xs, up, dn):
        a.text(x, u / 2, str(u), ha="center", va="center", fontsize=9, color="white")
        a.text(x, u + d / 2, str(d), ha="center", va="center", fontsize=9, color="white")
        a.text(x, CH[x][3] + 60, f"差し引き {u - d:+d}", ha="center", fontsize=9)
    a.set_xticks(list(xs))
    a.set_xticklabels([c[0] for c in CH], fontsize=8.5)
    a.set_ylim(0, 4700)
    a.set_ylabel("板無しから答えが変わった問いの数（白紙から、8,229 問）")
    a.set_title("上位 3 から板で選ぶと、外れに変える方が多い")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="upper left")
    fig.suptitle("仮の板の語を並べて一斉に走査する: 足した板と同じ高さ。決まりで書いた「出た語」との差は、並べ方ではなく別の所に有る", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49ze_row_board.png")
