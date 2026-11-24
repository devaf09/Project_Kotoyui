"""I49z の図: 仮の板（薄れる速さと、板の枚数）で選び直した時の、正しい物の割合。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49z_board.py 出力.png
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

X = ["1 組で半分", "2 組", "4 組", "8 組", "薄れない\n（話ごとに空）"]
D = {
    "白紙から": (.4084, {
        "九割以内・1 枚の板": [.4154, .4058, .4008, .3948, .4020],
        "九割以内・席ごとの板": [.4168, .4100, .4032, .3968, .4043],
        "上位 3・1 枚の板": [.3459, .3243, .3061, .2842, .3265],
        "上位 3・席ごとの板": [.3737, .3564, .3377, .3201, .3672]}),
    "青空の 3 つ組を先に": (.3983, {
        "九割以内・1 枚の板": [.4115, .3991, .3928, .3850, .3949],
        "九割以内・席ごとの板": [.4117, .4038, .3982, .3904, .3986],
        "上位 3・1 枚の板": [.3494, .3315, .3151, .2975, .3361],
        "上位 3・席ごとの板": [.3632, .3474, .3356, .3178, .3659]}),
}
SEAT = {"白紙から": ([.505, .307, .413], [.528, .301, .421]), "青空の 3 つ組を先に": ([.477, .315, .403], [.528, .304, .403])}
STY = {"九割以内・1 枚の板": ("#4c72b0", "-", "o"), "九割以内・席ごとの板": ("#dd8452", "-", "s"),
       "上位 3・1 枚の板": ("#4c72b0", ":", "o"), "上位 3・席ごとの板": ("#dd8452", ":", "s")}


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    for j, start in enumerate(D):
        a = ax[j]
        base, rows = D[start]
        a.axhline(base, color="k", lw=1, ls="--", label=f"板無し {base:.3f}".replace("0.", "."))
        for name, ys in rows.items():
            c, ls, m = STY[name]
            a.plot(range(4), ys[:4], color=c, ls=ls, marker=m, label=name)
            a.plot([4], ys[4:], color=c, marker=m, ls="none", mfc="white")
        a.set_xticks(range(5))
        a.set_xticklabels(X, fontsize=9)
        a.set_ylim(.27, .43)
        a.set_ylabel("正しい物の割合")
        a.set_title(f"{start}: 速く薄れるほど良い")
        a.grid(alpha=.3)
        a.legend(fontsize=8, loc="lower left")
    a = ax[2]
    seats = ["だれが", "何を", "どうした"]
    w = .2
    for k, (start, (b, v)) in enumerate(SEAT.items()):
        xs = [i + (k - .5) * 2 * w for i in range(3)]
        a.bar([x - w / 2 for x in xs], b, w, color="#c9c9c9", label="板無し" if k == 0 else None)
        a.bar([x + w / 2 for x in xs], v, w, color=["#dd8452", "#55a868"][k], label=f"席ごとの板・1 組で半分（{start}）")
        for x, p, q in zip(xs, b, v):
            a.text(x + w / 2, q + .005, f"{(q - p) * 100:+.1f}", ha="center", fontsize=8)
    a.set_xticks(range(3))
    a.set_xticklabels(seats)
    a.set_ylim(.25, .6)
    a.set_title("席ごと: 上がるのは「だれが」")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("仮の板で選び直す: 直前の 1 組ほどの属性だけが効く。上位 3 から選び直すと崩れ、一位と並ぶ候補だけなら少し上がる", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49z_board.png")
