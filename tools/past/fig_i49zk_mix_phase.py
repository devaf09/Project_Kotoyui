"""I49zk の図: 位相と振幅を直す解く形を、表と組んだ形に重ねた時の、正しい物の割合と、段ごとの当たり。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zk_mix_phase.py 出力.png
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
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

ROWS = [("解く形だけ・直さない", .4084, .3983, "#9ecae1"), ("解く形だけ・振幅だけ直す", .4220, .4201, "#9ecae1"),
        ("解く形だけ・位相＋振幅を直す", .4274, .4286, "#dd8452"), ("関係の表だけ", .4321, .4533, "#8c8c8c"),
        ("数えて選ぶ（表＋解く形）・直さない", .4443, .4638, "#4c72b0"), ("数えて選ぶ・位相＋振幅を直す", .4449, .4631, "#dd8452"),
        ("数えて選ぶ・位相＋振幅＋出た語", .4398, .4591, "#f5c9a8")]
LV = ["聞いた組\nそのもの", "片方だけ\n借りる", "述語側", "もう片方", "借りる", "答えられ\nない"]
# 青空の 3 つ組を先に: 表の当たり、解く形（直さない）、解く形（位相＋振幅）、問いの数
TB, SO, PH, N = [1532, 1334, 130, 521, 213, 0], [1191, 1159, 98, 593, 224, 13], [1348, 1258, 103, 600, 207, 11], [2697, 2929, 278, 1669, 614, 42]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for j, (start, col) in enumerate((("白紙から", 1), ("青空の 3 つ組を先に", 2))):
        a = ax[j]
        y = list(range(len(ROWS)))[::-1]
        vals = [r[col] for r in ROWS]
        a.barh(y, [v - .38 for v in vals], left=.38, color=[r[3] for r in ROWS], height=.62)
        for yy, v in zip(y, vals):
            a.text(v + .001, yy, f"{v:.3f}"[1:], va="center", fontsize=9)
        a.set_yticks(y)
        a.set_yticklabels([r[0] for r in ROWS] if j == 0 else [""] * len(ROWS), fontsize=9)
        a.set_xlim(.38, .48)
        a.set_xlabel("正しい物の割合")
        a.set_title(start)
        a.grid(axis="x", alpha=.3)
    a = ax[2]
    w = .27
    for k, (v, c, name) in enumerate(((TB, "#8c8c8c", "関係の表"), (SO, "#9ecae1", "解く形（直さない）"), (PH, "#dd8452", "解く形（位相＋振幅を直す）"))):
        xs = [i + (k - 1) * w for i in range(len(LV))]
        ys = [x / m for x, m in zip(v, N)]
        a.bar(xs, ys, w * .94, color=c, label=name)
        if k == 2:
            for x, yv, p, q in zip(xs, ys, PH, SO):
                a.text(x, yv + .008, f"{p - q:+d}", ha="center", fontsize=8, color="#b0562a")
    a.axvspan(-.5, 2.5, color="#f0f0f0", zorder=0)
    a.text(1, .665, "表の答えを使う段", ha="center", fontsize=9)
    a.text(4, .665, "解く形の答えを使う段", ha="center", fontsize=9)
    a.set_xticks(range(len(LV)))
    a.set_xticklabels(LV, fontsize=9)
    a.set_xlim(-.5, 5.5)
    a.set_ylim(0, .7)
    a.set_ylabel("その段の問いで当たった割合")
    a.set_title("段ごとの当たり（青空を先に。橙の上 = 増えた当たり）")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, .92))
    fig.suptitle("位相と振幅を直すと、表を使わない形は表に近づく。表と組んだ形の点は変わらない（増えた当たりは、表が勝つ段に集まる）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zk_mix_phase.png")
