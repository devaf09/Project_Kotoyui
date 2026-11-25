"""I49zh の図: いいとこ取りの形に振幅を直す形を入れた時の、段ごとの当たり（表、解く形、振幅を直した解く形）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zh_mix_amp.py 出力.png
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

LV = ["聞いた組\nそのもの", "片方だけ\n借りる", "述語側", "もう片方", "借りる", "答えられ\nない"]
# 始め方: (表の当たり, 解く形の当たり, 振幅を直した解く形の当たり, 問いの数, 全体の正しい物: 表だけ・数えて選ぶ・振幅を直す)
D = {
    "白紙から": ([1476, 1079, 51, 616, 334, 0], [1282, 972, 49, 670, 349, 39], [1341, 1001, 50, 701, 335, 45],
              [2585, 2132, 99, 2111, 1004, 298], (.4321, .4443, .4466)),
    "青空の 3 つ組を先に": ([1532, 1334, 130, 521, 213, 0], [1191, 1159, 98, 593, 224, 13], [1287, 1241, 100, 598, 216, 15],
                    [2697, 2929, 278, 1669, 614, 42], (.4533, .4638, .4641)),
}


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    w = .27
    for j, start in enumerate(D):
        a = ax[j]
        tb, so, am, n, tot = D[start]
        for k, (v, c, name) in enumerate(((tb, "#8c8c8c", "関係の表"), (so, "#9ecae1", "解く形（振幅を直さない）"), (am, "#dd8452", "解く形（振幅を直す）"))):
            ys = [x / m for x, m in zip(v, n)]
            xs = [i + (k - 1) * w for i in range(len(LV))]
            a.bar(xs, ys, w * .94, color=c, label=name)
            if k == 2:
                for x, y, p, q in zip(xs, ys, am, so):
                    a.text(x, y + .008, f"{p - q:+d}", ha="center", fontsize=8, color="#b0562a")
        a.axvspan(-.5, 2.5, color="#f0f0f0", zorder=0)
        a.text(1, .665, "表が勝つ段（表の答えを使う）", ha="center", fontsize=9)
        a.text(4, .665, "解く形が勝つ段（解く形の答えを使う）", ha="center", fontsize=9)
        a.set_xticks(range(len(LV)))
        a.set_xticklabels(LV, fontsize=9)
        a.set_xlim(-.5, 5.5)
        a.set_ylim(0, .7)
        a.set_ylabel("その段の問いで当たった割合")
        a.set_title("%s: 全体は 表だけ %s、数えて選ぶ %s、振幅を直す %s" % ((start,) + tuple(f"{x:.3f}"[1:] for x in tot)), fontsize=11)
        a.grid(axis="y", alpha=.3)
        a.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, .92))
    fig.suptitle("振幅を直して解く形が強くなった分（橙の上の数 = 増えた当たり）は、表がもともと勝っている段に多く、全体の点には出ない", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zh_mix_amp.png")
