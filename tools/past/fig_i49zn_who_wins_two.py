"""I49zn の図: 属性＋固有の位相の形でも、正解が 2〜3 位で負ける問いで、1 位は正解と属性の似た語か。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zn_who_wins_two.py 出力.png
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

FORMS = ["属性だけ", "属性＋固有の位相", "属性＋固有の位相＋振幅"]
COL = ["#8c8c8c", "#9ecae1", "#dd8452"]
LOST = {"白紙から": [.2695, .2627, .2562], "青空の 3 つ組を先に": [.3004, .2902, .2937]}
# 白紙から: 席ごとの（1 位と正解の近さ、正解とほかの候補の近さ）
NEAR = {"全体": ([.368, .366, .363], [.133, .130, .130]), "だれが": ([.514, .519, .514], [.127, .125, .125]),
        "何を": ([.166, .162, .169], [.104, .101, .103]), "どうした": ([.306, .311, .309], [.160, .157, .156])}
# 負けた問いの数（席ごと。白紙から ／ 青空を先に）
CNT = {"白紙から": {"だれが": [930, 859, 828], "何を": [624, 614, 596], "どうした": [664, 689, 684]},
       "青空を先に": {"だれが": [1003, 922, 913], "何を": [672, 653, 680], "どうした": [797, 813, 824]}}


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1, 1.5, 1.5]))
    a = ax[0]
    w = .26
    for k, name in enumerate(FORMS):
        xs = [i + (k - 1) * w for i in range(2)]
        ys = [LOST[s][k] for s in LOST]
        a.bar(xs, ys, w * .94, color=COL[k], label=name)
        for x, y in zip(xs, ys):
            a.text(x, y + .004, f"{y * 100:.1f}%", ha="center", fontsize=9)
    a.set_xticks(range(2))
    a.set_xticklabels(list(LOST))
    a.set_ylim(0, .36)
    a.set_ylabel("全部の問いに占める割合")
    a.set_title("正解が 2〜3 位で負ける問い: ほとんど減らない")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="lower right")
    a = ax[1]
    keys = [(st, se) for st in CNT for se in CNT[st]]
    w = .26
    for k, name in enumerate(FORMS):
        xs = [i + (k - 1) * w for i in range(len(keys))]
        ys = [CNT[st][se][k] for st, se in keys]
        a.bar(xs, ys, w * .94, color=COL[k], label=name)
    for i, (st, se) in enumerate(keys):
        d = CNT[st][se][1] - CNT[st][se][0]
        a.text(i, max(CNT[st][se]) + 15, "%+d" % d, ha="center", fontsize=9)
    a.set_xticks(range(len(keys)))
    a.set_xticklabels(["%s\n%s" % (se, st) for st, se in keys], fontsize=9)
    a.set_ylim(0, 1150)
    a.set_ylabel("負けた問いの数")
    a.set_title("席ごと: 「だれが」は減り、「どうした」は増える（数字は 固有を足した時の増減）")
    a.grid(axis="y", alpha=.3)
    a = ax[2]
    seats = list(NEAR)
    w = .13
    for k, name in enumerate(FORMS):
        xs = [i + (k - 1) * 2 * w for i in range(len(seats))]
        a.bar([x - w / 2 for x in xs], [NEAR[s][0][k] for s in seats], w * .94, color=COL[k], label=name + "・1 位の語")
        a.bar([x + w / 2 for x in xs], [NEAR[s][1][k] for s in seats], w * .94, color=COL[k], alpha=.35, hatch="//", edgecolor="white",
              label="ほかの候補（比べ役）" if k == 2 else None)
    for i, s in enumerate(seats):
        a.text(i, max(NEAR[s][0]) + .015, "%.1f 倍" % (NEAR[s][0][2] / NEAR[s][1][2]), ha="center", fontsize=9)
    a.set_xticks(range(len(seats)))
    a.set_xticklabels(seats)
    a.set_ylim(0, .62)
    a.set_ylabel("正解との属性の近さ（平均、白紙から）")
    a.set_title("負けた相手は、どの形でも正解と属性の似た語（数字は ＋振幅 での倍率）")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="upper right")
    fig.suptitle("固有の位相を足しても「属性が近い語に負ける」形は残る。負ける問いは 27.0% から 26.3%、負けた相手との近さは同じ", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zn_who_wins_two.png")
