"""I49zl の図: 誤差で直した後も、正解が 2〜3 位で負ける問いで、1 位は正解と属性の似た語か。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zl_who_wins_after.py 出力.png
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

FORMS = ["直さない", "振幅だけ直す", "位相＋振幅を直す"]
COL = ["#8c8c8c", "#9ecae1", "#dd8452"]
LOST = {"白紙から": [.2695, .2626, .2418], "青空の 3 つ組を先に": [.3004, .2932, .2693]}
# 白紙から: 席ごとの（1 位と正解の近さ、正解とほかの候補の近さ）
NEAR = {"全体": ([.368, .370, .364], [.133, .131, .128]), "だれが": ([.514, .511, .498], [.127, .127, .120]),
        "何を": ([.166, .171, .164], [.104, .102, .100]), "どうした": ([.306, .313, .316], [.160, .157, .159])}


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
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
    a.set_title("正解が 2〜3 位で負ける問い: 少し減る")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="lower right")
    a = ax[1]
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
    a.set_title("負けた相手は、どの形でも正解と属性の似た語（数字は 位相＋振幅 での倍率）")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="upper right")
    fig.suptitle("誤差で直しても「属性が近い語に負ける」形は残る。負ける問いの数が 1 割ほど減るだけで、負け方は変わらない", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zl_who_wins_after.png")
