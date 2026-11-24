"""I49x の説明の図: 「足す」と「掛ける」で、なぜ答えがあまり変わらないか。数は説明のために作った例（数えた値ではない）。

使い方: python tools/fig_i49x_toy.py 出力.png
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

CANDS = ["食べた", "取った", "歩いた", "泳いだ"]
CAT = [.60, .50, .70, .10]      # 「猫」から見た点
FISH = [.60, .55, .10, .70]     # 「魚」から見た点


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    xs = range(len(CANDS))
    a = ax[0]
    a.bar([x - .2 for x in xs], CAT, .4, color="#4c72b0", label="「猫」と合う度合い")
    a.bar([x + .2 for x in xs], FISH, .4, color="#55a868", label="「魚」と合う度合い")
    a.set_title("① 語ごとの点（猫が 魚を ？）")
    a.legend(fontsize=9, loc="upper center")
    a.set_ylim(0, 1.0)
    add = [c + f for c, f in zip(CAT, FISH)]
    mul = [c * f for c, f in zip(CAT, FISH)]
    for a, vals, title in ((ax[1], add, "② 足す（今までの形）"), (ax[2], mul, "③ 掛ける（収束の形）")):
        top = max(vals)
        a.bar(xs, vals, .6, color=["#dd8452" if v == top else "#c9c9c9" for v in vals])
        for x, v in zip(xs, vals):
            a.text(x, v + top * .02, f"{v:.2f}", ha="center", fontsize=10)
        a.set_title(title)
        a.set_ylim(0, top * 1.25)
    ax[1].annotate("片方としか合わない語も\nそこそこ残る", (2.5, .8), (2.0, 1.25), ha="center", fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="gray"))
    ax[2].annotate("片方としか合わない語は\nほぼ消える", (2.5, .07), (2.3, .25), ha="center", fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="gray"))
    ax[2].annotate("でも 1 位と 2 位の\n並びは同じまま", (.5, .33), (1.2, .40), ha="center", fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="gray"))
    for a in ax:
        a.set_xticks(list(xs))
        a.set_xticklabels(CANDS)
        a.grid(axis="y", alpha=.3)
    fig.suptitle("作った例: 掛けると「片方だけ合う語」は消えるが、落としているのは「両方と合う語どうし」の選び分け", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49x_toy.png")
