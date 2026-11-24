"""I49zv の図: 足した席を見せた時、正解の順位がどう移るか（波で解く形、同じ 1,654 問、元の順）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49zv_frame_who_wins.py 出力.png
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

RANKS = ["1 位", "2〜3 位（上位に居るのに負ける）", "それより下"]
COL = ["#55a868", "#dd8452", "#c9c9c9"]
# 席ごと（見せない, 見せる）の [1 位, 2〜3 位, それより下]
SEAT = [("が\n762 問", [313, 185, 264], [323, 175, 264]), ("を\n130 問", [33, 37, 60], [37, 35, 58]), ("どうした\n762 問", [55, 99, 608], [235, 127, 400])]
MOVE = [[372, 24, 5], [75, 226, 20], [148, 87, 697]]
NEAR = [("見せない", .3566, .1285), ("見せる", .3733, .1391)]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1.3, 1, .8]))
    a = ax[0]
    w = .38
    for i, (name, hid, shown) in enumerate(SEAT):
        for k, (vals, lab) in enumerate(((hid, "見せない"), (shown, "見せる"))):
            x, bottom = i + (k - .5) * w, 0
            for r in range(3):
                a.bar(x, vals[r], w * .94, bottom=bottom, color=COL[r], label=RANKS[r] if i == 0 and k == 0 else None)
                if vals[r] >= 30:
                    a.text(x, bottom + vals[r] / 2, str(vals[r]), ha="center", va="center", fontsize=10)
                bottom += vals[r]
            a.text(x, bottom + 8, lab, ha="center", fontsize=9)
    a.set_xticks(range(3))
    a.set_xticklabels([s[0] for s in SEAT])
    a.set_ylim(0, 900)
    a.set_ylabel("問いの数（足した席が有る枠の、隠した席ごと）")
    a.set_title("伸びたのは「どうした」だけ。順位の低かった正解が 1 位に上がった", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper center")
    a = ax[1]
    a.imshow(MOVE, cmap="Greens", vmin=0, vmax=700)
    for i in range(3):
        for j in range(3):
            a.text(j, i, str(MOVE[i][j]), ha="center", va="center", fontsize=13, color="w" if MOVE[i][j] > 400 else "k")
    short = ["1 位", "2〜3 位", "それより下"]
    a.set_xticks(range(3))
    a.set_xticklabels(short)
    a.set_yticks(range(3))
    a.set_yticklabels(short)
    a.set_xlabel("足した席を見せた時の順位")
    a.set_ylabel("見せない時の順位")
    a.set_title("2〜3 位で負けていた 321 問のうち、1 位に上がるのは 75 問", fontsize=11)
    a = ax[2]
    for i, (lab, x, y) in enumerate(NEAR):
        a.bar(i - .19, x, .36, color="#dd8452", label="1 位の語と正解の近さ" if i == 0 else None)
        a.bar(i + .19, y, .36, color="#8c8c8c", label="正解とほかの候補の近さ" if i == 0 else None)
        a.text(i - .19, x + .008, ("%.2f" % x)[1:], ha="center", fontsize=10)
        a.text(i + .19, y + .008, ("%.2f" % y)[1:], ha="center", fontsize=10)
    a.set_xticks(range(2))
    a.set_xticklabels(["%s\n負けた問い %d" % (n[0], q) for n, q in zip(NEAR, (321, 337))])
    a.set_ylim(0, .55)
    a.set_title("負け方は同じ（属性の似た語に負ける）", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper right")
    fig.suptitle("足した席を見せても、「似た語に負ける」問いは減らない。効くのは、に・で の語から動詞を当てる所", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zv_frame_who_wins.png")
