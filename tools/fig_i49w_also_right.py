"""I49w の図: 「正しい物」と「正しいと言える物」を積み上げた当たり。全体と席ごと。

数は PC で数えた集計値をそのまま置いている（集まりは「話と青空」）。
使い方: python tools/fig_i49w_also_right.py 出力.png
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

GROUPS = ["全体", "だれが", "何を", "どうした"]
# (正しい物, 合わせて)
D = {
    "白紙から": {"関係の表": [(.4321, .5893), (.5727, .6504), (.3088, .5509), (.4149, .5665)],
             "解く・属性": [(.4084, .5421), (.5053, .5658), (.3073, .5108), (.4127, .5498)]},
    "青空の 3 つ組を先に": {"関係の表": [(.4533, .6148), (.5625, .6475), (.3354, .5786), (.4619, .6183)],
                    "解く・属性": [(.3983, .5155), (.4772, .5392), (.3153, .4896), (.4025, .5177)]},
}
COL = {"関係の表": ("#4c72b0", "#a9bfe0"), "解く・属性": ("#dd8452", "#f2c9ae")}


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    w = .38
    for j, start in enumerate(D):
        a = ax[j]
        for k, (form, rows) in enumerate(D[start].items()):
            xs = [i + (k - .5) * w for i in range(len(GROUPS))]
            a.bar(xs, [r[0] for r in rows], w, color=COL[form][0], label=f"{form}: 正しい物")
            a.bar(xs, [r[1] - r[0] for r in rows], w, bottom=[r[0] for r in rows], color=COL[form][1],
                  label=f"{form}: 正しいと言える物")
            for x, r in zip(xs, rows):
                a.text(x, r[0] / 2, f"{r[0]:.2f}"[1:], ha="center", va="center", fontsize=9, color="white")
                a.text(x, r[1] + .008, f"{r[1]:.2f}"[1:], ha="center", fontsize=9)
        a.set_xticks(range(len(GROUPS)))
        a.set_xticklabels(GROUPS)
        a.set_ylim(0, .8)
        a.set_title(start)
        a.grid(axis="y", alpha=.3)
        a.legend(loc="upper right", fontsize=8, ncol=2)
    ax[0].set_ylabel("問いの割合（上の数は、合わせた割合）")
    fig.suptitle("正しい物と、正しいと言える物: 外れの 4 分の 1 ほどは、別の所で使われていた答えだった", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49w_also_right.png")
