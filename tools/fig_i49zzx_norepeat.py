"""I49zzx の図（2 枚目）: 「前と同じ 3 つ組は返さない」決まりを入れる前と後（run72 の合計の数だけ。5,987 問）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

N = 5987
FORMS = ("席ごと\n（人は引き継ぎ）", "動きを引いて、\n物は本体", "物を引いて、\n動きは本体")
PANELS = (("1. 3 席とも当たり", (193, 296, 237), (197, 296, 254), .06),
          ("2. 2 席以上当たり", (1011, 994, 888), (1027, 1028, 928), .22),
          ("3. 前の 3 つ組と同じ（オウム返し）", (304, 440, 681), (0, 0, 0), .14))


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for a, (title, before, after, ylim) in zip(ax, PANELS):
        for j, (vals, col, name) in enumerate(((before, "#bdbdbd", "決まり無し"), (after, "#3f73a8", "前と同じ物は返さない"))):
            for i, v in enumerate(vals):
                x = i + (j - .5) * .38
                a.bar(x, v / N, .38, color=col, label=name if i == 0 else None)
                a.text(x, v / N + ylim * .012, "%.3f" % (v / N), ha="center", fontsize=8.5)
        a.set_xticks(range(3)); a.set_xticklabels(FORMS, fontsize=9); a.set_ylim(0, ylim); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")
    ax[0].legend(fontsize=9, loc="upper left"); ax[0].set_ylabel("割合")
    fig.suptitle("前と同じ 3 つ組は返さない、と決めると、オウム返しが消えて、当たりは同じか少し上がる（I49zzx、run72）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzx_norepeat.png"), dpi=170)


if __name__ == "__main__":
    main()
