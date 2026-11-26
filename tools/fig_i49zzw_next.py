"""I49zzw の図: 前の 3 つ組から、次の 3 つ組の語を当てる（run69 の合計の数だけ。席ごと 5,987 問）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

SEATS = ("だれが", "何を", "どうした")
ROWS = (("当てずっぽう", "#bdbdbd"), ("引き継ぎ（前の同じ席の語）", "#5a9a68"), ("遷移の数え", "#d9a441"), ("遷移の波（語板の属性から始めた位相）", "#3f73a8"))
ALL = {"だれが": (285.2, 2780, 2344, 2377), "何を": (285.2, 572, 999, 1022), "どうした": (285.2, 106, 851, 943)}
N = 5987
CHG = {"だれが": (3207, 142.0, 0, 450, 650), "何を": (5415, 257.6, 0, 697, 736), "どうした": (5881, 280.4, 0, 833, 920)}


def bars(a, vals, title, ylim):
    w = .8 / len(ROWS)
    for j, (name, col) in enumerate(ROWS):
        xs = [i + (j - 1.5) * w for i in range(3)]
        ys = [v[j] for v in vals]
        a.bar(xs, ys, w, color=col, label=name)
        for x, y in zip(xs, ys):
            a.text(x, y + ylim * .012, "%.2f" % y, ha="center", fontsize=8.5)
    a.set_xticks(range(3)); a.set_xticklabels(SEATS); a.set_ylim(0, ylim); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    bars(ax[0], [[v / N for v in ALL[s]] for s in SEATS], "1. 全部の問い（席ごと 5,987 問）", .56)
    ax[0].legend(fontsize=9, loc="upper right"); ax[0].set_ylabel("当たりの割合")
    bars(ax[1], [[v / CHG[s][0] for v in CHG[s][1:]] for s in SEATS], "2. 前と違う語に替わる時だけ（だれが 3,207、何を 5,415、どうした 5,881 問）", .28)
    fig.suptitle("前の 3 つ組から、次の 3 つ組の語を当てる: 人は引き継ぎ、物と動きは遷移で、当てずっぽうの 3 倍ほど（I49zzw、run69）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzw_next.png"), dpi=170)


if __name__ == "__main__":
    main()
