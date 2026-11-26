"""I49zzz の図: 次の行の語は、話の中でもう出た語か、初めての語か（持ち主の 3 つの文。席ごと 4,016 組。合計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

KINDS = (("続く（前の行と同じ語）", "#5a9a68"), ("1 つ前の別の語へ戻る", "#9cc7a5"), ("もっと前に出た語へ戻る", "#d9c58a"), ("初めての語", "#3f73a8"))
SEATS = {"だれが": (1861, 856, 365, 934), "何を": (369, 182, 289, 3176), "どうした": (79, 55, 190, 3692)}
RUNS = {"1 行": (794, 661, 312, 591), "2 行": (362, 143, 33, 166), "3 行": (220, 33, 14, 63), "4 行以上": (485, 19, 6, 114)}


def stack(a, data, title):
    for i, vals in enumerate(data.values()):
        n, bottom = sum(vals), 0.0
        for (name, col), v in zip(KINDS, vals):
            a.bar(i, v / n, .62, bottom=bottom, color=col, label=name if i == 0 else None)
            if v / n >= .045:
                a.text(i, bottom + v / n / 2, "%.2f" % (v / n), ha="center", va="center", fontsize=9, color="white" if col in ("#3f73a8", "#5a9a68") else "black")
            bottom += v / n
    a.set_xticks(range(len(data))); a.set_xticklabels(list(data)); a.set_ylim(0, 1); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    stack(ax[0], SEATS, "1. 次の行の語は、どこから来るか（席ごと 4,016 組）")
    ax[0].set_ylabel("割合")
    stack(ax[1], RUNS, "2. だれが: 今の人が何行続いているか、ごと（2,358 / 704 / 330 / 624 組）")
    ax[1].legend(fontsize=9, loc="lower center", bbox_to_anchor=(.5, -.32), ncol=2)
    fig.suptitle("人は話の中に居る人から来る（8 割）。物と動きは、話の中でまだ出ていない語から来る（8〜9 割）（I49zzz）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzz_memory.png"), dpi=170)


if __name__ == "__main__":
    main()
