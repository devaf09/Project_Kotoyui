"""I49zzt の図: 言葉の 3 つ組で、掛け合わせた波（数える／外れた時だけ動かす）を本体と比べる（run62 の合計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

KINDS = ("全部\n（19,977 問）", "2 語の組を\n前に聞いた（7,276 問）", "2 語の組は\n初めて（12,701 問）")
N = (19977, 7276, 12701)
ROWS = (("当てずっぽう", "#bdbdbd", (1099.6, 393.5, 706.0)), ("本体（今の土台）", "#5a9a68", (8978, 3706, 5272)),
        ("掛け合わせて数える", "#d9a441", (4158, 2748, 1410)), ("外れた時だけ動かす（大きさ .1、1 回）", "#6a93c2", (3267, 2260, 1007)),
        ("外れた時だけ動かす（大きさ .3、4 回）", "#274f7f", (2922, 1994, 928)))


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, a = plt.subplots(figsize=(11, 5.2))
    w = .8 / len(ROWS)
    for j, (name, col, vals) in enumerate(ROWS):
        xs = [i + (j - (len(ROWS) - 1) / 2) * w for i in range(3)]
        ys = [v / n for v, n in zip(vals, N)]
        a.bar(xs, ys, w, color=col, label=name)
        for x, y in zip(xs, ys):
            a.text(x, y + .008, "%.2f" % y, ha="center", fontsize=8)
    a.set_xticks(range(3)); a.set_xticklabels(KINDS); a.set_ylabel("当たりの割合"); a.set_ylim(0, .62); a.grid(alpha=.3, axis="y")
    a.legend(fontsize=9, loc="upper right")
    a.set_title("言葉の 3 つ組では、掛け合わせた波だけでは本体に届かない（I49zzt、run62。持ち主の 3 つの文、3 通りの並べ替えの合計）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzt_triple.png"), dpi=170)


if __name__ == "__main__":
    main()
