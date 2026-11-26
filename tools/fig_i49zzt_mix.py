"""I49zzt の図 2: 本体と、属性から始めて数える形を組み合わせた時（run64 の合計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

KINDS = ("全部", "2 語の組を\n前に聞いた", "2 語の組は\n初めて")
# (問い, 両方当たり, 本体だけ当たり, 掛け合わせだけ当たり)
OVER = ((19977, 5135, 3843, 1158), (7276, 2893, 813, 324), (12701, 2242, 3030, 834))
LIMS = ("〜0.01", "〜0.05", "〜0.1", "〜0.2", "〜0.5")
UP = (64, 265, 430, 683, 975)
DOWN = (75, 260, 486, 803, 1269)


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    a = ax[0]
    parts = (("両方が当たり", "#5a9a68", 1), ("本体だけが当たり", "#a8cfb0", 2), ("掛け合わせだけが当たり", "#c0504d", 3))
    bottom = [0.0] * 3
    for name, col, k in parts:
        ys = [r[k] / r[0] for r in OVER]
        a.bar(range(3), ys, .55, bottom=bottom, color=col, label=name)
        for i, y in enumerate(ys):
            a.text(i, bottom[i] + y / 2, "%.2f" % y, ha="center", va="center", fontsize=9)
        bottom = [b + y for b, y in zip(bottom, ys)]
    a.set_xticks(range(3)); a.set_xticklabels(KINDS); a.set_ylim(0, .75); a.set_ylabel("問いに対する割合"); a.grid(alpha=.3, axis="y")
    a.set_title("1. どちらが当たるか（積み上げの高さが、どちらかが当たる割合）", fontsize=10.5); a.legend(fontsize=9, loc="upper right")
    a = ax[1]
    xs = range(len(LIMS))
    a.bar([x - .2 for x in xs], UP, .4, color="#3f73a8", label="当たりに変わった")
    a.bar([x + .2 for x in xs], DOWN, .4, color="#c0504d", label="外れに変わった")
    a.set_xticks(list(xs)); a.set_xticklabels(LIMS); a.set_xlabel("本体の 1 位と同点と見なす幅"); a.set_ylabel("問いの数（19,977 問のうち）"); a.grid(alpha=.3, axis="y")
    a.set_title("2. 同点に近い候補の中から、掛け合わせの点で選び直した時", fontsize=10.5); a.legend(fontsize=9, loc="upper left")
    fig.suptitle("本体と、属性から始めて数える形の組み合わせ: 当たる問いは一部ずれているが、同点の時に選び直すだけでは得にならない（I49zzt、run64）", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzt_mix.png"), dpi=170)


if __name__ == "__main__":
    main()
