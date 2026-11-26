"""I49zzza の図: 話の中でもう出た語を「持つ／遠ざける」形で、次の語を当てる（持ち主の 3 つの文。席ごと 6,085 問。合計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

N = 6085
PANELS = (("1. だれが: 話の中に居る人を持つ", (("当てずっぽう", 288.7, "#bdbdbd"), ("引き継ぎ", 2814, "#5a9a68"), ("遷移の数え", 2368, "#d9a441"),
                                        ("続いた数で\n選ぶ", 2814, "#9cc7a5"), ("話の中の人から\n遷移で", 2950, "#3f73a8")), .56),
          ("2. 何を: 話でもう出た語を除く", (("当てずっぽう", 288.7, "#bdbdbd"), ("当てずっぽう\n（除いた後）", 296.0, "#dddddd"), ("遷移の数え", 1000, "#d9a441"),
                                      ("もう出た語を\n除く", 990, "#3f73a8")), .24),
          ("3. どうした: 話でもう出た語を除く", (("当てずっぽう", 288.7, "#bdbdbd"), ("当てずっぽう\n（除いた後）", 355.3, "#dddddd"), ("遷移の数え", 824, "#d9a441"),
                                        ("もう出た語を\n除く", 1200, "#3f73a8")), .24))


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw=dict(width_ratios=(5, 4, 4)))
    for a, (title, rows, ylim) in zip(ax, PANELS):
        for i, (name, v, col) in enumerate(rows):
            a.bar(i, v / N, .62, color=col)
            a.text(i, v / N + ylim * .012, "%.3f" % (v / N), ha="center", fontsize=9)
        a.set_xticks(range(len(rows))); a.set_xticklabels([r[0] for r in rows], fontsize=9); a.set_ylim(0, ylim); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")
    ax[0].set_ylabel("当たりの割合")
    fig.suptitle("話の中でもう出た語の使い方を席で変える: 動きは、出た語を除くと上がる。人と物は、ほとんど変わらない（I49zzza）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzza_state.png"), dpi=170)


if __name__ == "__main__":
    main()
