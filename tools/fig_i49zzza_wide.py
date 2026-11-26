"""I49zzza の図（2 枚目）: 候補を「その話の語」から「ここまでに聞いた全部の語」へ広げた時（持ち主の 3 つの文。席ごと 6,085 問。合計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

N = 6085
PANELS = (("1. だれが", (("引き継ぎ", 2814, 2814), ("遷移の数え", 2368, 1691), ("話の中の人から\n遷移で", 2950, 2950)), .56),
          ("2. 何を", (("遷移の数え", 1000, 441), ("話でもう出た語を\n除く", 990, 241)), .24),
          ("3. どうした", (("遷移の数え", 824, 343), ("話でもう出た語を\n除く", 1200, 333)), .24))


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=(3, 2, 2)))
    for a, (title, rows, ylim) in zip(ax, PANELS):
        for i, (name, near, wide) in enumerate(rows):
            for j, (v, col, lab) in enumerate(((near, "#bdbdbd", "候補はその話の語（約 20 語、当てずっぽう .047）"), (wide, "#3f73a8", "候補は聞いた全部の語（約 2,000 語、当てずっぽう .001）"))):
                x = i + (j - .5) * .38
                a.bar(x, v / N, .38, color=col, label=lab if i == 0 else None)
                a.text(x, v / N + ylim * .012, "%.3f" % (v / N), ha="center", fontsize=8.5)
        a.set_xticks(range(len(rows))); a.set_xticklabels([r[0] for r in rows], fontsize=9); a.set_ylim(0, ylim); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")
    ax[0].set_ylabel("当たりの割合")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=1)
    fig.suptitle("候補を広げると、「出た語を除く」の伸びは消える。「話の中の人を持つ」は残る（I49zzza、広い候補）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzza_wide.png"), dpi=170)


if __name__ == "__main__":
    main()
