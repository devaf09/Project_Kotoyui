"""I49zzu の図: 相手の行の語から、返事の行の語が当たるか（run65、雑談の集計の数だけ。2 つから選ぶ。5 割なら、つながりは無い）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

KINDS = ("全部", "相手の行にも\n有った語（繰り返し）", "相手の行に\n無かった語（別の語へ）")
DATA = {"相手 → 返事（確かめる組 6,342）": ((.573, .9783, .5407), (.4927, .9533, .4871), (9051, 669, 8382)),
        "同じ話し手の続き（確かめる組 1,478）": ((.5294, .9714, .5072), (.5178, .6667, .5153), (731, 35, 696))}


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a, (name, (real, swap, n)) in zip(ax, DATA.items()):
        a.bar([i - .2 for i in range(3)], real, .4, color="#3f73a8", label="本物の相手の行")
        a.bar([i + .2 for i in range(3)], swap, .4, color="#bdbdbd", label="相手の行を別の所の行と入れ替えた時")
        for i in range(3):
            a.text(i - .2, real[i] + .012, "%.2f" % real[i], ha="center", fontsize=9)
            a.text(i + .2, swap[i] + .012, "%.2f" % swap[i], ha="center", fontsize=9)
            a.text(i, .03, "%s 問" % format(n[i], ","), ha="center", fontsize=8.5, color="w")
        a.axhline(.5, color="k", lw=1, ls=":")
        a.set_xticks(range(3)); a.set_xticklabels(KINDS); a.set_ylim(0, 1.08); a.set_title(name, fontsize=10.5); a.grid(alpha=.3, axis="y")
    ax[0].set_ylabel("2 つから選んだ時の当たり（点線は 5 割）"); ax[1].legend(fontsize=9, loc="upper right")
    fig.suptitle("相手の行の語と返事の行の語には、弱いが数えられるつながりが有る（I49zzu、雑談 960 会話のうち 96 会話を外して確かめ）", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzu_reply.png"), dpi=170)


if __name__ == "__main__":
    main()
