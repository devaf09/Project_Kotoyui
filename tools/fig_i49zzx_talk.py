"""I49zzx の図（3 枚目）: 話の形（出た動きを除く、話の中の人から選ぶ）を返事の組み方に入れた時（run73 の合計の数だけ。5,987 問）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

N = 5987
FORMS = (("今までの一番\n（動きを引いて、物は本体）", "#bdbdbd"), ("人は話の中の人から", "#9cc7a5"), ("動きは出た語を除く", "#9dbbd8"), ("両方", "#3f73a8"))
SEAT = {"だれが": (2780, 2924, 2780, 2924), "何を": (1024, 1014, 1229, 1234), "どうした": (943, 943, 1519, 1519)}
WHOLE = {"3 席とも当たり": (296, 315, 426, 466), "2 席以上当たり": (1028, 1036, 1418, 1419), "前に丸ごと聞いた\n3 つ組だった": (916, 1122, 748, 863)}
REAL_HEARD = 764


def bars(a, data, title, ylim):
    w = .8 / len(FORMS)
    for j, (name, col) in enumerate(FORMS):
        for i, vals in enumerate(data.values()):
            x = i + (j - 1.5) * w
            a.bar(x, vals[j] / N, w, color=col, label=name.replace("\n", "") if i == 0 else None)
            a.text(x, vals[j] / N + ylim * .012, ("%.3f" % (vals[j] / N))[1:], ha="center", fontsize=8.5)
    a.set_xticks(range(len(data))); a.set_xticklabels(list(data)); a.set_ylim(0, ylim); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    bars(ax[0], SEAT, "1. 席ごとの当たり", .62)
    ax[0].legend(fontsize=9, loc="upper right"); ax[0].set_ylabel("割合")
    bars(ax[1], WHOLE, "2. 丸ごとの当たりと、まとまり", .30)
    ax[1].plot([1.55, 2.45], [REAL_HEARD / N] * 2, color="#333333", ls="--", lw=1)
    ax[1].text(2.0, .225, "点線: 本物の次の 3 つ組が\n前に聞いた物 .128", fontsize=8.5, ha="center")
    fig.suptitle("その話で出た動きを返さない、と決めると、丸ごとの当たりが .049 から .078 に上がる（I49zzx、run73）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzx_talk.png"), dpi=170)


if __name__ == "__main__":
    main()
