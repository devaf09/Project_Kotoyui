"""I49zzq の図: 状況波を位相の形に置き換えた時、位相の次元ごとに元の形とどれだけ同じになるか（MRMP の集計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

DIMS = [16, 64, 256, 1024, 4096]
DATA = {"同じ話し手が続ける時（4,370 組）": dict(元=.5822, 選ぶ=[.5483, .5812, .5789, .5794, .5822], 同じ=[.7291, .838, .9382, .9817, .9954],
                                    向き1=[.8929, .9747, .9973, .9998, 1.0], 向き3=[.7007, .88, .9896, .9992, 1.0]),
        "次の話し手が受ける時（94,738 組）": dict(元=.5433, 選ぶ=[.5256, .5385, .5448, .5429, .5431], 同じ=[.7416, .8459, .9381, .983, .9964],
                                     向き1=[.8236, .9568, .9942, .9994, 1.0], 向き3=[.3088, .7937, .9701, .9985, .9999])}
MIX = [.1277, .0374, .009, .0023, .0006]


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    for (title, d), col in zip(DATA.items(), ("#3f73a8", "#d9a441")):
        ax[0].plot(DIMS, d["同じ"], color=col, lw=2.2, marker="o", label=title)
        ax[1].plot(DIMS, d["向き1"], color=col, lw=2.2, marker="o", label=title + "・一番強い向き")
        ax[1].plot(DIMS, d["向き3"], color=col, lw=1.4, ls="--", marker="x", label=title + "・3 番目の向き")
        ax[2].plot(DIMS, d["選ぶ"], color=col, lw=2.2, marker="o", label=title + "・位相の形")
        ax[2].axhline(d["元"], color=col, lw=1.2, ls=":", label=title + "・元の形")
    for x, y in zip(DIMS, DATA["同じ話し手が続ける時（4,370 組）"]["同じ"]):
        ax[0].annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax[0].set_title("1. 元の形と同じ方を選んだ割合", fontsize=11); ax[0].set_ylim(.6, 1.05)
    ax[1].set_title("2. 強い向きが、元の形の向きと重なる度合い（1 が同じ）", fontsize=11); ax[1].set_ylim(.2, 1.05)
    ax[2].set_title("3. 2 つから選ぶ当たり（当てずっぽう .5）", fontsize=11); ax[2].set_ylim(.5, .6)
    for a in ax:
        a.set_xscale("log", base=2); a.set_xticks(DIMS); a.set_xticklabels([str(x) for x in DIMS]); a.set_xlabel("位相の次元（周波数の本数）")
        a.grid(alpha=.3); a.legend(fontsize=8, loc="lower right")
    fig.suptitle("状況波を位相の形に置き換える: 位相の次元ごとに、元の形（数えた表）とどれだけ同じ働きになるか（I49zzq、話し手の印の有る雑談）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzq_phase.png"), dpi=170)


if __name__ == "__main__":
    main()
