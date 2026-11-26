"""I49zzg の続きの図 2: 器の速い波と遅い波のずれと、正解の出方（run66 の数だけ。話の 2 つ目からの 3 つ組、17,961 問を、ずれの小さい順に 5 等分）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

GROUPS = ("小さい\n(.19〜.36)", "(.36〜.41)", "(.41〜.45)", "(.45〜.50)", "大きい\n(.50〜.69)")
HEARD = {"合計": [.3795, .3739, .37, .3572, .329], "並べ替え 1": [.371, .361, .371, .353, .334], "並べ替え 2": [.383, .378, .372, .357, .331],
         "並べ替え 3": [.388, .381, .361, .365, .322]}
HIT = {"本体の当たり（全部）": ([.4393, .4571, .4591, .4468, .4331], "#5a9a68"), "正解が聞いた語の時": ([.5385, .5577, .5658, .5269, .5237], "#3f73a8"),
       "正解が新しい語の時": ([.3786, .3971, .3964, .4023, .3886], "#c0504d")}


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, ys in HEARD.items():
        big = name == "合計"
        ax[0].plot(range(5), ys, color="#274f7f" if big else "#9db8d6", lw=2.6 if big else 1.2, marker="o", label=name, zorder=3 if big else 2)
    ax[0].set_ylim(.28, .42); ax[0].set_title("1. 正解が「この話でもう聞いた語」だった割合", fontsize=10.5); ax[0].legend(fontsize=9, loc="lower left")
    for name, (ys, col) in HIT.items():
        ax[1].plot(range(5), ys, color=col, lw=2.2, marker="o", label=name)
    ax[1].set_ylim(.3, .62); ax[1].set_title("2. 本体の当たり", fontsize=10.5); ax[1].legend(fontsize=9, loc="upper right")
    for a in ax:
        a.set_xticks(range(5)); a.set_xticklabels(GROUPS); a.set_xlabel("速い波と遅い波のずれ（小さい順に 5 等分）"); a.grid(alpha=.3)
    fig.suptitle("器の速い波と遅い波のずれが大きい時は、もう聞いた語が正解になりにくい（I49zzg の続き、run66）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzg_gap.png"), dpi=170)


if __name__ == "__main__":
    main()
