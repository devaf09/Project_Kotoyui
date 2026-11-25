"""I49zzi の図: 器（内面の波）に届く語の割合（run39 の集計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

SEATS = ["だれが", "何を", "どうした"]
KINDS = [.6994, .8000, .8102]
TIMES = [.7903, .9182, .9498]
TRI = [5, 156, 1198, 3105]


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4))
    x = range(3)
    a.bar([i - .2 for i in x], KINDS, .4, label="語の種類で数える", color="#9bb7d4")
    a.bar([i + .2 for i in x], TIMES, .4, label="出た回数で数える", color="#3f73a8")
    for i in x:
        a.text(i - .2, KINDS[i] + .01, "%.0f%%" % (KINDS[i] * 100), ha="center")
        a.text(i + .2, TIMES[i] + .01, "%.0f%%" % (TIMES[i] * 100), ha="center")
    a.set_xticks(list(x)); a.set_xticklabels(SEATS); a.set_ylim(0, 1.18)
    a.set_ylabel("属性を持つ語の割合"); a.set_title("席ごと: 器に届く語の割合")
    a.legend(loc="upper left", ncol=2, fontsize=9)
    tot = sum(TRI)
    b.bar(range(4), TRI, color=["#c8452c", "#d9a441", "#9bb7d4", "#3f73a8"])
    for i, v in enumerate(TRI):
        b.text(i, v + 40, "%d\n(%.1f%%)" % (v, v * 100 / tot), ha="center")
    b.set_xticks(range(4)); b.set_xticklabels(["0 席", "1 席", "2 席", "3 席"]); b.set_ylim(0, 3700)
    b.set_ylabel("3 つ組の数"); b.set_title("3 つ組ごと: 器に届く席の数（全 4,464 組）")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzi_reach.png")
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main()
