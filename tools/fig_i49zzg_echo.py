"""I49zzg の続きの図: 器で選び直した先が、この話でもう聞いた語だったかどうかで分けた、当たりに変わった数と外れに変わった数（run60 の合計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

WIDTHS = ("〜0.01", "〜0.05", "〜0.1", "〜0.2", "〜0.5")
# (当たりに変わった, 外れに変わった)
DATA = {"器に近い候補・聞いた語を選んだ": [(22, 38), (100, 169), (175, 323), (307, 499), (450, 831)],
        "器に近い候補・まだ聞いていない語を選んだ": [(33, 24), (112, 86), (182, 152), (273, 269), (328, 395)],
        "器から遠い候補・聞いた語を選んだ": [(6, 19), (57, 87), (115, 191), (185, 352), (286, 648)],
        "器から遠い候補・まだ聞いていない語を選んだ": [(57, 32), (196, 101), (307, 214), (494, 362), (658, 643)]}


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True)
    xs = range(len(WIDTHS))
    for a, (name, rows) in zip(ax, DATA.items()):
        a.bar([x - .2 for x in xs], [r[0] for r in rows], .4, color="#3f73a8", label="当たりに変わった")
        a.bar([x + .2 for x in xs], [r[1] for r in rows], .4, color="#c0504d", label="外れに変わった")
        a.set_xticks(list(xs)); a.set_xticklabels(WIDTHS); a.set_xlabel("同点と見なす幅"); a.set_title(name, fontsize=10.5); a.grid(alpha=.3, axis="y")
    ax[0].legend(fontsize=9, loc="upper left"); ax[0].set_ylabel("問いの数")
    fig.suptitle("器で選び直した時の損は、「この話でもう聞いた語」を選んだ時に片寄る（I49zzg の続き、run60。3 通りの並べ替えの合計 19,929 問）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzg_echo.png"), dpi=170)


if __name__ == "__main__":
    main()
