"""I49zzs の図: 掛け合わせた波で 3 つ組を持つ、外れた時だけ位相を動かすと見ていない組まで当たる（run59 の数。小さい世界）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

WAYS = ("2 つずつの数え", "掛け合わせて\n数える", "外れた時だけ動かす\n（同じ表）", "外れた時だけ動かす\n（席ごとの表）")
COL = ("#9a9a9a", "#d9a441", "#3f73a8", "#5a9a68")
RECALL = {"猫と犬（2 つ）": [.5, 1, 1, 1], "10 進": [.1, 1, 1, 1], "16 進": [.0625, 1, 1, 1]}          # 全部の組を聞いた時（4096 次元）
UNSEEN = {"10 進、足し算": [0, .0455, 1.0, .3182], "16 進、足し算": [0, .0377, .9623, .2075],          # 2 割を外した時（4096 次元、動かす大きさ .3）
          "10 進、でたらめ": [0, .0909, .1818, .1818], "16 進、でたらめ": [.0377, .0943, .1132, .0566]}
CHANCE = {"10 進、足し算": .1, "16 進、足し算": .0625, "10 進、でたらめ": .1, "16 進、でたらめ": .0625}
ETAS = [.03, .1, .3]
BY_ETA = {"10 進、2 割を外す": ([.3636, .6364, 1.0], [.0909, .0909, .1818]), "10 進、5 割を外す": ([.3, .48, .56], [.06, .06, .1]),
          "16 進、2 割を外す": ([.2264, .6981, .9623], [.0377, .0566, .1132]), "16 進、5 割を外す": ([.186, .4419, .814], [.0233, .0465, .0233])}
DIMS = [256, 1024, 4096]
BY_DIM = {"10 進、2 割を外す": [.4545, .4545, .6364], "10 進、5 割を外す": [.2, .36, .48],
          "16 進、2 割を外す": [.2264, .434, .6981], "16 進、5 割を外す": [.1705, .1783, .4419]}
LINE = {"10 進、2 割を外す": ("#3f73a8", "-"), "10 進、5 割を外す": ("#3f73a8", "--"), "16 進、2 割を外す": ("#c0504d", "-"), "16 進、5 割を外す": ("#c0504d", "--")}


def bars(a, data, title, chance=None):
    n = len(data)
    w = .8 / len(WAYS)
    for j, way in enumerate(WAYS):
        a.bar([i + (j - 1.5) * w for i in range(n)], [v[j] for v in data.values()], w, color=COL[j], label=way.replace("\n", ""))
    if chance:
        for i, k in enumerate(data):
            a.plot([i - .45, i + .45], [chance[k]] * 2, color="k", lw=1, ls=":")
    a.set_xticks(range(n)); a.set_xticklabels(list(data), fontsize=9); a.set_ylim(0, 1.08); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.4), gridspec_kw=dict(width_ratios=[1, 1.3, 1, 1]))
    bars(ax[0], RECALL, "1. 聞いた組を引けるか（全部の組を聞いた時の当たり）")
    bars(ax[1], UNSEEN, "2. 見ていない組が当たるか（2 割を外して置いた時。点線は当てずっぽう）", CHANCE)
    ax[1].legend(fontsize=8.5, loc="upper right")
    for name, (ring, null) in BY_ETA.items():
        c, ls = LINE[name]
        ax[2].plot(range(3), ring, color=c, ls=ls, lw=2.2, marker="o", label=name)
        ax[2].plot(range(3), null, color=c, ls=ls, lw=1, marker="x", alpha=.6)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels([str(e) for e in ETAS]); ax[2].set_xlabel("外れた時に動かす大きさ（手で置いた値）")
    ax[2].set_title("3. 動かす大きさと、見ていない組の当たり\n（太い線は足し算、細い × はでたらめの世界）", fontsize=10.5)
    ax[2].set_ylim(0, 1.08); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8.5, loc="upper left")
    for name, ys in BY_DIM.items():
        c, ls = LINE[name]
        ax[3].plot(range(3), ys, color=c, ls=ls, lw=2.2, marker="o", label=name)
    ax[3].set_xticks(range(3)); ax[3].set_xticklabels([str(d) for d in DIMS]); ax[3].set_xlabel("位相の次元")
    ax[3].set_title("4. 次元と、見ていない組の当たり\n（同じ表、動かす大きさ .1）", fontsize=10.5); ax[3].set_ylim(0, 1.08); ax[3].grid(alpha=.3)
    fig.suptitle("2 つの語を掛け合わせた波で 3 つ組を持つ。外れた時だけ位相を動かすと、決まりの有る世界では見ていない組まで当たる（I49zzs、小さい世界）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzs_bind.png"), dpi=170)


if __name__ == "__main__":
    main()
