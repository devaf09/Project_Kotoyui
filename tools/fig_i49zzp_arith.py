"""I49zzp の図: 算数の文字の並びから、数字の波に数の並びと位相の関係が出るか（作り物の世界だけ。その場で数える）。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import i49zzp_arith_run as R

COL = {"小さい世界": "#3f73a8", "全部の世界": "#d9a441", "1 の位だけの世界": "#b0413e"}
NOTE = {"小さい世界": "小さい世界（答えが 1 文字の式だけ）", "全部の世界": "全部の世界（9+9=18、3-5=-2 も入る）", "1 の位だけの世界": "1 の位だけの世界（7+5=2。回数の偏りが無い）"}


def worlds():
    out = {}
    for name, (add, sub) in (("小さい世界", R.world()), ("全部の世界", R.world(True)), ("1 の位だけの世界", R.ring_world())):
        lines = add + sub
        out[name] = (R.run(lines), R.run(R.null_world(lines)))
    return out


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    res = worlds()
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.6))
    a = ax[0]
    for name, (real, null) in res.items():
        xs = list(range(1, 10))
        a.plot(xs, [real["数の差ごとの近さ"][str(x)] for x in xs], color=COL[name], lw=2.2, marker="o", label=NOTE[name])
        a.plot(xs, [null["数の差ごとの近さ"][str(x)] for x in xs], color=COL[name], lw=1.4, ls="--", marker="x", label="同じ世界で、答えだけを入れ替えた時")
    a.set_xlabel("2 つの数字の差"); a.set_ylabel("数字の波の近さ（1 が同じ）"); a.set_ylim(0, 1.08)
    a.set_title("1. 近い数ほど、波も近いか", fontsize=11); a.grid(alpha=.3); a.legend(fontsize=8, loc="lower left")
    a = ax[1]
    for name, mk in (("小さい世界", "o"), ("全部の世界", "s")):
        p = res[name][0]["位相"]["座標"]
        xs, ys = [p[c][0] for c in R.DIGITS], [p[c][1] for c in R.DIGITS]
        a.plot(xs, ys, color=COL[name], lw=1, alpha=.6)
        a.scatter(xs, ys, color=COL[name], marker=mk, s=42, label=name)
        for c, x, y in zip(R.DIGITS, xs, ys):
            a.annotate(c, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=11, color=COL[name])
    a.scatter([0], [0], color=COL["1 の位だけの世界"], marker="*", s=160, label="1 の位だけの世界（10 個とも同じ点）")
    a.set_xlabel("一番強い向き"); a.set_ylabel("2 番目に強い向き")
    a.set_title("2. 数字の波を、強い 2 つの向きの面に置いた所", fontsize=11); a.grid(alpha=.3); a.legend(fontsize=8, loc="lower right")
    a = ax[2]
    names = list(res)
    for i, name in enumerate(names):
        real, null = res[name]
        k = "位相の足し算が合う式の割合_全部覚えた時"
        a.bar(i - .2, real[k], .4, color=COL[name], label="式のまま" if i == 0 else None)
        a.bar(i + .2, null[k], .4, color=COL[name], alpha=.35, hatch="//", label="答えだけを入れ替えた時" if i == 0 else None)
        a.text(i - .2, real[k] + .015, "%.2f" % real[k], ha="center", fontsize=10)
        a.text(i + .2, null[k] + .015, "%.2f" % null[k], ha="center", fontsize=10)
    a.axhline(.1, color="#333333", lw=1.2, ls=":", label="当てずっぽう（10 個から 1 個）")
    a.set_xticks(range(len(names))); a.set_xticklabels(["小さい世界", "全部の世界\n（答えが 1 文字の式）", "1 の位だけの世界\n（波が全部同じで、読めない）"], fontsize=9)
    a.set_ylim(0, .62); a.set_ylabel("位相の足し算 φ(a)±φ(b) が、答えの数字の位相に一番近い式の割合")
    a.set_title("3. 位相の足し算は、式に合うか", fontsize=11); a.grid(axis="y", alpha=.3); a.legend(fontsize=8, loc="upper right")
    fig.suptitle("算数の文字（0〜9 ＋ − ＝）の並びを数えるだけで、数字の波に数の並びと位相の関係が出るか（I49zzp）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "docs", "images", "i49zzp_arith.png"), dpi=170)


if __name__ == "__main__":
    main()
