"""I49zzp の図: 覚えた式から、ずれを運んで見せていない式に答える（2+3=5 から 3+3= を答える）。作り物の世界だけ。その場で数える。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import i49zzp_arith_run as R

KEYS = ("そのまま借りる", "位相で運ぶ", "位置で運ぶ", "順で運ぶ")
COL = ("#999999", "#b0413e", "#3f73a8", "#d9a441")
NOTE = ("覚えた式の答えをそのまま言う", "位相（強い 2 つの向きの角度）で運ぶ", "位置（一番強い向きの座標）で運ぶ", "順（位置の順に並べた番号）で運ぶ")


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), sharey=True)
    for a, digits in zip(ax, (R.DIGITS, R.HEX)):
        cols = []
        for name, (add, sub) in (("小さい世界", R.world(False, digits)), ("全部の世界", R.world(True, digits)), ("1 の位だけ\nの世界", R.ring_world(digits))):
            lines = add + sub
            cols.append((name, R.shift_test(lines, True, digits)))
            cols.append((name + "\n答えを入れ替え", R.shift_test(R.null_world(lines), True, digits)))
        for i, (name, r) in enumerate(cols):
            for k, key in enumerate(KEYS):
                v = r[key]["当たり"]
                a.bar(i + (k - 1.5) * .2, v, .2, color=COL[k], alpha=1 if i % 2 == 0 else .45, label=NOTE[k] if i == 0 else None)
                a.text(i + (k - 1.5) * .2, v + .012, ("%.2f" % v).lstrip("0") if v else "0", ha="center", fontsize=8)
        a.axhline(1 / len(digits), color="#333333", lw=1.2, ls=":", label="当てずっぽう")
        a.set_xticks(range(len(cols))); a.set_xticklabels(["%s\n（%d 問）" % (n, r["試した式"]) for n, r in cols], fontsize=8.5)
        a.set_ylim(0, 1.0); a.grid(axis="y", alpha=.3)
        a.set_title("%d 進（0〜%s）" % (len(digits), digits[-1]), fontsize=11)
    ax[0].set_ylabel("見せていない式の当たり"); ax[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("2+3=5 を覚えていて、3+3= に答えられるか: 片方の数が 1 違う覚えた式から、波の上のずれを運ぶ（I49zzp）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "docs", "images", "i49zzp_shift.png"), dpi=170)


if __name__ == "__main__":
    main()
