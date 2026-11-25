"""I49zzk の図: 学習から作った時制の波（語ごとの強さ、形どうしの近さ、形を選ぶ進み）。作り物の世界の数えをその場で作る。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]


def main():
    import i49zzk_tense_run as R
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    r = R.run()
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw=dict(width_ratios=[1.15, .9, 1]))
    st = r["時制の影響を受ける強さ"]
    names = list(st)
    y = np.arange(len(names))[::-1]
    ax[0].barh(y + .2, [st[n]["引かない"] for n in names], .4, color="#cfcfcf", label="平均を引かない")
    ax[0].barh(y - .2, [st[n]["平均を引く"] for n in names], .4, color="#3f73a8", label="平均を引く")
    ax[0].set_yticks(y); ax[0].set_yticklabels([n.replace(":", " ") for n in names], fontsize=9)
    ax[0].set_xlabel("時制の影響を受ける強さ（波の長さ）"); ax[0].legend(loc="lower right", fontsize=9)
    ax[0].set_title("語ごと: 時制に関係の有る語だけが強いか", fontsize=11)
    t = r["形の波どうしの近さ"]
    m = np.array(t["表"])
    ax[1].imshow(m, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i in range(5):
        for j in range(5):
            ax[1].text(j, i, "%.2f" % m[i, j], ha="center", va="center", fontsize=9, color="white" if abs(m[i, j]) > .6 else "black")
    ax[1].set_xticks(range(5)); ax[1].set_xticklabels(t["時制"], fontsize=9); ax[1].set_yticks(range(5)); ax[1].set_yticklabels(t["時制"], fontsize=9)
    ax[1].set_title("出来た形の波どうしの近さ（並びは置いていない）", fontsize=11)
    c = r["形を選ぶ"]
    xs = range(len(R.MARKS))
    for key, col, lab, style in (("つなぎの語で持つ", "出ていない動詞", "つなぎの語で持つ（出ていない動詞）", dict(color="#3f73a8", ls="-", marker="o")),
                                 ("つなぎの語で持つ・平均を引かない", "出ていない動詞", "同じ・平均を引かない", dict(color="#9bb7d4", ls="--", marker="o")),
                                 ("動詞ごとにひとまとめ", "学習に出た動詞", "動詞ごとに持つ（学習に出た動詞）", dict(color="#d9a441", ls="-", marker="s")),
                                 ("動詞ごとにひとまとめ", "出ていない動詞", "動詞ごとに持つ（出ていない動詞）", dict(color="#c8452c", ls="-", marker="s"))):
        ax[2].plot(xs, [p[col] for p in c[key]], lw=2, label=lab, **style)
    ax[2].axhline(r["当てずっぽう"], color="gray", ls=":")
    ax[2].text(len(R.MARKS) - 1, r["当てずっぽう"] + .015, "当てずっぽう", color="gray", ha="right", fontsize=9)
    ax[2].set_xticks(list(xs)); ax[2].set_xticklabels([str(k) for k in R.MARKS]); ax[2].set_ylim(0, 1.08)
    ax[2].set_xlabel("聞いた文の数"); ax[2].set_ylabel("選んだ形が世界の決まりに合う割合")
    ax[2].legend(loc="center right", fontsize=8.5); ax[2].grid(alpha=.3)
    ax[2].set_title("時の語から形を選ぶ", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "docs", "images", "i49zzk_tense.png"), dpi=170)


if __name__ == "__main__":
    main()
