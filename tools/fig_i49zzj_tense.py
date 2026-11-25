"""I49zzj の図: 時の語と時制の近さ（山の幅 2 通り）と、会話で状況波が付いて行く割合（作り物の世界の数えをその場で作る）。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]


def main():
    import i49zzj_tense_run as R
    import i49zzj_tense_wave as TW
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    keep = (TW.WIDTH, TW.HALF)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1, .9]))
    words = list(R.RULE)
    try:
        for a, w in zip(ax[:2], (6.0, 8.0)):
            TW.WIDTH, TW.HALF = w, keep[1]
            r = R.run()
            m = np.array([[r["時の語と時制"][x]["近さ"][t] for t in TW.TENSES] for x in words])
            a.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i, x in enumerate(words):
                for j, t in enumerate(TW.TENSES):
                    a.text(j, i, "%.2f" % m[i, j], ha="center", va="center", color="white" if m[i, j] > .6 else "black", fontsize=9)
                    if t in R.RULE[x]:
                        a.add_patch(Rectangle((j - .47, i - .47), .94, .94, fill=False, ec="#2e8b57", lw=2))
                j = TW.TENSES.index(r["時の語と時制"][x]["一番近い"])
                ok = r["時の語と時制"][x]["一番近い"] in R.RULE[x]
                a.text(j + .33, i - .25, "○" if ok else "×", color="#2e8b57" if ok else "#c8452c", fontsize=11, ha="center", va="center")
            a.set_xticks(range(5)); a.set_xticklabels(TW.TENSES); a.set_yticks(range(len(words))); a.set_yticklabels(words)
            n = sum(r["時の語と時制"][x]["一番近い"] in R.RULE[x] for x in words)
            a.set_title("山の幅 %d（隣との近さ %.2f）: %d / 8 語が合う" % (w, r["隣の時制との近さ"], n), fontsize=11)
        halves = (0.25, 0.5, 1.0, 2.0)
        for w, c in ((6.0, "#3f73a8"), (8.0, "#d9a441")):
            ys = []
            for h in halves:
                TW.WIDTH, TW.HALF = w, h
                ys.append(R.run()["会話"]["山の形のまま"]["直後の文"])
            ax[2].plot(range(len(halves)), ys, "o-", color=c, lw=2, label="山の幅 %d" % w)
    finally:
        TW.WIDTH, TW.HALF = keep
    ax[2].set_xticks(range(len(halves))); ax[2].set_xticklabels([str(h) for h in halves])
    ax[2].set_xlabel("状況波が半分に薄れるまでの文の数"); ax[2].set_ylabel("時制が合った数（8 回の切り替え）")
    ax[2].set_ylim(0, 8.6); ax[2].grid(alpha=.3); ax[2].legend(loc="lower left")
    ax[2].set_title("会話: 時の語が変わった直後の文", fontsize=11)
    fig.suptitle("状況波（時制）の小さな実験。数字は時の語と時制の近さ、緑の枠は世界の決まり、○×は一番近い時制", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "docs", "images", "i49zzj_tense.png"), dpi=170)


if __name__ == "__main__":
    main()
