"""I49zzk の図: 分け方を決めず、切り出しを全部数えた時の、切り出しごとの強さと、形を選ぶ進み。作り物の世界の数えをその場で作る。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    r = R.run_spans()
    rows = r["切り出しごとの強さ"]
    free = [x for x in rows if not x["動詞の字を含む"]]
    stem = [x for x in rows if x["動詞の字を含む"] and len(x["切り出し"]) <= 2 and x["強さ"] < .35]
    show = free + stem
    fig, ax = plt.subplots(1, 2, figsize=(14, 7.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    y = range(len(show))[::-1]
    ax[0].barh(list(y), [x["強さ"] for x in show], color=["#3f73a8" if not x["動詞の字を含む"] else "#d9a441" for x in show])
    ax[0].set_yticks(list(y)); ax[0].set_yticklabels([x["切り出し"] for x in show], fontsize=8.5)
    ax[0].set_xlabel("時制の影響を受ける強さ（波の長さ）")
    ax[0].set_title("切り出しごとの強さ（青: 動詞の字を含まない %d 個の全部、黄: 動詞の頭の字）\n覚えた切り出しは全部で %d 種類（形は %d 種類）" % (len(free), r["切り出しの種類"], r["形の種類"]), fontsize=10.5)
    ax[0].grid(axis="x", alpha=.3)
    xs = range(len(R.MARKS))
    c = r["形を選ぶ"]
    for key, col in (("切り出しを全部持つ", "#3f73a8"), ("つなぎの語に分けて持つ", "#c8452c")):
        ax[1].plot(xs, [p["決まりの内"] for p in c[key]], "o-", color=col, lw=2, label=key + "（決まりの内）")
        ax[1].plot(xs, [p["一番多い時制"] for p in c[key]], "s--", color=col, lw=1.5, alpha=.7, label=key + "（一番多い時制）")
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels([str(k) for k in R.MARKS]); ax[1].set_ylim(0, 1.08)
    ax[1].set_xlabel("聞いた文の数"); ax[1].set_ylabel("学習に出ていない動詞で、選んだ形が合う割合")
    ax[1].legend(loc="lower right", fontsize=9); ax[1].grid(alpha=.3)
    ax[1].set_title("時の語から形を選ぶ（学習に出ていない動詞 2 つ）", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "docs", "images", "i49zzk_spans.png"), dpi=170)


if __name__ == "__main__":
    main()
