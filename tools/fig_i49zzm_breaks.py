"""I49zzm の図: 状況波が合わなくなった所を、場の切れ目として読めるか（run48 の集計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, a = plt.subplots(figsize=(8.6, 4.8))
    v = [.6802, .5847, .6856]
    a.bar(range(3), v, color=["#3f73a8", "#bdbdbd", "#9bb7d4"], width=.6)
    for i, x in enumerate(v):
        a.text(i, x + .006, "%.3f" % x, ha="center")
    a.axhline(.5, color="gray", ls=":")
    a.text(2.45, .505, "見分けられない", color="gray", ha="right", fontsize=9)
    a.set_xticks(range(3))
    a.set_xticklabels(["学んだ波で見る\n（切れ目の直後 対 続きの中）", "同じ終わりの数だけで見る\n（真似るだけ）", "確かめ: 発言はそのままで\n前の発言だけ別の場に替える"], fontsize=9.5)
    a.set_ylim(.4, .76); a.set_ylabel("続きの中の方が、前の発言と合う割合")
    a.set_title("状況波が合わなくなった所は、場の切れ目か\n（青空文庫の会話。切れ目の直後 19,766 発言、続きの中 33,914 発言）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzm_breaks.png"), dpi=170)


if __name__ == "__main__":
    main()
