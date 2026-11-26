"""I49zzx の図: 前の 3 つ組から、次の 3 つ組を 1 つの返事として組む（run72 の合計の数だけ。5,987 問）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

N = 5987
FORMS = ("全部\n引き継ぎ", "席ごと\n（全部、遷移の波）", "席ごと\n（人は引き継ぎ）", "動きを引いて、\n物は本体", "物を引いて、\n動きは本体")
WHOLE = (0, 183, 193, 296, 237)
TWO = (268, 940, 1011, 994, 888)
HEARD = (780, 665, 507, 1064, 1191)
SAME = (5987, 172, 304, 440, 681)
REAL_HEARD = 764


def bars(a, rows, title, ylim, clip=None):
    w = .8 / len(rows)
    for j, (name, col, vals) in enumerate(rows):
        for i, v in enumerate(vals):
            x, y = i + (j - (len(rows) - 1) / 2) * w, v / N
            a.bar(x, min(y, clip or y), w, color=col, label=name if i == 0 else None)
            a.text(x, min(y, clip or y) + ylim * .012, ("%.2f" % y) if y >= .1 else ("%.3f" % y), ha="center", fontsize=8.5)
    a.set_xticks(range(len(FORMS))); a.set_xticklabels(FORMS, fontsize=9); a.set_ylim(0, ylim); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    bars(ax[0], (("3 席とも当たり", "#3f73a8", WHOLE), ("2 席以上当たり", "#9dbbd8", TWO)), "1. 組んだ 3 つ組が、本物の次の 3 つ組と合う割合", .21)
    ax[0].legend(fontsize=9, loc="upper left"); ax[0].set_ylabel("割合")
    bars(ax[1], (("前に丸ごと聞いた 3 つ組だった", "#5a9a68", HEARD), ("前の 3 つ組と同じ（オウム返し）", "#c96f5a", SAME)),
         "2. 組んだ 3 つ組のまとまりと、オウム返し（全部引き継ぎの赤は 1.00 で、上を切って描いた）", .30, clip=.27)
    ax[1].axhline(REAL_HEARD / N, color="#333333", ls="--", lw=1)
    ax[1].text(1.5, REAL_HEARD / N + .004, "本物の次の 3 つ組が、前に聞いた物: %.2f" % (REAL_HEARD / N), fontsize=8.5, ha="center", va="bottom")
    ax[1].legend(fontsize=9, loc="upper center")
    fig.suptitle("3 つの席を合わせて 1 つの返事に組む: 残りを本体で埋めると、丸ごとの当たりが上がる（I49zzx、run72）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzx_reply.png"), dpi=170)


if __name__ == "__main__":
    main()
