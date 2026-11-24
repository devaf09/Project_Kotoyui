"""I49zr の図: 席を増やした枠（が・を・に・で・と・へ・から・どうした）で数えた結果。

数は PC で数えた集計値をそのまま置いている（3 つのテキスト、451 話・枠 5,314）。
使い方: python tools/fig_i49zr_frames.py 出力.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = os.environ.get("KOTOYUI_FONT", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

C = dict(表="#4c72b0", 解く形="#8c8c8c", 数えて選ぶ="#55a868")
# 足した席が有る枠の、が・を・どうした を隠す問い 1,654。4 通りの並べ方（見せない, 見せる）
SHOW = {"表": ([.2443, .2418, .2461, .2364], [.2969, .2823, .2890, .2896], "z +4.1〜+5.2"),
        "解く形": ([.2424, .2449, .2467, .2509], [.3597, .3428, .3495, .3434], "z +9.9〜+12.2"),
        "数えて選ぶ": ([.2618, .2509, .2606, .2618], [.3761, .3573, .3609, .3640], "z +9.7〜+11.0")}
# 三つの席だけの枠（4,552）。4 通りの並べ方（今までの部品, 作り直した部品）
THREE = {"表": ([.3581, .3615, .3605, .3593], [.3038, .3023, .3031, .3038], "z −22 前後"),
         "解く形": ([.3419, .3426, .3444, .3429], [.3448, .3474, .3479, .3459], "z +3.2〜+5.4"),
         "数えて選ぶ": ([.3895, .3918, .3907, .3911], [.3562, .3599, .3588, .3577], "z −13 前後")}
# 全部の席、並べ替え 3、隠した席ごと（席, 問い, 表, 解く形, 数えて選ぶ）
SEAT = [("が", 5314, .4307, .4513, .4614), ("を", 4682, .2407, .2809, .2935), ("どうした", 5314, .2162, .2742, .2806),
        ("に", 549, .2787, .2514, .2805), ("で", 48, .2083, .25, .25), ("から", 17, .2353, .1176, .1765), ("へ", 12, .0833, .0833, .1667)]


def pair(a, data, left, right, title, ylim):
    w = .36
    for i, (name, (x, y, z)) in enumerate(data.items()):
        for k, (ys, al) in enumerate(((x, .45), (y, 1))):
            px = i + (k - .5) * w
            a.bar(px, sum(ys) / 4, w * .94, color=C[name], alpha=al)
            a.plot([px] * 4, ys, ".", color="k", ms=3)
            a.text(px, max(ys) + .004, ("%.3f" % (sum(ys) / 4))[1:], ha="center", fontsize=9)
            a.text(px, ylim[0] + .004, (left, right)[k], ha="center", fontsize=8, color="k")
        a.text(i, ylim[1] - .012, z, ha="center", fontsize=9)
    a.set_xticks(range(3))
    a.set_xticklabels(list(data))
    a.set_ylim(*ylim)
    a.set_title(title, fontsize=11)
    a.grid(axis="y", alpha=.3)


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    pair(ax[0], SHOW, "見せない", "見せる", "足した席（に・で など）を見せると、同じ 1,654 問が当たりやすくなる", (.2, .4))
    ax[0].set_ylabel("正しい物（棒は 4 通りの並べ方の平均、点は並べ方ごと）")
    pair(ax[1], THREE, "今まで", "作り直し", "三つの席だけの行では、作り直した表が今までの表より下", (.28, .42))
    a = ax[2]
    w = .27
    for i, (s, n, t, u, m) in enumerate(SEAT):
        for k, (v, name) in enumerate(((t, "表"), (u, "解く形"), (m, "数えて選ぶ"))):
            a.bar(i + (k - 1) * w, v, w * .94, color=C[name], label=name if i == 0 else None)
        a.text(i, max(t, u, m) + .012, "{:,} 問".format(n), ha="center", fontsize=9)
    a.axvline(2.5, color="k", lw=.8, ls=":")
    a.text(4.7, .36, "足した席\n（問いが少なく、「で」より右は\n揺れが大きい）", ha="center", fontsize=9)
    a.set_xticks(range(len(SEAT)))
    a.set_xticklabels([s[0] for s in SEAT])
    a.set_xlabel("隠した席")
    a.set_ylim(0, .52)
    a.set_title("隠した席ごとの当たり（全部の席、並べ替え 3）", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper right")
    fig.suptitle("席を増やした枠（451 話・枠 5,314）: 足した席は効く。ただし、作り直した表は今までの表に届いていない", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zr_frames.png")
