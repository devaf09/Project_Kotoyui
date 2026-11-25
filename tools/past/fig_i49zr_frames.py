"""I49zr・I49zs の図: 席を増やした枠（が・を・に・で・と・へ・から・どうした）で数えた結果。

数は PC で数えた集計値をそのまま置いている（3 つのテキスト、451 話・枠 5,314）。
使い方: python tools/past/fig_i49zr_frames.py 出力.png
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
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

BLUE, GRAY, GREEN = "#4c72b0", "#8c8c8c", "#55a868"
# 足した席が有る枠の、が・を・どうした を隠す問い 1,654。4 通りの並べ方（色, [(名札, 4 つの数)], z）
SHOW = {"表": (BLUE, [("見せない", [.2443, .2418, .2461, .2364]), ("見せる", [.2969, .2823, .2890, .2896])], "z +4.1〜+5.2"),
        "表・席ごと": (BLUE, [("見せない", [.2557, .2509, .2557, .2461]), ("見せる", [.3210, .3035, .3083, .3071])], "z +5.2〜+6.3"),
        "解く形": (GRAY, [("見せない", [.2424, .2449, .2467, .2509]), ("見せる", [.3597, .3428, .3495, .3434])], "z +9.9〜+12.2"),
        "数えて選ぶ": (GREEN, [("見せない", [.2618, .2509, .2606, .2618]), ("見せる", [.3761, .3573, .3609, .3640])], "z +9.7〜+11.0"),
        "数えて選ぶ\n・席ごと": (GREEN, [("見せない", [.2660, .2636, .2696, .2684]), ("見せる", [.4002, .3736, .3827, .3857])], "z +11.0〜+12.8")}
# 三つの席だけの枠（4,552）
THREE = {"表": (BLUE, [("今まで", [.3581, .3615, .3605, .3593]), ("作り直し", [.3038, .3023, .3031, .3038]), ("席ごと", [.3571, .3600, .3600, .3585])],
                "今まで → 席ごと\nz −0.4〜−1.1"),
         "解く形": (GRAY, [("今まで", [.3419, .3426, .3444, .3429]), ("作り直し", [.3448, .3474, .3479, .3459])], "z +3.2〜+5.4"),
         "数えて選ぶ": (GREEN, [("今まで", [.3895, .3918, .3907, .3911]), ("作り直し", [.3562, .3599, .3588, .3577]), ("席ごと", [.3909, .3937, .3923, .3921])],
                   "今まで → 席ごと\nz +0.7〜+1.2")}
# 全部の席、並べ替え 3、隠した席ごと（席, 問い, 表, 表・席ごと, 解く形, 数えて選ぶ・席ごと）
SEAT = [("が", 5314, .4307, .4291, .4513, .4787), ("を", 4682, .2407, .3358, .2809, .3454), ("どうした", 5314, .2162, .2879, .2742, .3256),
        ("に", 549, .2787, .3078, .2514, .3461), ("で", 48, .2083, .2083, .25, .2708)]
SEAT_FORMS = [("表", BLUE, .45), ("表・席ごと", BLUE, 1), ("解く形", GRAY, 1), ("数えて選ぶ・席ごと", GREEN, 1)]


def group(a, data, title, ylim):
    for i, (name, (col, series, z)) in enumerate(data.items()):
        w = .8 / len(series)
        for k, (lab, ys) in enumerate(series):
            px = i + (k - (len(series) - 1) / 2) * w
            a.bar(px, sum(ys) / 4, w * .94, color=col, alpha=.4 + .6 * k / max(1, len(series) - 1))
            a.plot([px] * 4, ys, ".", color="k", ms=3)
            a.text(px, max(ys) + .003, ("%.3f" % (sum(ys) / 4))[1:], ha="center", fontsize=8)
            a.text(px, ylim[0] + .003, lab, ha="center", fontsize=7)
        a.text(i, ylim[1] - (ylim[1] - ylim[0]) * .09, z, ha="center", fontsize=8)
    a.set_xticks(range(len(data)))
    a.set_xticklabels(list(data), fontsize=9)
    a.set_ylim(*ylim)
    a.set_title(title, fontsize=11)
    a.grid(axis="y", alpha=.3)


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.8), gridspec_kw=dict(width_ratios=[1.25, 1.1, 1.1]))
    group(ax[0], SHOW, "足した席（に・で など）を見せると、同じ 1,654 問が当たりやすくなる", (.2, .44))
    ax[0].set_ylabel("正しい物（棒は 4 通りの並べ方の平均、点は並べ方ごと）")
    group(ax[1], THREE, "三つの席だけの行: 席ごとに段を分けると、今までの表に並ぶ", (.28, .43))
    a = ax[2]
    w = .2
    for i, row in enumerate(SEAT):
        for k, (name, col, al) in enumerate(SEAT_FORMS):
            a.bar(i + (k - 1.5) * w, row[2 + k], w * .94, color=col, alpha=al, label=name if i == 0 else None)
        a.text(i, max(row[2:]) + .012, "{:,} 問".format(row[1]), ha="center", fontsize=9)
    a.axvline(2.5, color="k", lw=.8, ls=":")
    a.text(3.5, .40, "足した席", ha="center", fontsize=10)
    a.set_xticks(range(len(SEAT)))
    a.set_xticklabels([s[0] for s in SEAT])
    a.set_xlabel("隠した席（「で」は揺れが大きい。と・へ・から は 20 問より少なく、描いていない）")
    a.set_ylim(0, .62)
    a.set_title("隠した席ごとの当たり（全部の席、並べ替え 3）", fontsize=11)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=9, loc="upper right", ncol=2)
    fig.suptitle("席を増やした枠（451 話・枠 5,314）: 足した席は効く。表は「どの席が同じ語だったか」で段を分けると、今までの表に並ぶ", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zr_frames.png")
