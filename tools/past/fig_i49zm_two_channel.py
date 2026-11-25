"""I49zm の図: 属性の波と、語ごとに固有の位相を 1 本に並べた時の、正しい物の割合と、どの種類の問いで変わったか。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zm_two_channel.py 出力.png
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

# 始め方: (属性だけ, 表だけ, 固有 256 本 [混ぜ .5, 1, 2], 固有 1,024 本, 属性＋語の波・混ぜ 1, 固有＋振幅を直す)
D = {"白紙から": (.4084, .4321, [.4107, .4109, .3969], [.4107, .4141, .4105], .4128, .4237),
     "青空の 3 つ組を先に": (.3983, .4533, [.4058, .4076, .3949], [.4077, .4118, .4072], .4013, .4199)}
LV = ["聞いた組\nそのもの", "片方だけ\n借りる", "述語側", "もう片方", "借りる", "答えられ\nない"]
# 属性だけ → 属性＋固有（混ぜ 1・1,024 本）で、段ごとに増えた当たり
DIFF = {"白紙から": [22, 23, -5, -12, 17, 2], "青空の 3 つ組を先に": [68, 64, -2, -19, 0, 0]}


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.3), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for j, start in enumerate(D):
        a = ax[j]
        base, table, k256, k1024, wave, amp = D[start]
        a.axhline(table, color="#8c8c8c", lw=1.5, ls="-.", label="関係の表だけ " + f"{table:.3f}"[1:])
        a.axhline(base, color="k", lw=1, ls="--", label="属性だけ（今までの形） " + f"{base:.3f}"[1:])
        a.plot(range(3), k256, color="#9ecae1", marker="o", label="属性＋固有の位相 256 本")
        a.plot(range(3), k1024, color="#4c72b0", marker="s", label="属性＋固有の位相 1,024 本")
        a.plot([1], [wave], color="#8172b3", marker="^", ms=9, ls="none", label="属性＋語の波（似た語が近い波。比べ役） " + f"{wave:.3f}"[1:])
        a.plot([1], [amp], color="#dd8452", marker="D", ms=9, ls="none", label="固有 1,024 本＋振幅を誤差で直す " + f"{amp:.3f}"[1:])
        a.set_xticks(range(3))
        a.set_xticklabels(["固有を弱く（.5）", "同じ強さ（1）", "固有を強く（2）"])
        a.set_xlabel("属性と固有の混ぜ具合")
        a.set_ylim(.375, .46)
        a.set_ylabel("正しい物の割合")
        a.set_title(start)
        a.grid(alpha=.3)
        a.legend(fontsize=7.5, loc="upper right" if j == 0 else "center right")
    a = ax[2]
    w = .38
    for k, (start, c) in enumerate((("白紙から", "#4c72b0"), ("青空の 3 つ組を先に", "#dd8452"))):
        xs = [i + (k - .5) * w for i in range(len(LV))]
        a.bar(xs, DIFF[start], w * .94, color=c, label=start)
        for x, v in zip(xs, DIFF[start]):
            a.text(x, v + (2 if v >= 0 else -2), f"{v:+d}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    a.axhline(0, color="k", lw=.8)
    a.axvspan(-.5, 2.5, color="#f0f0f0", zorder=0)
    a.text(1, 82, "聞いた事の有る組", ha="center", fontsize=9)
    a.text(4, 82, "聞いた事の無い組", ha="center", fontsize=9)
    a.set_xticks(range(len(LV)))
    a.set_xticklabels(LV, fontsize=9)
    a.set_xlim(-.5, 5.5)
    a.set_ylim(-35, 92)
    a.set_ylabel("固有の位相を並べて増えた当たり（問い）")
    a.set_title("増えたのは、聞いた事の有る組")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="lower right")
    fig.suptitle("属性＋固有の位相: 狙い通り「既知との一致」は伸びる（+0.6 / +1.4 ポイント）。聞いた事の無い組は伸びず、表には届かない", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zm_two_channel.png")
