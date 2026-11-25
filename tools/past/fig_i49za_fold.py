"""I49za の図: 板を「ふだん」と比べる決まり（行の折り畳みの決まり）を入れた時の、正しい物の割合。

数は PC で数えた集計値をそのまま置いている（1 組で半分に薄れる板）。
使い方: python tools/past/fig_i49za_fold.py 出力.png
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

U = ["比べない", "明るい分だけ", "残す・\nほかは半分", "引く"]
D = {
    "白紙から": (.4084, {"九割以内・1 枚": [.4154, .4148, .4152, .4158], "九割以内・席ごと": [.4168, .4178, .4167, .4156],
                     "上位 3・1 枚": [.3459, .3213, .3427, .3206], "上位 3・席ごと": [.3737, .3694, .3740, .3534]}),
    "青空の 3 つ組を先に": (.3983, {"九割以内・1 枚": [.4115, .4070, .4093, .4066], "九割以内・席ごと": [.4117, .4114, .4123, .4094],
                            "上位 3・1 枚": [.3494, .3328, .3449, .3297], "上位 3・席ごと": [.3632, .3609, .3638, .3531]}),
}
COL = {"九割以内・1 枚": "#4c72b0", "九割以内・席ごと": "#dd8452", "上位 3・1 枚": "#a9bfe0", "上位 3・席ごと": "#f2c9ae"}


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    w = .2
    for j, start in enumerate(D):
        a = ax[j]
        base, rows = D[start]
        for k, (name, ys) in enumerate(rows.items()):
            xs = [i + (k - 1.5) * w for i in range(4)]
            a.bar(xs, ys, w, color=COL[name], label=name)
            for x, y in zip(xs, ys):
                a.text(x, y + .002, f"{y:.3f}"[1:], ha="center", fontsize=7, rotation=90)
        a.axhline(base, color="k", ls="--", lw=1, label=f"板無し {base:.3f}".replace("0.", "."))
        a.set_xticks(range(4))
        a.set_xticklabels(U)
        a.set_ylim(.3, .45)
        a.set_title(start)
        a.grid(axis="y", alpha=.3)
        a.legend(fontsize=8, ncol=3, loc="upper center")
    ax[0].set_ylabel("正しい物の割合")
    fig.suptitle("板をふだんと比べる決まり: 一位と並ぶ候補の決め手に使う時は、入れても入れなくても同じ", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49za_fold.png")
