"""I49x の図: 見せた 2 語の点を足すか掛けるか。元のままとの差（ポイント）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49x_and.py 出力.png
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

VERS = ["別々に出して足す", "掛ける", "小さい方", "掛ける・\n無ければ足す"]
# 元のまま, 各版（正しい物）
D = {
    "属性・白紙から": (.4084, [.4082, .4129, .4013, .4131]),
    "属性・青空を先に": (.3983, [.3982, .4062, .3709, .4062]),
    "語の波・白紙から": (.3699, [.3765, .3490, .3380, .3569]),
    "語の波・青空を先に": (.3424, [.3497, .3206, .3047, .3259]),
}
COL = ["#dd8452", "#f2c9ae", "#55a868", "#b5dcc0"]


def main(out):
    fig, a = plt.subplots(figsize=(10.5, 5.2))
    w = .2
    for k, (name, (base, vals)) in enumerate(D.items()):
        xs = [i + (k - 1.5) * w for i in range(len(VERS))]
        ds = [(v - base) * 100 for v in vals]
        a.bar(xs, ds, w, color=COL[k], label=f"{name}（元のまま {base:.3f}）".replace("0.", "."))
        for x, d in zip(xs, ds):
            a.text(x, d + (.08 if d >= 0 else -.08), f"{d:+.1f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
    a.axhline(0, color="k", lw=.8)
    a.set_xticks(range(len(VERS)))
    a.set_xticklabels(VERS)
    a.set_ylabel("元のまま（2 語の波を足す）との差（ポイント）")
    a.set_ylim(-4.6, 2.2)
    a.grid(axis="y", alpha=.3)
    a.legend(loc="lower left", fontsize=9)
    a.set_title("見せた 2 語の点の合わせ方: 掛けても 1 ポイント弱しか動かない。語の波では下がる", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49x_and.png")
