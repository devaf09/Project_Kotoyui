"""I49s の図: 関係の表と解く形の差が、表のどの段の問いで出ているか（記録 21 節の数字を、そのまま描く）。
使い方: python tools/past/fig_i49s_where_gap.py 出力.png
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

FONT = os.environ.get("KOTOYUI_FONT", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
if os.path.exists(FONT):
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
INK, RED, BLUE, GREY, GREEN = "#30323d", "#d1495b", "#00798c", "#9a9a9a", "#5b8e3e"
ST = ["2 語とも\n同じ組を聞いた", "片方だけ\n借りる", "述語側の\n1 語だけ", "もう片方の\n1 語だけ", "2 語とも\n借りる", "表は\n答えられない"]
D = {"白紙から": dict(n=[2585, 2132, 99, 2111, 1004, 298], t=[.571, .506, .515, .292, .333, .000], s=[.496, .456, .495, .317, .348, .131],
                    d=[-194, -107, -2, 54, 15, 39]),
     "青空の 3 つ組を先に": dict(n=[2697, 2929, 278, 1669, 614, 42], t=[.568, .455, .468, .312, .347, .000], s=[.442, .396, .353, .355, .365, .310],
                           d=[-341, -175, -32, 72, 11, 13])}
fig = plt.figure(figsize=(14, 9.4))
fig.patch.set_facecolor("white")
fig.suptitle("関係の表と、表を持たない解く形（属性・席あり）の差は、どの問いで出ているか（8,229 問）", fontsize=15, color=INK, y=0.975)
x = np.arange(6)
for j, (start, d) in enumerate(D.items()):
    ax = fig.add_axes([0.06 + 0.48 * j, 0.54, 0.43, 0.33])
    ax.bar(x - 0.2, d["t"], 0.38, color=GREY, label="関係の表")
    ax.bar(x + 0.2, d["s"], 0.38, color=BLUE, label="解く・属性（席あり）")
    for xi, (a, b, n) in enumerate(zip(d["t"], d["s"], d["n"])):
        ax.text(xi - 0.2, a + 0.01, ("%.2f" % a)[1:], ha="center", fontsize=9.5, color=INK)
        ax.text(xi + 0.2, b + 0.01, ("%.2f" % b)[1:], ha="center", fontsize=9.5, color=INK)
        ax.text(xi, -0.115, "%s 問" % format(n, ","), ha="center", fontsize=9, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(ST, fontsize=9)
    ax.set_ylim(0, 0.72)
    ax.set_title("%s: 段ごとの当たり" % start, fontsize=12.5, color=INK, loc="left")
    if j == 0:
        ax.set_ylabel("1 位の当たり", fontsize=11, color=INK)
        ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax = fig.add_axes([0.06 + 0.48 * j, 0.11, 0.43, 0.27])
    ax.bar(x, d["d"], 0.6, color=[RED if v < 0 else GREEN for v in d["d"]])
    for xi, v in enumerate(d["d"]):
        ax.text(xi, v + (8 if v >= 0 else -8), "%+d" % v, ha="center", va="bottom" if v >= 0 else "top", fontsize=10.5, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ST, fontsize=9)
    ax.set_ylim(-400, 130)
    ax.set_title("%s: 解く形が勝った問いの数 − 負けた問いの数（合計 %+d）" % (start, sum(d["d"])), fontsize=12, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.text(0.5, 0.925, "横の並び = 表が答えに使った段。左ほど「聞いた組そのもの」を使い、右ほど「似た語や片方の語」に頼っている。", ha="center", fontsize=11, color=INK)
fig.text(0.5, 0.035, "読み: 解く形が負けているのは、見せた 2 語とまったく同じ組を前に聞いていた問い。表はその数だけを使うが、解く形は全部を 1 本に足すので、同じ組の分が薄まる。", ha="center", fontsize=11, color=INK)
fig.text(0.5, 0.010, "表が似た語に頼る問い（右側）と、表が答えられない問いでは、解く形の方が当たる。", ha="center", fontsize=11, color=INK)
fig.savefig(sys.argv[1], dpi=170)
