"""I49r の図: 席の印を足した時の当たり（記録 20 節の数字を、そのまま描く）。
使い方: python tools/fig_i49r_seat_mark.py 出力.png
"""
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
INK, RED, BLUE, GOLD, GREY = "#30323d", "#d1495b", "#00798c", "#c9861a", "#9a9a9a"
LBLUE, LGOLD = "#8fc3cb", "#e6c48d"
D = {"白紙から": {"関係の表": (.432, .573, .309, .415), "属性": (.408, .505, .307, .413), "属性 + 席の印": (.411, .511, .301, .421),
                 "語の波": (.370, .426, .299, .385), "語の波 + 席の印": (.400, .499, .293, .406)},
     "青空の 3 つ組を先に": {"関係の表": (.453, .562, .335, .462), "属性": (.398, .477, .315, .403), "属性 + 席の印": (.397, .492, .298, .399),
                        "語の波": (.342, .358, .323, .346), "語の波 + 席の印": (.374, .467, .303, .351)}}
COL = {"関係の表": GREY, "属性": LBLUE, "属性 + 席の印": BLUE, "語の波": LGOLD, "語の波 + 席の印": GOLD}
fig = plt.figure(figsize=(14, 8.6))
fig.patch.set_facecolor("white")
fig.suptitle("席の印（その席によく座る語、を波に足しておく）を足すと、どこが変わるか（会話 600 本、8,229 問）", fontsize=15, color=INK, y=0.975)
for j, (start, d) in enumerate(D.items()):
    ax = fig.add_axes([0.06 + 0.48 * j, 0.14, 0.43, 0.68])
    x = np.arange(4)
    for i, (name, vals) in enumerate(d.items()):
        ax.bar(x + (i - 2) * 0.165, vals, 0.155, color=COL[name], label=name)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 2) * 0.165, v + 0.006, ("%.3f" % v)[1:], ha="center", fontsize=8.5, color=INK, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["全体", "だれが", "何を", "どうした"], fontsize=11.5)
    ax.set_ylim(0, 0.68)
    ax.set_title(start, fontsize=13, color=INK, loc="left")
    if j == 0:
        ax.set_ylabel("1 位の当たり", fontsize=11, color=INK)
        ax.legend(fontsize=10.5, frameon=False, loc="upper right", ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.text(0.5, 0.875, "薄い色 = 席の印なし、濃い色 = 席の印あり。青 = 属性を元にした形、黄 = 語の波を元にした形、灰色 = 関係の表。", ha="center", fontsize=11, color=INK)
fig.text(0.5, 0.055, "読み: 語の波では、席の印で「だれが」が 7〜11 ポイント上がり、全体で 3 ポイント上がる（揺れの外）。属性では変わらない（揺れの内）。", ha="center", fontsize=11.5, color=INK)
fig.text(0.5, 0.022, "属性の割合には「人らしさ」などが初めから入っていて、その席によく座る語の傾きを、もう持っているためと見ている。", ha="center", fontsize=11.5, color=INK)
fig.savefig(sys.argv[1], dpi=170)
