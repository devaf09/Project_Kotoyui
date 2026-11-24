"""I49zzd の図: 介入で教わった事を、本体と同じ形の外付けに持った時の数え（448 話の半分を本体に聞かせ、残り半分で問う。並べ替え 3 通り）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49zzd_taught_body.py 出力.png
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

NAMES = ("本体だけ（後半を聞かない）", "外付けを付ける: 介入 10%", "介入 30%", "介入 100%", "本体が後半も聞く（比べ役）", "介入の板（2 語の組の鍵）: 介入 100%")
COL = ("#9aa3ad", "#b7dcc0", "#6fbf8a", "#2e8b57", "#2f6fb0", "#d98c2b")
# 並べ替えごとの（本体だけ, 10%, 30%, 100%, 本体が聞く, 介入の板 100%（並べ替え 3 は数えていない））
DATA = [(.4057, .4106, .4356, .4435, .4542, .4057), (.3912, .3991, .4298, .4383, .4438, .3909), (.4002, .4083, .4282, .4431, .4502, None)]
# 介入の割合ごとの（当たりに変わった, 外れに変わった）: 並べ替え 1・2・3
MOVE = {"介入 10%": ((184, 151), (297, 244), (208, 154)), "介入 30%": ((389, 190), (415, 157), (378, 192)), "介入 100%": ((452, 200), (471, 156), (450, 165))}


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw=dict(width_ratios=(1.7, 1)))
    a = ax[0]
    for g, vals in enumerate(DATA):
        for k, v in enumerate(vals):
            if v is None:
                continue
            x = g + (k - 2.5) * .14
            a.bar(x, v, .13, color=COL[k], label=NAMES[k] if g == 0 else None)
            a.text(x, v + .003, ("%.3f" % v)[1:], ha="center", fontsize=8, rotation=90)
    a.set_xticks(range(3))
    a.set_xticklabels(["並べ替え 1（6,651 問）", "並べ替え 2（6,687 問）", "並べ替え 3（6,639 問）"], fontsize=11)
    a.set_ylim(.3, .52)
    a.set_ylabel("当たった割合（縦の軸は .30 から）")
    a.set_title("全部の問い: 外付けを付けると、教えた分だけ上がる", fontsize=12.5, loc="left")
    a.legend(loc="upper left", fontsize=9.5, ncol=2)
    a.grid(axis="y", alpha=.3)
    b = ax[1]
    for i, (name, rows) in enumerate(MOVE.items()):
        for k, (up, dn) in enumerate(rows):
            x = i + (k - 1) * .28
            b.bar(x, up, .26, color="#2e8b57", label="当たりに変わった" if (i, k) == (0, 0) else None)
            b.bar(x, -dn, .26, color="#c0504d", label="外れに変わった（害）" if (i, k) == (0, 0) else None)
            b.text(x, up + 8, str(up), ha="center", fontsize=9)
            b.text(x, -dn - 30, str(dn), ha="center", fontsize=9)
    b.axhline(0, color="#333", lw=.8)
    b.set_xticks(range(3))
    b.set_xticklabels(list(MOVE), fontsize=11)
    b.set_ylim(-330, 560)
    b.set_ylabel("問いの数（3 本は並べ替え 1・2・3）")
    b.set_title("本体だけの時から、答えの当たり外れが変わった問い", fontsize=12.5, loc="left")
    b.legend(loc="upper left", fontsize=10)
    b.grid(axis="y", alpha=.3)
    fig.suptitle("介入で教わった事を「本体と同じ形の外付け」に持つと、初めて当たりが増えた（本体の答えは 1 つも変わらない）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
