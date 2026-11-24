"""I49zz の図: 「学生が 歌を に、板の 読む は違う」と 1 回教えた前後の、板の点（鍵の形は今まで）。

数は PC で数えた集計値をそのまま置いている。問いは、動作試験のために自分で作った物。
使い方: python tools/fig_i49zz_deny.py 出力.png
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

# （問い, 種類, 読む の点: 前, 後（None = 0 以下で上位に出ない）, 書く の点）
DATA = [("学生が 本を", "教えた", 1.81, 1.43, .81), ("母が 手紙を", "教えた", 1.35, 1.18, 1.17), ("父が 本を", "教えていない\n似た問い", 1.17, 1.02, .75),
        ("学生が 雑誌を", "教えていない\n似た問い", .84, .81, .37), ("学生が 手紙を", "教えていない\n似た問い", 1.05, .64, .67),
        ("学生が 歌を", "「読む は違う」\nと教えた", .79, None, .41), ("先生が 歌を", "教えていない\n遠い問い", .77, .14, .54), ("父が 歌を", "教えていない\n遠い問い", .70, .28, .55)]


def main(out):
    fig, ax = plt.subplots(figsize=(14, 6.2))
    for i, (q, kind, b, a, w) in enumerate(DATA):
        ax.bar(i - .2, b, .38, color="#9aa3ad", label="読む の点（教える前）" if i == 0 else None)
        ax.bar(i + .2, a or 0, .38, color="#2f6fb0", label="読む の点（「違う」を教えた後）" if i == 0 else None)
        ax.plot([i - .42, i + .42], [w, w], color="#c0504d", lw=2.5, label="書く の点（前後で同じ）" if i == 0 else None)
        ax.text(i - .2, b + .03, "%.2f" % b, ha="center", fontsize=10)
        near = a is not None and abs(a - w) < .08
        ax.text(i + .2, (a or 0) - .09 if near else (a or 0) + .03, "%.2f" % a if a is not None else "0 以下", ha="center", fontsize=10,
                color="white" if near else "#2f6fb0")
    ax.axvline(4.5, color="#888", lw=.8, ls=":")
    ax.set_xticks(range(len(DATA)))
    ax.set_xticklabels(["%s\n%s" % (d[0], d[1]) for d in DATA], fontsize=10)
    ax.set_ylim(0, 2.25)
    ax.set_ylabel("板の点")
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.text(5.0, 1.25, "歌 の問いは 3 つとも 読む が止まった。\nただ、赤い線（書く）が上に残り、\n板の答えは 書く に替わっただけ", fontsize=11)
    ax.annotate("巻き込み: 読む が 書く を\nわずかに下回り、外れに替わった", xy=(4.2, .72), xytext=(3.0, 1.75), fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("「板のこの答えは違う」を 1 回教えた前後: 遠い問いの 読む は止まるが、書く が残り、手紙 を巻き込む", fontsize=13, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
