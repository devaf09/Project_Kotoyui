"""I49zz の図: 「違う」の止める強さを、手で置いた 1 と、板が出した分とで比べる（鍵の形は今まで、同じ 19 問のうち 8 問）。

数は PC で数えた集計値をそのまま置いている。問いは、動作試験のために自分で作った物。
使い方: python tools/fig_i49zz_stop_amount.py 出力.png
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

# （問い, 種類, 読む の点: 教える前, 強さ 1, 出した分（None = 0 以下で上位に出ない）, 書く の点）
DATA = [("学生が 本を", "教えた", 1.81, 1.43, 1.51, .81), ("母が 手紙を", "教えた", 1.35, 1.17, 1.21, 1.17), ("父が 本を", "教えていない\n似た問い", 1.17, 1.02, 1.05, .75),
        ("学生が 雑誌を", "教えていない\n似た問い", .84, .81, .81, .37), ("学生が 手紙を", "教えていない\n似た問い", 1.05, .64, .73, .67),
        ("学生が 歌を", "「読む は違う」\nと教えた", .79, None, None, .41), ("先生が 歌を", "教えていない\n遠い問い", .77, .14, .27, .54), ("父が 歌を", "教えていない\n遠い問い", .70, .28, .37, .55)]
COL = ("#9aa3ad", "#2f6fb0", "#3a9a5c")
NAMES = ("読む の点（教える前）", "読む の点（止める強さ 1。手で置いた値）", "読む の点（板が出した分だけ引く。置く値なし）")


def main(out):
    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    for i, d in enumerate(DATA):
        for k in range(3):
            v = d[2 + k]
            x = i + (k - 1) * .27
            ax.bar(x, v or 0, .26, color=COL[k], label=NAMES[k] if i == 0 else None)
            under = v is not None and 0 <= d[5] - v < .1
            ax.text(x, v - .08 if under else (v or 0) + .03, ("%.2f" % v) if v is not None else "0\n以下", ha="center", fontsize=9,
                    color="white" if under else "#222")
        ax.plot([i - .43, i + .43], [d[5], d[5]], color="#c0504d", lw=2.5, label="書く の点（どの形でも同じ）" if i == 0 else None)
    ax.axvline(4.5, color="#888", lw=.8, ls=":")
    ax.set_xticks(range(len(DATA)))
    ax.set_xticklabels(["%s\n%s" % (d[0], d[1]) for d in DATA], fontsize=10)
    ax.set_ylim(0, 2.3)
    ax.set_ylabel("板の点")
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.annotate("出した分なら、読む（.73）が\n書く（.67）の上に残り、巻き込まない", xy=(4.27, .78), xytext=(2.9, 1.6), fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(5.0, 1.05, "歌 の問いは、どちらの形でも 読む が\n赤い線（書く）の下まで下がる。\n板の答えが 書く に替わる所は同じ", fontsize=11)
    ax.set_title("止める強さの比べ: 板が出した分だけ引くと、遠い問いは同じように止まり、手紙 の巻き込みが無くなる", fontsize=13, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
