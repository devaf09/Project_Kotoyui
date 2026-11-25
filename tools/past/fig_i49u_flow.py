"""I49u の図: 並んだ上位から話の流れで選び直した時の、一位の割合と、当たり・外れに変わった数。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49u_flow.py 出力.png
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

NAMES = ["そのまま", "流れ・九割以内", "流れ・上位 3", "流れ・上位 5", "流れ・順位を足す",
         "流れ（自分の分を引く）\n・上位 3", "出た語・上位 3"]
# (一位, 当たりに変わった数, 外れに変わった数)
D = {
    "白紙から": {
        "関係の表": [(.4321, 0, 0), (.4326, 64, 60), (.3220, 330, 1236), (.2892, 327, 1503), (.3394, 214, 977), (.1995, 364, 2278), (.4276, 438, 475)],
        "解く・属性": [(.4084, 0, 0), (.4020, 386, 439), (.3265, 712, 1386), (.2999, 691, 1584), (.3499, 464, 946), (.2253, 623, 2130), (.4307, 874, 691)],
    },
    "青空の 3 つ組を先に": {
        "関係の表": [(.4533, 0, 0), (.4549, 67, 54), (.3361, 375, 1339), (.3019, 376, 1622), (.3506, 231, 1076), (.2181, 373, 2308), (.4402, 456, 564)],
        "解く・属性": [(.3983, 0, 0), (.3949, 489, 517), (.3361, 806, 1318), (.3163, 838, 1513), (.3404, 547, 1024), (.2321, 668, 2036), (.4242, 919, 706)],
    },
}
COL = {"関係の表": "#4c72b0", "解く・属性": "#dd8452"}


def main(out):
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ys = list(range(len(NAMES)))[::-1]
    h = .38
    for j, start in enumerate(D):
        a = ax[0][j]
        for k, (form, rows) in enumerate(D[start].items()):
            pos = [y + (.5 - k) * h for y in ys]
            a.barh(pos, [r[0] for r in rows], h, color=COL[form], label=form)
            a.axvline(rows[0][0], color=COL[form], ls=":", lw=1)
            for p, r in zip(pos, rows):
                a.text(r[0] + .004, p, f"{r[0]:.3f}"[1:], va="center", fontsize=8)
        a.set_yticks(ys)
        a.set_yticklabels(NAMES, fontsize=9)
        a.set_xlim(.15, .5)
        a.set_xlabel("一位が正解の割合（点線は選び直さない時）")
        a.set_title(f"{start}: 一位の割合")
        a.legend(loc="center right", fontsize=9)
        a.grid(axis="x", alpha=.3)
        b = ax[1][j]
        rows = D[start]["解く・属性"]
        b.barh(ys, [r[1] for r in rows], .6, color="#55a868", label="当たりに変わった")
        b.barh(ys, [-r[2] for r in rows], .6, color="#c44e52", label="外れに変わった")
        for y, r in zip(ys, rows):
            if r[1] or r[2]:
                b.text(r[1] + 30, y, str(r[1]), va="center", fontsize=8)
                b.text(-r[2] - 30, y, str(r[2]), va="center", ha="right", fontsize=8)
        b.axvline(0, color="k", lw=.8)
        b.set_yticks(ys)
        b.set_yticklabels(NAMES, fontsize=9)
        b.set_xlim(-2700, 1500)
        b.set_xlabel("問いの数（全 8,229 問）")
        b.set_title(f"{start}: 解く・属性で答えが変わった問い")
        b.legend(loc="upper left", fontsize=9)
        b.grid(axis="x", alpha=.3)
    fig.suptitle("並んだ上位から選び直す: 流れの波では壊す方が多い。「この話でもう出た語」だけが上積みになった", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49u_flow.png")
