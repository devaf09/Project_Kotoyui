"""I49zf の図: 並べた板（上位 3 から選ぶ）に、条件を 1 つずつ足した時の、正しい物の割合。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49zf_gate.py 出力.png
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

STEPS = ["いつも板に一番近い語に替える\n（前の話の語も板に残る）",
         "＋ 話ごとに板を空に戻す",
         "＋ 板に同じ波が有る候補だけ",
         "＋ 似た波は見ず、同じ波の\n重さ（新しさ）だけで選ぶ",
         "＋ 板を席ごとに分ける",
         "決まりで書いた「出た語」"]
D = {"白紙から": (.4084, [.3647, .3939, .4047, .4174, .4214, .4307]),
     "青空の 3 つ組を先に": (.3983, [.3661, .3954, .4073, .4134, .4163, .4242])}
COL = ["#c44e52", "#dd8452", "#dd8452", "#dd8452", "#4c72b0", "#55a868"]
# 席ごとの割合（白紙から）: 板無し、板・席ごと・同じ波の重さ・話ごとに空、出た語
SEAT = [("板無し", [.505, .307, .413], "#c9c9c9"), ("並べた板（席ごと・同じ波の重さ）", [.560, .307, .398], "#4c72b0"),
        ("決まりで書いた「出た語」", [.554, .314, .424], "#55a868")]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1.25, .9]))
    for j, start in enumerate(D):
        a = ax[j]
        base, ys = D[start]
        y = list(range(len(STEPS)))[::-1]
        a.barh(y, [v - .35 for v in ys], left=.35, color=COL, height=.62)
        a.axvline(base, color="k", lw=1, ls="--")
        a.text(base, len(STEPS) - .45, "板無し " + f"{base:.3f}"[1:], ha="center", va="bottom", fontsize=9)
        for yy, v in zip(y, ys):
            a.text(v + .001, yy, f"{v:.3f}"[1:], va="center", fontsize=9, bbox=dict(fc="white", ec="none", pad=1.2))
        a.set_yticks(y)
        a.set_yticklabels(STEPS if j == 0 else [""] * len(STEPS), fontsize=9)
        a.set_xlim(.35, .445)
        a.set_ylim(-.6, len(STEPS) - .1)
        a.set_xlabel("正しい物の割合")
        a.set_title(start)
        a.grid(axis="x", alpha=.3)
    a = ax[2]
    w = .26
    for k, (name, v, c) in enumerate(SEAT):
        xs = [i + (k - 1) * w for i in range(3)]
        a.bar(xs, v, w * .94, color=c, label=name)
        for x, p in zip(xs, v):
            a.text(x, p + .004, f"{p:.3f}"[1:], ha="center", fontsize=8)
    a.set_xticks(range(3))
    a.set_xticklabels(["だれが", "何を", "どうした"])
    a.set_ylim(.25, .66)
    a.set_title("席ごと（白紙から）: 残る差は「どうした」")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="upper right")
    fig.suptitle("板が崩れていた理由: 前の話の語まで拾う事と、出た語が無くても替える事。直すと板無しを越えるが、決まりの形にはまだ少し届かない", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zf_gate.png")
