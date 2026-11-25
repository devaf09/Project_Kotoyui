"""I49zj の図: 本番の流れで、語の位相を誤差で動かした時の、正しい物の割合（直す速さと合図ごと）。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/past/fig_i49zj_phase.py 出力.png
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

# 始め方: (直さない, 振幅だけ, 出た語・上位 3, 位相・隠した語, 位相・聞いた組, 位相＋振幅（速さ .1）・隠した語, 同・聞いた組)
D = {"白紙から": (.4084, .4220, .4307, [.4239, .4200, .3973], [.4171, .4131, .3970], .4267, .4274),
     "青空の 3 つ組を先に": (.3983, .4201, .4242, [.4118, .3952, .3648], [.4116, .4005, .3599], .4230, .4286)}
SEAT = [("直さない", [.477, .315, .403], "#c9c9c9"), ("振幅だけ", [.507, .322, .431], "#9ecae1"),
        ("位相＋振幅（合図: 聞いた組）", [.541, .317, .428], "#dd8452"), ("出た語・上位 3（比べ役）", [.552, .316, .405], "#55a868")]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1]))
    for j, start in enumerate(D):
        a = ax[j]
        base, amp, said, hid, seen, both_h, both_s = D[start]
        a.axhline(said, color="#55a868", lw=1.5, ls="-.", label="出た語・上位 3（比べ役） " + f"{said:.3f}"[1:])
        a.axhline(amp, color="#9ecae1", lw=1.5, ls="-", label="振幅だけ直す " + f"{amp:.3f}"[1:])
        a.axhline(base, color="k", lw=1, ls="--", label="直さない " + f"{base:.3f}"[1:])
        a.plot(range(3), hid, color="#4c72b0", marker="o", label="位相を動かす（合図: 隠した語）")
        a.plot(range(3), seen, color="#8172b3", marker="s", label="位相を動かす（合図: 聞いた組は外れにしない）")
        a.plot([.93], [both_h], color="#dd8452", marker="D", ms=9, ls="none", mfc="white", label="位相＋振幅（合図: 隠した語） " + f"{both_h:.3f}"[1:])
        a.plot([1.07], [both_s], color="#dd8452", marker="D", ms=9, ls="none", label="位相＋振幅（合図: 聞いた組） " + f"{both_s:.3f}"[1:])
        a.set_xticks(range(3))
        a.set_xticklabels(["遅い（.03）", "中（.1）", "速い（.3）"])
        a.set_xlabel("位相を直す速さ")
        a.set_ylim(.33, .44)
        a.set_ylabel("正しい物の割合")
        a.set_title(f"{start}: ゆっくりなら上がり、速いと崩れる")
        a.grid(alpha=.3)
        a.legend(fontsize=7.5, loc="lower left")
    a = ax[2]
    w = .2
    for k, (name, v, c) in enumerate(SEAT):
        xs = [i + (k - 1.5) * w for i in range(3)]
        a.bar(xs, v, w * .94, color=c, label=name)
        for x, p in zip(xs, v):
            a.text(x, p + .004, f"{p:.3f}"[1:], ha="center", fontsize=7.5)
    a.set_xticks(range(3))
    a.set_xticklabels(["だれが", "何を", "どうした"])
    a.set_ylim(.25, .68)
    a.set_title("席ごと（青空を先に）")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="upper right")
    fig.suptitle("本番の材料で語の位相を誤差で動かす: ゆっくりなら上がる。位相と振幅を一緒に直す形が一番上（.427 / .429）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zj_phase.png")
