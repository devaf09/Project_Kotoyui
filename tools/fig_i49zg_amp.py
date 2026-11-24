"""I49zg の図: 読む時の振幅だけを誤差で直した時の、正しい物の割合（直す速さごと）と、席ごとの変わり方。

数は PC で数えた集計値をそのまま置いている。
使い方: python tools/fig_i49zg_amp.py 出力.png
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

LR = [.003, .01, .03]
# 始め方: (直さない, 出た語・上位 3, {版: 速さごとの値（無い所は None）})
D = {
    "白紙から": (.4084, .4307, {
        "外れた時だけ直す・振幅 1 組": [.4184, .4131, .4061],
        "外れた時だけ直す・席ごと": [.4163, .4213, .4103],
        "毎回直す・振幅 1 組": [None, .4163, .4037],
        "毎回直す・席ごと": [None, .4220, .4129]}),
    "青空の 3 つ組を先に": (.3983, .4242, {
        "外れた時だけ直す・振幅 1 組": [.4171, .4100, .3964],
        "外れた時だけ直す・席ごと": [.4188, .4163, .4086],
        "毎回直す・振幅 1 組": [None, .4111, .3991],
        "毎回直す・席ごと": [None, .4201, .4172]}),
}
STY = {"外れた時だけ直す・振幅 1 組": ("#4c72b0", ":", "o"), "外れた時だけ直す・席ごと": ("#4c72b0", "-", "s"),
       "毎回直す・振幅 1 組": ("#dd8452", ":", "o"), "毎回直す・席ごと": ("#dd8452", "-", "s")}
# 席ごと（青空の 3 つ組を先に）
SEAT = [("振幅を直さない", [.477, .315, .403], "#c9c9c9"), ("毎回直す・席ごと・速さ .01", [.507, .322, .431], "#dd8452"),
        ("出た語・上位 3（比べ役）", [.552, .316, .405], "#55a868")]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
    for j, start in enumerate(D):
        a = ax[j]
        base, said, rows = D[start]
        a.axhline(said, color="#55a868", lw=1.5, ls="-.", label="出た語・上位 3（比べ役） " + f"{said:.3f}"[1:])
        a.axhline(base, color="k", lw=1, ls="--", label="振幅を直さない " + f"{base:.3f}"[1:])
        for name, ys in rows.items():
            c, ls, m = STY[name]
            xs = [i for i, y in enumerate(ys) if y is not None]
            a.plot(xs, [ys[i] for i in xs], color=c, ls=ls, marker=m, label=name)
        a.set_xticks(range(3))
        a.set_xticklabels(["遅い（.003）", "中（.01）", "速い（.03）"])
        a.set_xlabel("振幅を直す速さ")
        a.set_ylim(.375, .435)
        a.set_ylabel("正しい物の割合")
        a.set_title(f"{start}: ゆっくり直すと上がり、速いと消える")
        a.grid(alpha=.3)
        a.legend(fontsize=8, loc="lower left")
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
    a.set_title("席ごと（青空を先に）: 効く席が、出た語と違う")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="upper right")
    fig.suptitle("波は固定のまま、読む時の振幅だけを誤差で直す: 1 周だけでも 1〜2 ポイント上がる。効くのは「どうした」と「だれが」", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "i49zg_amp.png")
