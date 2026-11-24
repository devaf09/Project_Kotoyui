"""I49zi の図: 小さい世界で、語の位相を誤差で動かした時の広がり当て（聞いていないがありそうな 3 つ組の 1 語を当てる）。

使い方: python tools/fig_i49zi_small_grad.py 記録.json 出力.png   （記録は tools/i49zi_small_grad.py の出力）
"""
import json
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

BARS = [("固定", "隠した語", "固定（今の形）", "#8c8c8c", None),
        ("振幅だけ", "話に合う語", "振幅だけ", "#c9c9c9", None),
        ("位相・覚えた波はそのまま", "隠した語", "位相を動かす・覚えた波はそのまま（合図: 隠した語）", "#9ecae1", None),
        ("位相・覚えた波はそのまま", "話に合う語", "位相を動かす・覚えた波はそのまま（合図: 話に合う語）", "#4c72b0", None),
        ("位相・覚えた波を作り直す", "隠した語", "位相を動かす・覚えた波を作り直す（合図: 隠した語）", "#f5c9a8", None),
        ("位相・覚えた波を作り直す", "話に合う語", "位相を動かす・覚えた波を作り直す（合図: 話に合う語）", "#dd8452", None)]
KEY = "広がり当て"


def main(src, out):
    rows = json.load(open(src, encoding="utf-8"))["rows"]

    def get(T, K, p, form, teach, lr):
        for r in rows:
            if (r["話"], r["本数"], r["回"], r["形"], r["合図"]) == (T, K, p, form, teach) and (form == "固定" or abs(r["速さ"] - lr) < 1e-9):
                return r[KEY]
        return None

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1, 1, .95]))
    Ks = (4, 8, 16)
    w = .14
    for j, p in enumerate((1, 5)):
        a = ax[j]
        for k, (form, teach, name, c, _) in enumerate(BARS):
            ys = [get(40, K, p, form, teach, .03) for K in Ks]
            xs = [i + (k - 2.5) * w for i in range(len(Ks))]
            a.bar(xs, ys, w * .94, color=c, label=name)
            for x, y in zip(xs, ys):
                a.text(x, y + .01, f"{y:.2f}"[1:] if round(y, 2) < 1 else "1.0", ha="center", fontsize=6.5)
        a.set_xticks(range(len(Ks)))
        a.set_xticklabels(["位相 %d 本" % K for K in Ks])
        a.set_ylim(0, 1.3)
        a.set_yticks([0, .2, .4, .6, .8, 1.0])
        a.set_ylabel("広がり当て（聞いていない組の 1 語を当てる）")
        a.set_title("流れを %d 周（語 202、直す速さ .03）" % p)
        a.grid(axis="y", alpha=.3)
        a.legend(fontsize=7, loc="upper left")
    a = ax[2]
    lrs = (.01, .03, .1)
    for form, teach, name, c, _ in BARS[2:]:
        a.plot(range(3), [get(40, 8, 5, form, teach, lr) for lr in lrs], color=c, marker="o", lw=2,
               label=name.replace("位相を動かす・", "").replace("（合図: ", "・").replace("）", ""))
    a.axhline(get(40, 8, 5, "固定", "隠した語", 0), color="k", ls="--", lw=1, label="固定")
    a.set_xticks(range(3))
    a.set_xticklabels(["遅い（.01）", "中（.03）", "速い（.1）"])
    a.set_xlabel("位相を直す速さ")
    a.set_ylim(.4, 1.02)
    a.set_title("直す速さと合図（位相 8 本、5 周）")
    a.grid(alpha=.3)
    a.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("小さい世界: 位相を誤差で動かすと上がる。覚えた波を古いまま残しても崩れない。「話に合う答えは外れにしない」合図の方がずっと良い", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "i49zi_small_grad.png")
