"""I49zzr の図: 状況波を自然に出た向きへずらして書き替え、違和感チェックに通した結果（run51 雑談、run52 青空文庫の集計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

NS = [3, 4, 5]
CHECK = {"雑談（MRMP）": dict(本物=[.8076, .5705, .357], 壊した=[.2361, .0646, .0293], 点=[78.6, 75.3, 66.4]),
         "青空文庫の会話": dict(本物=[.8804, .67, .4524], 壊した=[.3719, .0925, .0324], 点=[75.4, 78.9, 71.0])}
PASS = {"雑談（MRMP）": (.6454, .67), "青空文庫の会話": (.794, .79)}
NEG = {"雑談（MRMP）": 211 / 1740, "青空文庫の会話": 397 / 1762}
# 青空文庫（run52）: 向きごとの、両端 30 個の終わりに付く印の割合（正の側, 負の側）
ENDS = [("向き 1", dict(丁寧=(.3, .8), 過去=(.1, .0), 問い=(.3333, .1667))),
        ("向き 2", dict(丁寧=(.4, .8333), 過去=(.0, .6667), 問い=(.1667, .2))),
        ("向き 3", dict(丁寧=(1.0, .8333), 過去=(.0, .2333), 問い=(.3, .0667)))]
COL = {"雑談（MRMP）": "#d9a441", "青空文庫の会話": "#3f73a8"}


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.6))
    a = ax[0]
    for j, (name, d) in enumerate(CHECK.items()):
        for i, n in enumerate(NS):
            x = i + (j - .5) * .42
            a.bar(x - .1, d["本物"][i], .2, color=COL[name], label=name + "・本物の文を通す" if i == 0 else None)
            a.bar(x + .1, d["壊した"][i], .2, color=COL[name], alpha=.4, hatch="//", label=name + "・壊した文を通す" if i == 0 else None)
            a.text(x - .1, d["本物"][i] + .015, ("%.2f" % d["本物"][i]).lstrip("0"), ha="center", fontsize=8)
            a.text(x + .1, d["壊した"][i] + .015, ("%.2f" % d["壊した"][i]).lstrip("0"), ha="center", fontsize=8)
            a.text(x, 1.02, "%.0f 点" % d["点"][i], ha="center", fontsize=9, color=COL[name])
    a.set_xticks(range(3)); a.set_xticklabels(["文字 %d つの並び" % n for n in NS]); a.set_ylim(0, 1.42)
    a.set_ylabel("通す割合"); a.set_title("1. 違和感チェックそのものの確かめ", fontsize=11); a.grid(axis="y", alpha=.3); a.legend(fontsize=8, loc="upper right")
    a = ax[1]
    for j, name in enumerate(PASS):
        w, c = PASS[name]
        a.bar(j - .27, w, .25, color=COL[name], label="状況波で選んだ書き替えが、チェックを通る" if j == 0 else None)
        a.bar(j, c, .25, color=COL[name], alpha=.4, hatch="//", label="状況波を見ずに選んだ書き替えが、チェックを通る" if j == 0 else None)
        a.bar(j + .27, NEG[name], .25, color="#b0413e", label="打ち消しの印が入れ替わった書き替え" if j == 0 else None)
        for x, v in ((j - .27, w), (j, c), (j + .27, NEG[name])):
            a.text(x, v + .015, ("%.2f" % v).lstrip("0"), ha="center", fontsize=9)
    a.set_xticks(range(2)); a.set_xticklabels(list(PASS)); a.set_ylim(0, 1.12); a.set_ylabel("書き替えた行のうちの割合")
    a.set_title("2. 材料の 300 文 × 6 つの状況の書き替え", fontsize=11); a.grid(axis="y", alpha=.3); a.legend(fontsize=8, loc="upper right")
    a = ax[2]
    marks = ("丁寧", "過去", "問い")
    mc = ("#3f73a8", "#d9a441", "#5a9a68")
    for i, (name, d) in enumerate(ENDS):
        for k, m in enumerate(marks):
            pos, neg = d[m]
            x = i + (k - 1) * .28
            a.bar(x - .065, pos, .13, color=mc[k], label="%sの印・正の側" % m if i == 0 else None)
            a.bar(x + .065, neg, .13, color=mc[k], alpha=.4, hatch="//", label="%sの印・負の側" % m if i == 0 else None)
            a.text(x - .065, pos + .015, ("%.1f" % pos).lstrip("0") or "0", ha="center", fontsize=7.5)
            a.text(x + .065, neg + .015, ("%.1f" % neg).lstrip("0") or "0", ha="center", fontsize=7.5)
    a.set_xticks(range(3)); a.set_xticklabels([n for n, _ in ENDS]); a.set_ylim(0, 1.3); a.set_ylabel("両端 30 個の終わりに、印が付く割合")
    a.set_title("3. 自然に出た向きの両端（青空文庫。名前は付けていない）", fontsize=11); a.grid(axis="y", alpha=.3); a.legend(fontsize=7.5, loc="upper right", ncol=3)
    fig.suptitle("状況波（位相の形）を、自然に出た向きへずらして文の終わりを書き替え、違和感チェックに通す（I49zzr）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzr_air.png"), dpi=170)


if __name__ == "__main__":
    main()
