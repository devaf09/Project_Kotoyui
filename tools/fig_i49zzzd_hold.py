"""I49zzzd の図: 持った波で「だれが」を選ぶ（半減期ごと）と、自分で続けた時の場面の物差し（持ち主の 3 つの文。run76。合計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

N = 499
HALVES = (1, 4, 16, 64, 256, 1024)
WHO = (("持った波との共鳴（属性の波）", (218, 179, 93, 14, 7, 7), "#3f73a8", "o"),
       ("持った波との共鳴（語そのもの）", (218, 178, 93, 39, 20, 18), "#e08a2c", "s"),
       ("持った波で遷移を引く", (121, 79, 34, 18, 18, 18), "#6aa56a", "^"))
HAND, LAST = 240, 108
ACT = (33, 22, 18, 23, 31, 31)
ACT_LAST = 35
# 自分で続ける（12 個の話 × 10 回 = 120 個の動き）。出した動きが、初めの行の動きと同じ話に出た事が有る数／隣の動きと同じ話に出た事が有る数。
SCENE = (("本物の話\n（95 個）", 47 / 95, 48 / 95), ("前の行から", 59 / 120, 110 / 120), ("半減期 1", 62 / 120, 107 / 120), ("4", 71 / 120, 112 / 120),
         ("16", 71 / 120, 107 / 120), ("64", 71 / 120, 110 / 120), ("256", 73 / 120, 112 / 120), ("1024", 74 / 120, 115 / 120))


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2), gridspec_kw=dict(width_ratios=(3, 2.2, 3.4)))
    x = range(len(HALVES))
    a = ax[0]
    a.axhline(HAND / N, color="#555555", ls="--", lw=1.2, label="今までの形（話の中の人を集めて、遷移の波で選ぶ。話の切れ目を教える） %.3f" % (HAND / N))
    a.axhline(LAST / N, color="#aaaaaa", ls=":", lw=1.4, label="前の行だけから遷移の波で引く %.3f" % (LAST / N))
    for name, v, col, mk in WHO:
        a.plot(x, [k / N for k in v], color=col, marker=mk, label=name)
    for i, k in enumerate(WHO[0][1]):
        a.text(i, k / N + (.018 if i < 3 else -.035), "%.3f" % (k / N), ha="center", fontsize=8.5, color="#3f73a8")
    a.set_ylim(0, .6); a.set_title("1. だれが（広い候補 519 語、499 問）", fontsize=10.5)
    a.set_ylabel("当たりの割合"); a.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=1)
    a = ax[1]
    a.axhline(ACT_LAST / N, color="#aaaaaa", ls=":", lw=1.4, label="前の行だけから %.3f" % (ACT_LAST / N))
    a.plot(x, [k / N for k in ACT], color="#6aa56a", marker="^", label="持った波で遷移を引く")
    for i, k in enumerate(ACT):
        a.text(i, k / N + .006, "%.3f" % (k / N), ha="center", fontsize=8.5)
    a.set_ylim(0, .15); a.set_title("2. どうした（広い候補 569 語、499 問）", fontsize=10.5); a.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=1)
    for a in ax[:2]:
        a.set_xticks(list(x)); a.set_xticklabels([str(h) for h in HALVES]); a.set_xlabel("半減期（行の数）"); a.grid(alpha=.3)
    a = ax[2]
    for i, (name, s, p) in enumerate(SCENE):
        for j, (v, col, lab) in enumerate(((s, "#3f73a8", "初めの行の動きと、同じ話に出た事が有る"), (p, "#bdbdbd", "隣の行の動きと、同じ話に出た事が有る"))):
            a.bar(i + (j - .5) * .38, v, .38, color=col, label=lab if i == 0 else None)
            a.text(i + (j - .5) * .38, v + .015, "%.2f" % v, ha="center", fontsize=8)
    a.set_xticks(range(len(SCENE))); a.set_xticklabels([s[0] for s in SCENE], fontsize=8.5); a.set_ylim(0, 1.12); a.grid(alpha=.3, axis="y")
    a.set_xlabel("動きの引き方（右の 6 つは、持った波で引く。半減期）")
    a.set_title("3. 自分で続けた時の場面の物差し（12 個の話、少ない）", fontsize=10.5); a.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=1)
    fig.suptitle("「だれが」は短く持つ波で選べる（.437）。長く持つと前の話と混ざって外れる。動きと場面には、持った波は効かなかった（I49zzzd）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzzd_hold.png"), dpi=170)


if __name__ == "__main__":
    main()
