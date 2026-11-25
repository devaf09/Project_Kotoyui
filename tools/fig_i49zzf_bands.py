"""I49zzf: 属性の帯ごとの見分け（run30）の図。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

font = os.environ.get("KOTOYUI_FONT")
if font:
    fm.fontManager.addfont(font)
    plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
plt.rcParams["axes.unicode_minus"] = False

# 席 → (全部の帯の見分け, {帯: (その帯だけ, その帯を抜く, 力の取り分, 近さの取り分)})
D = {
    "だれが（363 語）": (.7295, {"A": (.5846, .7300, .0936, .0577), "B": (.5270, .7320, .2116, .0102), "C": (.6126, .7254, .1504, .0085),
                           "D": (.6935, .6298, .4068, .1376), "E": (.4989, .7182, .1045, .0014), "G": (.5933, .7263, .0331, .0224)}),
    "何を（1,052 語）": (.6174, {"A": (.5953, .6179, .1174, .0643), "B": (.5619, .5990, .2052, .0117), "C": (.6151, .5976, .1660, .0054),
                            "D": (.5936, .5746, .3060, .0385), "E": (.5132, .6333, .1730, .0040), "G": (.5523, .6216, .0323, .0197)}),
    "どうした（461 語）": (.6315, {"A": (.6051, .6619, .2114, .1545), "B": (.6764, .5658, .2577, .0158), "C": (.7107, .5721, .2572, .0378),
                             "D": (.5112, .7232, .2187, .0136), "G": (.5907, .6329, .0550, .0421)}),
}
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey="row")
for j, (seat, (full, bands)) in enumerate(D.items()):
    names = list(bands)
    x = range(len(names))
    ax = axes[0][j]
    ax.bar([i - .2 for i in x], [bands[b][0] for b in names], .4, color="#4477aa", label="その帯だけで測る")
    ax.bar([i + .2 for i in x], [bands[b][1] for b in names], .4, color="#ee7733", label="その帯を抜いて測る")
    ax.axhline(full, color="black", ls="--", lw=1, label="全部の帯")
    ax.axhline(.5, color="gray", ls=":", lw=1)
    ax.set_ylim(.45, .78)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_title(seat)
    ax.grid(axis="y", alpha=.3)
    if j == 0:
        ax.set_ylabel("同じ使われ方の語を近くに置けている割合\n（.5 = 見分けられない）")
    if j == 1:
        ax.legend(loc="upper right", fontsize=9)
    ax = axes[1][j]
    ax.bar([i - .2 for i in x], [bands[b][2] for b in names], .4, color="#999999", label="波の力に占める割合")
    ax.bar([i + .2 for i in x], [bands[b][3] / sum(v[3] for v in bands.values()) for b in names], .4, color="#cc3311", label="近さ（cos）に占める割合")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.grid(axis="y", alpha=.3)
    if j == 0:
        ax.set_ylabel("割合")
        ax.legend(loc="upper left", fontsize=8)
fig.suptitle("語板の属性の帯ごとに、語の近さへの効き方を数える（上: 見分け、下: 近さを決めている帯）")
fig.tight_layout()
for out in ("docs/images/i49zzf_bands.png", "/mnt/project-files/kotoyui_intervention/figs/i49zzf_bands.png"):
    fig.savefig(out, dpi=170)
