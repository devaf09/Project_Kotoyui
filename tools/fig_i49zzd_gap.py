"""I49zzd: 2 つの答えが違った問いを、選ぶ時に見た数えの差ごとに分けた数（run33、並べ替え 3 通りの合計）の図。"""
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

names = ["−3 より下", "−3〜−2", "−2〜−1", "−1〜0", "0", "0〜1", "1〜2", "2〜3", "3 以上"]
D = {  # 割合 → 区切りごとの (外付けだけ当たり, 本体だけ当たり)
    "介入 10%": [(98, 865), (110, 1024), (98, 762), (330, 731), (516, 882), (405, 367), (284, 182), (0, 0), (0, 0)],
    "介入 30%": [(126, 1248), (109, 469), (160, 596), (496, 804), (275, 338), (451, 288), (600, 229), (104, 18), (27, 4)],
    "介入 100%": [(171, 1656), (136, 392), (303, 628), (395, 510), (101, 148), (327, 199), (406, 158), (274, 72), (366, 92)],
}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
for ax, (title, rows) in zip(axes, D.items()):
    x = range(len(names))
    ax.bar([i - .2 for i in x], [r[0] for r in rows], .4, color="#4477aa", label="外付けだけ当たり")
    ax.bar([i + .2 for i in x], [r[1] for r in rows], .4, color="#ee7733", label="本体だけ当たり")
    ax.axvline(4.5, color="black", ls="--", lw=1)
    ax.text(4.6, ax.get_ylim()[1] * .02 + 1500, "ここから右は\n外付けの答えを使う", fontsize=9, va="top")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_title(title)
    ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("問いの数（並べ替え 3 通りの合計）")
axes[2].legend(loc="upper right", fontsize=9)
fig.suptitle("本体と外付けの答えが違った問いを、選ぶ時に見た数えの差ごとに分ける（横: (外付けの当たり − 本体の当たり) ÷ √(合計)）")
fig.tight_layout()
for out in ("docs/images/i49zzd_gap.png", "/mnt/project-files/kotoyui_intervention/figs/i49zzd_gap.png"):
    fig.savefig(out, dpi=170)
