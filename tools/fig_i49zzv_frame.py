"""I49zzv の図: 意味の印で骨組みを決め、丁寧の有り無しだけを替える（run67 雑談、run68 青空文庫の集計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

SETS = ("雑談（MRMP、2,979 行）", "青空文庫の会話（2,120 行）")
COL = ("#d9a441", "#3f73a8")


def bars(a, labels, rows, title, ylim, fmt="%.2f"):
    w = .8 / len(rows)
    for j, (name, vals) in enumerate(rows):
        xs = [i + (j - (len(rows) - 1) / 2) * w for i in range(len(labels))]
        a.bar(xs, vals, w, color=COL[j], label=name)
        for x, v in zip(xs, vals):
            a.text(x, v + ylim * .012, fmt % v, ha="center", fontsize=8.5)
    a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=8.5); a.set_ylim(0, ylim); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    bars(ax[0], ("文字のまま\n書き替える\n（守らない）", "文字のまま\n（中身を守る、\n下限 20）", "意味の印で\n骨組みを決める\n（文字の印で数える）", "意味の印で\n骨組みを決める\n（意味の印で数える）"),
         ((SETS[0], (.1213, .0843, .0208, 0.0)), (SETS[1], (.2253, .1833, .0476, 0.0))), "1. 打ち消しが入れ替わった割合", .3)
    ax[0].legend(fontsize=9, loc="upper right")
    bars(ax[1], ("組めた", "元と同じ意味の\n組から組んだ形が、\n元の行と同じ", "作り替えた行が\n違和感チェック\nを通る", "本物の行が\n違和感チェック\nを通る"),
         ((SETS[0], (.9923, .86, .3575, .4005)), (SETS[1], (.9925, .6863, .508, .5642))), "2. 形の正しさ", 1.12)
    bars(ax[2], ("状況波で選ぶ", "前の行の丁寧の印に\n合わせる", "いつも多い方を選ぶ"),
         ((SETS[0], (.5184, .5313, .5507)), (SETS[1], (.5521, .534, .5982))), "3. 丁寧の有り無しが、本物の行と合う割合（前の行から選ぶ）", .75)
    ax[2].axhline(.5, color="k", lw=1, ls=":")
    fig.suptitle("打ち消し・過去は意味の印で先に決め、丁寧の有り無しだけを替える: 打ち消しは入れ替わらない。丁寧は、前の行からは選べない（I49zzv）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzv_frame.png"), dpi=170)


if __name__ == "__main__":
    main()
