"""I49zzy の図（2 枚目）: 広い表と細かい表を、数えの多さを重みにして混ぜる形（κ）。持ち主の文の動詞 4,379 個。合計の数だけ。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FORMS = (("I42 のまま", "#bdbdbd"), ("直した形（細かい表を先）", "#d9a441"), ("混ぜる κ=1", "#c6d8ea"), ("混ぜる κ=3", "#7fa6cc"), ("混ぜる κ=10", "#3f73a8"))
MEANS = ("過去", "丁寧", "否定")
DATA = {"雑談 240 会話から作った部品": ((.875, .987, .956), (.957, .987, .956), (.967, .985, .968), (.967, .987, .968), (.929, .987, .968)),
        "雑談 960 会話から作った部品": ((.886, .896, .981), (.965, .896, .967), (.979, .891, .961), (.979, .896, .993), (.979, .950, .994))}


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    w = .8 / len(FORMS)
    for a, (title, rows) in zip(ax, DATA.items()):
        for j, (name, col) in enumerate(FORMS):
            for i, v in enumerate(rows[j]):
                x = i + (j - 2) * w
                a.bar(x, v, w, color=col, label=name if i == 0 else None)
                a.text(x, v + .002, ("%.3f" % v)[1:], ha="center", fontsize=8)
        a.set_xticks(range(3)); a.set_xticklabels(MEANS); a.set_ylim(.84, 1.03); a.set_title(title + "（流れだけを混ぜる形）", fontsize=10.5); a.grid(alpha=.3, axis="y")
    ax[0].set_ylabel("物差しの形と合う割合")
    fig.suptitle("数えの多さを重みにして混ぜると、過去と否定は一番良くなる。丁寧は κ に寄る（I49zzy、案 2）", fontsize=12)
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=5)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzy_mix.png"), dpi=170)


if __name__ == "__main__":
    main()
