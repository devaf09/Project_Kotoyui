"""I49zzy の図: 持ち主の文の動詞を、過去・丁寧・否定の形に組めるか（I42 のまま／直した形 2 通り。合計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FORMS = (("I42 のまま", "#bdbdbd"), ("直した形（細かい表を先）", "#3f73a8"), ("直した形（掛け合わせ）", "#9dbbd8"))
MEANS = ("過去", "丁寧", "否定")
OWNER = {"雑談 240 会話から作った部品": ((.875, .987, .956), (.957, .987, .956), (.904, .987, .956)),
         "雑談 960 会話から作った部品": ((.886, .896, .981), (.965, .896, .967), (.934, .896, .981))}
HELD = (("動詞\n（240）", (1229, 1234, 1233), 1549), ("形容詞\n（240）", (683, 710, 709), 837), ("動詞\n（960）", (4903, 4955, 4960), 6147), ("形容詞\n（960）", (2888, 2993, 2995), 3447))


def bars(a, labels, rows, title, lo):
    w = .8 / len(FORMS)
    for j, (name, col) in enumerate(FORMS):
        for i, v in enumerate(rows[j]):
            x = i + (j - 1) * w
            a.bar(x, v, w, color=col, label=name if i == 0 else None)
            a.text(x, v + .003, ("%.3f" % v)[1:], ha="center", fontsize=8)
    a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=9); a.set_ylim(lo, 1.02); a.set_title(title, fontsize=10.5); a.grid(alpha=.3, axis="y")


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, (name, rows) in zip(ax, OWNER.items()):
        bars(a, MEANS, rows, "持ち主の文の動詞（4,379 個）: " + name, .8)
    ax[0].set_ylabel("物差しの形と合う割合")
    bars(ax[2], [h[0] for h in HELD], [[h[1][j] / h[2] for h in HELD] for j in range(3)], "雑談の外した行: 元と同じ形に組めた割合", .7)
    ax[2].legend(fontsize=9, loc="upper left")
    fig.suptitle("形を組む所の直し: 過去の形（読んだ・遊んだ）が直る。否定は、選び方で少し下がる（I49zzy）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzy_form.png"), dpi=170)


if __name__ == "__main__":
    main()
