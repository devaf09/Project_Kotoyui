"""I49zzo の図: 状況波を入れ替えた時の、上位 50 の終わりに印が付く割合（run49 の集計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

WAVES = ["間柄の波\n（丁寧の印）", "時の波\n（過去の印）", "言葉づかいの波\n（古風の印）", "やり取りの波\n（問いの印）"]
SIDES = [("丁寧", "くだけ"), ("過去", "今"), ("古風", "今風"), ("問われた", "言い切られた")]
DATA = {"同じ話し手が続ける時（候補 177）": ([.96, .26, .38, .28], [.00, .00, .00, .24], [.6328, .0904, .1638, .2147]),
        "次の話し手が受ける時（候補 318）": ([.58, .50, .22, .06], [.36, .02, .04, .44], [.4654, .0786, .1384, .1761])}


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    for a, (title, (pos, neg, base)) in zip(ax, DATA.items()):
        for i in range(4):
            a.bar(i - .2, pos[i], .4, color="#3f73a8", label="片方の状況を入れた時" if i == 0 else None)
            a.bar(i + .2, neg[i], .4, color="#d9a441", label="反対の状況を入れた時" if i == 0 else None)
            a.text(i - .2, pos[i] + .02, "%s\n%.2f" % (SIDES[i][0], pos[i]), ha="center", fontsize=9)
            a.text(i + .2, neg[i] + .02, "%s\n%.2f" % (SIDES[i][1], neg[i]), ha="center", fontsize=9)
            a.plot([i - .42, i + .42], [base[i], base[i]], color="#333333", lw=1.5, ls="--", label="候補の全体での割合" if i == 0 else None)
        a.set_xticks(range(4)); a.set_xticklabels(WAVES); a.set_ylim(0, 1.18); a.set_title(title, fontsize=11)
        a.grid(axis="y", alpha=.3)
    ax[0].set_ylabel("上位 50 の終わりに、その印が付く割合"); ax[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("状況波を入れ替えると、出る文の終わりが変わるか（青空文庫の会話で学んだ波）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzo_steer.png"), dpi=170)


if __name__ == "__main__":
    main()
