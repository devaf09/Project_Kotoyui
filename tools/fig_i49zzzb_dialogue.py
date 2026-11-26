"""I49zzzb の図: 会話（雑談 960 会話、約 10 万行）で、だれが・どうした がどれだけ引き継がれるか。持ち主の文（77 節）との比べ。合計の数だけ。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

KINDS = (("続く（前と同じ語）", "#5a9a68"), ("1 つ前の別の語へ戻る", "#9cc7a5"), ("もっと前に出た語へ戻る", "#d9c58a"), ("初めての語", "#3f73a8"))
WHO = {"持ち主の文\n（1 人が書いた話）": (.463, .213, .091, .233), "雑談\n（だれが を言った行だけ）": (.067, .040, .103, .789)}
SAME = (("相手の行 → 返事", "#3f73a8", (.093, .090)), ("同じ話し手の\n1 個前の行 → 今の行", "#9dbbd8", (.059, .042)), ("離れた行（比べ役）", "#bdbdbd", (.021, .019)))
HAVE = (("だれが", .164), ("何を", .012), ("どうした", .585))


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    a = ax[0]
    for i, (name, v) in enumerate(HAVE):
        a.bar(i, v, .6, color="#7fa6cc"); a.text(i, v + .012, "%.3f" % v, ha="center", fontsize=9)
    a.set_xticks(range(3)); a.set_xticklabels([h[0] for h in HAVE]); a.set_ylim(0, .7); a.set_title("1. 雑談の行で、その席が取れた割合", fontsize=10.5); a.grid(alpha=.3, axis="y"); a.set_ylabel("割合")
    a = ax[1]
    for i, vals in enumerate(WHO.values()):
        bottom = 0.0
        for (name, col), v in zip(KINDS, vals):
            a.bar(i, v, .55, bottom=bottom, color=col, label=name if i == 0 else None)
            if v >= .04:
                a.text(i, bottom + v / 2, "%.2f" % v, ha="center", va="center", fontsize=9, color="white" if col in ("#3f73a8", "#5a9a68") else "black")
            bottom += v
    a.set_xticks(range(2)); a.set_xticklabels(list(WHO), fontsize=9); a.set_ylim(0, 1); a.set_title("2. だれが: 次の語はどこから来るか", fontsize=10.5)
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2)
    a = ax[2]
    w = .8 / len(SAME)
    for j, (name, col, vals) in enumerate(SAME):
        for i, v in enumerate(vals):
            x = i + (j - 1) * w
            a.bar(x, v, w, color=col, label=name.replace("\n", "") if i == 0 else None); a.text(x, v + .002, "%.3f" % v, ha="center", fontsize=9)
    a.set_xticks(range(2)); a.set_xticklabels(("だれが", "どうした")); a.set_ylim(0, .13); a.set_title("3. 雑談: 2 つの行で同じ語だった割合", fontsize=10.5); a.grid(alpha=.3, axis="y")
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=1)
    fig.suptitle("会話では、主語を言う行は 6 行に 1 行。言った主語の 8 割は初めての語で、持ち主の文とは逆（I49zzzb、案 3-2）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzzb_dialogue.png"), dpi=170)


if __name__ == "__main__":
    main()
