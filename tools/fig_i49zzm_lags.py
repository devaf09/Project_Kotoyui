"""I49zzm の図: 何個前の発話まで見るかごとの、続き方の倍率と、2 つから選ぶ当たり（run46 の集計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

LAGS = [1, 2, 3, 4]
CARRY = dict(丁寧=[1.08, 2.42, 1.06, 2.42], 過去=[2.36, 2.07, 1.64, 1.84], 問い=[0.68, 1.48, 0.82, 1.48])
STACK = [.5775, .6771, .6786, .6968]
ALONE = [.5775, .6577, .5547, .6432]


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for (k, v), c, m in zip(CARRY.items(), ("#3f73a8", "#c8452c", "#d9a441"), ("o", "s", "^")):
        a.plot(LAGS, v, m + "-", color=c, lw=2, label=k)
        for x, y in zip(LAGS, v):
            a.text(x, y + .07, "%.2f" % y, ha="center", fontsize=9, color=c)
    a.axhline(1, color="gray", ls=":")
    a.text(4.05, 1.03, "続かない時", color="gray", fontsize=9, ha="right")
    a.set_xticks(LAGS); a.set_xticklabels(["1 個前\n（相手）", "2 個前\n（自分）", "3 個前\n（相手）", "4 個前\n（自分）"])
    a.set_ylim(0, 2.9); a.set_ylabel("前の発話に有る時、今の発話にも有る倍率"); a.legend(loc="lower right"); a.grid(alpha=.3)
    a.set_title("どの状況が、だれの発話から続くか\n（2 人が交互に話すなら、偶数個前は自分）", fontsize=11)
    b.plot(LAGS, STACK, "o-", color="#3f73a8", lw=2.5, label="そこまでの発話を全部重ねる")
    b.plot(LAGS, ALONE, "s--", color="#9a9a9a", lw=1.5, label="その発話 1 つだけを見る")
    for x, y in zip(LAGS, STACK):
        b.text(x, y + .008, "%.3f" % y, ha="center", fontsize=9, color="#3f73a8")
    b.axhline(.5, color="gray", ls=":")
    b.text(4.05, .505, "当てずっぽう", color="gray", fontsize=9, ha="right")
    b.set_xticks(LAGS); b.set_xticklabels(["1 個前", "2 個前", "3 個前", "4 個前"]); b.set_ylim(.45, .74)
    b.set_ylabel("続く発話の終わりを 2 つから選ぶ当たり"); b.legend(loc="lower right"); b.grid(alpha=.3)
    b.set_title("弱い手掛かりも、重ねると強くなるか（16,886 問）", fontsize=11)
    fig.suptitle("青空文庫の会話（作品の切れ目はまたがない）: 何個前の発話まで見るか", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzm_lags.png"), dpi=170)


if __name__ == "__main__":
    main()
