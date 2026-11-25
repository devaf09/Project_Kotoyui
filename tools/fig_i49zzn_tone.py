"""I49zzn の図: 相手で変わる丁寧さと、口調から関係・話し手を当てる試し（run47 の集計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.9))
    a = ax[0]
    a.bar([0, 1], [.1087, .7473], color=["#d9a441", "#3f73a8"], width=.55)
    for i, v in enumerate([.1087, .7473]):
        a.text(i, v + .02, "%.0f%%" % (v * 100), ha="center", fontsize=12)
    a.set_xticks([0, 1]); a.set_xticklabels(["家族宛て\n（2,623 発言）", "初対面の人宛て\n（4,979 発言）"]); a.set_ylim(0, .95)
    a.set_ylabel("丁寧な終わりの割合")
    a.set_title("同じ人が、相手で話し方を変えるか\n（両方に 10 回以上話した 12 人の全員が同じ向き）", fontsize=10.5)
    a = ax[1]
    a.bar([0, 1], [.8303, .6515], color=["#3f73a8", "#bdbdbd"], width=.55)
    for i, v in enumerate([.8303, .6515]):
        a.text(i, v + .015, "%.3f" % v, ha="center")
    a.set_xticks([0, 1]); a.set_xticklabels(["文の終わりから当てる", "多い方を言い続ける"]); a.set_ylim(.4, .95)
    a.set_ylabel("当たりの割合")
    a.set_title("口調から関係を当てる\n（宛先が家族か初対面か、3,796 問）", fontsize=10.5)
    a = ax[2]
    x = [0, 1]
    a.bar([i - .2 for i in x], [.3855, .3894], .4, color="#3f73a8", label="文の終わりから当てる")
    a.bar([i + .2 for i in x], [.4006, .4039], .4, color="#bdbdbd", label="一番多く話した人を言い続ける")
    for i, (u, v) in enumerate([(.3855, .4006), (.3894, .4039)]):
        a.text(i - .2, u + .006, "%.3f" % u, ha="center"); a.text(i + .2, v + .006, "%.3f" % v, ha="center")
    a.axhline(1 / 3, color="gray", ls=":"); a.text(1.45, .337, "当てずっぽう", color="gray", ha="right", fontsize=9)
    a.set_xticks(x); a.set_xticklabels(["家族入りの会話", "初対面の会話"]); a.set_ylim(.25, .47)
    a.set_ylabel("当たりの割合"); a.legend(loc="upper right", fontsize=8.5)
    a.set_title("口調から話し手を当てる\n（3 人から。約 2.5 万問ずつ）", fontsize=10.5)
    fig.suptitle("話し手の印の有る 3 人の雑談（MRMP、960 会話）: 口調は「だれが」ではなく「だれに」で決まる", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzn_tone.png"), dpi=170)


if __name__ == "__main__":
    main()
