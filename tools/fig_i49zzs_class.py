"""I49zzs の図 2: 席ごとに語が違う世界（語に類が有る）で、聞いていない組が当たるか（run61 の数。小さい世界、4096 次元、表は席ごと）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HOLDS = ("2 割", "5 割", "8 割")
# 世界 → (当てずっぽう, 掛け合わせて数える, 動かす .03, .1, .3, でたらめの世界で .3)
DATA = {"類 2 × 4 語（組 64）": (.5, [.5, .5312, .4706], [.6667, .5938, .451], [.9167, .75, .451], [.8333, .625, .5686], [.4167, .5, .549]),
        "類 4 × 5 語（組 400）": (.25, [.175, .245, .2375], [.525, .295, .2062], [.8, .455, .2344], [.9375, .74, .25], [.2875, .27, .2687]),
        "類 4 × 10 語（組 1,600）": (.25, [.3063, .265, .2539], [.5094, .3362, .2289], [.8812, .5737, .2719], [.9875, .9175, .3695], [.2125, .2412, .25]),
        "類 8 × 5 語（組 1,600）": (.125, [.1406, .1288, .1211], [.2469, .15, .1047], [.7562, .4037, .1445], [.9719, .7925, .1844], [.1156, .1288, .1164])}
LINES = (("掛け合わせて数える", "#d9a441", "-", 2.2, "s"), ("外れた時だけ動かす（大きさ .03）", "#9db8d6", "-", 1.8, "o"), ("同（.1）", "#6a93c2", "-", 2.0, "o"),
         ("同（.3）", "#274f7f", "-", 2.6, "o"), ("でたらめの世界で、同（.3）", "#c0504d", "--", 1.4, "x"))


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 4, figsize=(19, 5), sharey=True)
    for a, (name, d) in zip(ax, DATA.items()):
        a.axhline(d[0], color="k", lw=1, ls=":")
        for (lab, col, ls, lw, mk), ys in zip(LINES, d[1:]):
            a.plot(range(3), ys, color=col, ls=ls, lw=lw, marker=mk, label=lab)
        a.set_xticks(range(3)); a.set_xticklabels(HOLDS); a.set_xlabel("隠して置いた組の割合"); a.set_ylim(0, 1.05); a.set_title(name, fontsize=10.5); a.grid(alpha=.3)
    ax[0].set_ylabel("聞いていない組の当たり（点線は当てずっぽう）"); ax[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("席ごとに語が違っても、語に類が有れば、外れた時だけ位相を動かす形は聞いていない組に届く（I49zzs、誰が・何を・どうする の小さい世界）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzs_class.png"), dpi=170)


if __name__ == "__main__":
    main()
