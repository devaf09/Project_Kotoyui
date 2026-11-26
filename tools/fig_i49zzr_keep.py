"""I49zzr の図: 書き替えで中身を守る（終わりの、手前の並びの種類の数の下限を変えた時。run56 雑談、run57 青空文庫、run58 WEB 小説の集計の数だけ）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

KEEPS = [0, 5, 20, 50, 100]
DATA = {"雑談（MRMP）": dict(消えた=[.2057, .0838, .058, .0413, .0303], 打ち消し=[.1213, .1209, .0843, .0509, .0448],
                         変わった=[.9667, .8822, .8433, .8078, .7689], 合い方=[.84577, .73694, .61921, .50787, .38145]),
        "青空文庫の会話": dict(消えた=[.2543, .0893, .0489, .0279, .0166], 打ち消し=[.2253, .2174, .1833, .1784, .1502],
                        変わった=[.9789, .8894, .8639, .8378, .8028], 合い方=[1.30637, 1.08205, .9639, .79263, .64866]),
        "WEB 小説の会話": dict(消えた=[.2564, .0832, .0334, .0143, .008], 打ち消し=[.2213, .2136, .2049, .1631, .1377],
                          変わった=[.9839, .9078, .8811, .855, .8311], 合い方=[.93955, .76674, .67677, .576, .46402])}
COL = {"雑談（MRMP）": "#d9a441", "青空文庫の会話": "#3f73a8", "WEB 小説の会話": "#5a9a68"}
PANELS = (("消えた", "1. 中身の字（漢字・かたかな）が消えた行の割合", (0, .3)), ("打ち消し", "2. 打ち消しの印が入れ替わった行の割合", (0, .3)),
          ("変わった", "3. 書き替えで行が変わった割合", (.6, 1.02)), ("合い方", "4. 状況波との合い方（守らない時を 1 として）", (0, 1.08)))


def main():
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 4, figsize=(19, 5.2))
    xs = range(len(KEEPS))
    for a, (key, title, lim) in zip(ax, PANELS):
        for name, d in DATA.items():
            ys = d[key] if key != "合い方" else [v / d[key][0] for v in d[key]]
            a.plot(xs, ys, color=COL[name], lw=2.2, marker="o", label=name)
        a.set_xticks(list(xs)); a.set_xticklabels(["守らない" if k == 0 else str(k) for k in KEEPS])
        a.set_xlabel("替えてよい終わりの、手前の並びの種類の数の下限"); a.set_ylim(*lim); a.set_title(title, fontsize=10.5); a.grid(alpha=.3)
    ax[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("書き替えで中身を守る: 「どこにでも付く終わり」だけを替える（I49zzr。材料の 300 文 × 6 つの状況）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "i49zzr_keep.png"), dpi=170)


if __name__ == "__main__":
    main()
