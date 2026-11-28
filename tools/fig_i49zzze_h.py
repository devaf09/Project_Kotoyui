"""I49zzze〜h の図: 道具が最後に出す 1 行（数だけの JSON）を保存した物から描く。
使い方: python tools/fig_i49zzze_h.py --kind 切れ目|テーマ|疑問|薄める --in docs/numbers/i49zzze.json [--name i49zzze_boundary]
  --in の JSON は、数と、信号や引き方の名前だけ（材料の語は入れない。「自分で続ける」の文が入った --out の方は、使わない）。
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

KINDS = {"切れ目": "i49zzze_boundary", "テーマ": "i49zzzf_theme", "疑問": "i49zzzg_doubt", "薄める": "i49zzzh_shelter"}
BLUE, GREY, ORANGE, GREEN = "#3f73a8", "#bdbdbd", "#e08a2c", "#6aa56a"


def boundary(d):
    modes = [m for m in ("話の中だけ", "流れの全部") if m in d]
    sig = d[modes[-1]]["信号"]
    names = sorted(sig, key=lambda n: -(sig[n]["見分け"] or 0))[:14][::-1]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    for a, key, title in ((ax[0], "見分け", "見分け（切れ目の行の信号が、他の行より大きい割合。.5 が当てずっぽう）"), (ax[1], "F1", "しきい値を前半の数えから選んだ時の F1")):
        for j, (m, col) in enumerate(zip(modes, (GREY, BLUE))):
            vals = [d[m]["信号"][n].get(key) or 0 for n in names]
            ys = [i + (j - .5) * .38 for i in range(len(names))]
            a.barh(ys, vals, .38, color=col, label="遷移の波の聞かせ方: " + m)
            for y, v in zip(ys, vals):
                a.text(v + .01, y, "%.2f" % v, va="center", fontsize=7.5)
        a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=8.5); a.set_xlim(0, 1.12); a.set_title(title, fontsize=10); a.grid(alpha=.3, axis="x")
    ax[0].axvline(.5, color="#555555", ls="--", lw=1)
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.06), ncol=2)
    n = d[modes[-1]]
    fig.suptitle("話の切れ目で、器の中の信号は立つか（I49zzze。%d 行のうち切れ目 %d 行。上位 14 個）" % (n["行"], n["切れ目の行"]), fontsize=12)
    return fig


def theme(d):
    c, sc = d["どうした"], d["場面"]
    names = [n for n in c if n != "問い"]
    short = [n.replace("（前の行から）", "").replace("（", "\n（") for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6))
    vals = [c[n] / max(c["問い"], 1) for n in names]
    ax[0].bar(range(len(names)), vals, color=BLUE)
    for i, v in enumerate(vals):
        ax[0].text(i, v + .002, "%.3f" % v, ha="center", fontsize=8.5)
    ax[0].set_title("1. 次の行の動きの当たり（広い候補、%d 問）" % c["問い"], fontsize=10.5); ax[0].set_ylabel("当たりの割合")
    for j, (k, col, lab) in enumerate(((1, BLUE, "出した動きが、その話の本物の続きに有る"), (3, GREY, "同じ動きが、別の話の続きに有る（たまたまの重なり）"),
                                      (2, GREEN, "出した物が、その話の本物の続きに有る"), (4, "#dddddd", "同じ物が、別の話の続きに有る"))):
        vs = [sc[n][k] / max(sc[n][0], 1) if n in sc else 0 for n in names]
        xs = [i + (j - 1.5) * .2 for i in range(len(names))]
        ax[1].bar(xs, vs, .2, color=col, label=lab)
        for x, v in zip(xs, vs):
            ax[1].text(x, v + .008, "%.2f" % v, ha="center", fontsize=6.5)
    n = max((sc[x][0] for x in sc), default=0)
    ax[1].set_title("2. 自分で続けた時の場面の物差し（引き方ごとに %d 個）" % n, fontsize=10.5); ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)
    for a in ax:
        a.set_xticks(range(len(names))); a.set_xticklabels(short, fontsize=7.5); a.grid(alpha=.3, axis="y")
    fig.suptitle("テーマ波と話の束ねで動きを引く（I49zzzf。話の切れ目: %s）" % d.get("切れ目", ""), fontsize=12)
    return fig


def doubt(d):
    lack = d["足りない"]
    keys = [k for k in lack if k.startswith("だれが")]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5), gridspec_kw=dict(width_ratios=(3, 3, 2.4)))
    for a, split in zip(ax[:2], ("割れ方（エントロピー）", "上位 2 つの近さ")):
        for k, col in zip(keys, (BLUE, ORANGE, GREEN, GREY)):
            bins = lack[k][split]["等分"]
            a.plot(range(1, len(bins) + 1), [b["当たり"] / max(b["数"], 1) for b in bins], marker="o", color=col,
                   label="%s（見分け %.2f）" % (k, lack[k][split]["見分け"] or 0))
        a.set_xticks(range(1, len(bins) + 1)); a.set_xlabel("割れ方の小さい順に等分した組（1 が一番割れていない）"); a.set_ylim(0, 1.05)
        a.set_title("隠した「だれが」を前提で埋めた時の当たり: " + split, fontsize=10); a.grid(alpha=.3); a.legend(fontsize=8.5)
    ax[0].set_ylabel("当たりの割合")
    mis = d["合わない"]
    h = sorted(mis)[0]
    seats = [s for s in mis[h]["話の初めの行"] if s != "行"]
    for j, (k, col) in enumerate((("話の初めの行", BLUE), ("それ以外の行", GREY))):
        vs = [mis[h][k][s] for s in seats]
        xs = [i + (j - .5) * .38 for i in range(len(seats))]
        ax[2].bar(xs, vs, .38, color=col, label=k)
        for x, v in zip(xs, vs):
            ax[2].text(x, v + .01, "%.2f" % v, ha="center", fontsize=8.5)
    ax[2].set_xticks(range(len(seats))); ax[2].set_xticklabels(seats); ax[2].set_ylim(0, 1.12); ax[2].grid(alpha=.3, axis="y"); ax[2].legend(fontsize=8.5, loc="lower right")
    ax[2].set_title("言われた語の、前提との合わなさ（%s）" % h, fontsize=10)
    fig.suptitle("前提の割れ方は、疑問の物差しになるか（I49zzzg）", fontsize=12)
    return fig


def shelter(d):
    c, lv = d["だれが"], d["段ごとの行の数"]
    modes = list(c)
    hs = [k for k in c[modes[0]] if k != "問い"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    for j, (h, col) in enumerate(zip(hs, (BLUE, ORANGE, GREEN))):
        vs = [c[m][h] / max(c[m]["問い"], 1) for m in modes]
        xs = [i + (j - (len(hs) - 1) / 2) * .38 for i in range(len(modes))]
        ax[0].bar(xs, vs, .38, color=col, label="持った波との共鳴（語そのもの）" + h)
        for x, v in zip(xs, vs):
            ax[0].text(x, v + .006, "%.3f" % v, ha="center", fontsize=8.5)
    ax[0].set_xticks(range(len(modes))); ax[0].set_xticklabels(modes, fontsize=9); ax[0].set_ylabel("当たりの割合"); ax[0].grid(alpha=.3, axis="y")
    ax[0].set_title("1. 「だれが」の当たり（%d 問）" % c[modes[0]]["問い"], fontsize=10.5); ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.12))
    gains = sorted({g for v in lv.values() for g in v}, reverse=True)
    for j, (k, col) in enumerate((("話の初めの行", BLUE), ("それ以外の行", GREY))):
        tot = max(sum(lv[k].values()), 1)
        vs = [lv[k].get(g, 0) / tot for g in gains]
        xs = [i + (j - .5) * .38 for i in range(len(gains))]
        ax[1].bar(xs, vs, .38, color=col, label="%s（%d 行）" % (k, sum(lv[k].values())))
        for x, v in zip(xs, vs):
            ax[1].text(x, v + .008, "%.2f" % v, ha="center", fontsize=8.5)
    ax[1].set_xticks(range(len(gains))); ax[1].set_xticklabels(["入り方 %s%s" % (g, "\n（切る）" if float(g) == 0 else "\n（そのまま）" if float(g) == 1 else "") for g in gains], fontsize=9)
    w = d["続けて待たせた行"]
    ax[1].set_title("2. 段ごとの行の割合（続けて待たせた行: %d 回、平均 %.1f、最大 %d）" % (w["回"], w["平均"], w["最大"]), fontsize=10.5); ax[1].grid(alpha=.3, axis="y"); ax[1].legend(fontsize=8.5)
    fig.suptitle("外の入力を薄める・切って待たせると、器の中で何が変わるか（I49zzzh）", fontsize=12)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=list(KINDS))
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--name")
    a = ap.parse_args()
    f = os.environ.get("KOTOYUI_FONT")
    if f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    d = json.load(open(a.src, encoding="utf-8"))
    fig = {"切れ目": boundary, "テーマ": theme, "疑問": doubt, "薄める": shelter}[a.kind](d)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", (a.name or KINDS[a.kind]) + ".png"), dpi=170)


if __name__ == "__main__":
    main()
