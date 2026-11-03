"""M1 の見積もりの図: ラベルの広げ方・2DWIG の袋・隠し味の効き・口調の当たり（docs/images/est_m1.png）

results/est_label_spread.txt・est_wig_bag.txt・est_flavor_counts.txt・est_flavor_style.txt の最後の実行から作る。
実行: python tools/fig_est_m1.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def rows(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, "results", name), encoding="utf-8") if l.strip()]


def last(rs, **kw):
    m = [r for r in rs if all(r.get(k) == v for k, v in kw.items())]
    return m[-1] if m else None


def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # (a) ラベルの広げ方
    ls = rows("est_label_spread.txt")[-1]
    names = ["すぐ次（k=1）", "2 つ先（k=2）", "3 つ先（k=3）", "4 つ先（k=4）", "同じ作品の\nでたらめな組"]
    vals = [ls["agree_k1"], ls["agree_k2"], ls["agree_k3"], ls["agree_k4"], ls["chance"]]
    cols = ["#999", "#2a7", "#999", "#2a7", "#ccc"]
    a = ax[0, 0]
    a.bar(names, vals, color=cols)
    a.axhline(80, ls="--", c="r", lw=1)
    a.text(4.4, 81, "広げる基準 80%", ha="right", color="r", fontsize=9)
    for i, v in enumerate(vals):
        a.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    a.set_ylabel("両方に一人称がある組で、類が同じ割合（%）")
    a.set_title("(a) 交互の会話でラベルを広げる（地の文をはさまない会話の列）")
    a.set_ylim(0, 105)
    # (b) 2DWIG の袋
    wb = rows("est_wig_bag.txt")
    conds = [("nb", "end123+bi", 0, "real", "数えた表\n語尾+2つ組（SN2 と同じ）"), ("nb", "end123+bi", 0, "spread", "同\n+広げたラベル"),
             ("nb", "end123+bi+tri", 0, "spread", "同 + 3つ組\n+広げたラベル"), ("nb", "end123+bi", 4096, "real", "数えた表\n散らして 4096"),
             ("nb", "end123+bi", 1024, "real", "数えた表\n散らして 1024"), ("lr", "end123+bi", 0, "real", "回帰\n語尾+2つ組"),
             ("lr", "end123+bi", 4096, "real", "回帰\n散らして 4096")]
    a = ax[0, 1]
    xs, acc, mac = [], [], []
    for m, f, H, t, lab in conds:
        r = last(wb, method=m, feats=f, H=H, train=t)
        if r:
            xs.append(lab)
            acc.append(r["test_acc"])
            mac.append(r["macro_recall"])
    import numpy as np
    x = np.arange(len(xs))
    a.bar(x - 0.2, acc, 0.4, label="正解率", color="#37a")
    a.bar(x + 0.2, mac, 0.4, label="類ごとの正解率の平均", color="#e93")
    a.axhline(38.6, ls="--", c="k", lw=1)
    a.axhline(43.1, ls=":", c="g", lw=1)
    a.text(-0.5, 36.3, "いつも「私」38.6%", ha="left", fontsize=8)
    a.text(-0.5, 64.5, "緑の点線: SN2 の光（位置並べ）43.1%", ha="left", fontsize=8, color="g")
    a.set_xticks(x, xs, fontsize=7, rotation=25)
    a.set_ylabel("%（12 類、テスト 4,101 件）")
    a.set_title("(b) 2DWIG の袋の中身と大きさ（回帰 = ロジスティック回帰）")
    a.legend(fontsize=8, loc="upper right")
    a.set_ylim(0, 70)
    # (c) 隠し味の効き
    fc = rows("est_flavor_counts.txt")
    a = ax[1, 0]
    labs, ga, gc = [], [], []
    for g in ("g5", "g12"):
        for cnt in ("hard", "soft"):
            for cue, nm in (("oracle", "本当の類"), ("spread", "写した類"), ("prev", "2つ前の発言"), ("prefix", "書きかけ")):
                r = last(fc, groups=g, count=cnt, cue=cue)
                if r:
                    labs.append(f"{g} {cnt}\n{nm}")
                    ga.append(r["gain_acc"])
                    gc.append(-r["gain_ce"])
    x = np.arange(len(labs))
    a.bar(x - 0.2, ga, 0.4, label="正解率の差（点）", color="#37a")
    a2 = a.twinx()
    a2.bar(x + 0.2, gc, 0.4, label="CE の改善（nat、右の軸）", color="#e93")
    a.axhline(0.5, ls="--", c="r", lw=1)
    a.text(len(labs) - 0.5, 0.52, "判断の基準 +0.5 点", ha="right", color="r", fontsize=8)
    a.axhline(0, c="k", lw=0.6)
    a.set_xticks(x, labs, fontsize=6.5, rotation=90)
    a.set_ylabel("隠し味あり − なし（正解率、点）")
    a2.set_ylabel("CE の改善（nat / 単位）")
    a.set_ylim(-0.6, 1.0)
    a2.set_ylim(-0.12, 0.2)
    a.set_title("(c) 隠し味の効き（次のサブワード、3-gram、テスト 5,000 文）")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    # (d) 口調の当たり
    st = rows("est_flavor_style.txt")
    a = ax[1, 1]
    cs = [("none", "隠し味なし"), ("mix0.5", "λ=0.5"), ("mix0.8", "λ=0.8"), ("mix1.0", "λ=1.0")]
    rs = [last(st, cond=c) for c, _ in cs]
    a.bar([n for _, n in cs], [r["style_acc"] for r in rs], yerr=[r["style_acc_se"] or 0 for r in rs], color=["#999", "#9bd", "#37a", "#137"], capsize=4)
    for i, r in enumerate(rs):
        a.text(i, r["style_acc"] + 2, f"{r['style_acc']:.1f}", ha="center", fontsize=9)
    a.axhline(20, ls="--", c="k", lw=1)
    a.text(3.4, 21, "偶然 20%", ha="right", fontsize=8)
    jr = rs[-1]["judge_real_acc"]
    a.axhline(jr, ls=":", c="g", lw=1)
    a.text(3.4, jr + 1, f"本物の会話文での当て直し {jr:.1f}%", ha="right", fontsize=8, color="g")
    a.set_ylabel("決めた組で当て直された割合（%）")
    a.set_title("(d) 口調の当たり（5 組 × 100 文、3 seed の平均 ± 標準誤差）")
    a.set_ylim(0, 75)
    fig.suptitle("M1 の見積もり（数えた表、青空文庫 1,000 作品の会話文、GPU なし）", fontsize=13)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "images", "est_m1.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
