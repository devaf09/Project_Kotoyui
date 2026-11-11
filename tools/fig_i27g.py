"""図: 実験I27g（行ごとの板の展開・人物の統一感・前後の行からの補い）。結果の jsonl の最後の行から作る。
図は非公開の置き場に置く（Web 小説の結果は _wn を付ける）。図には作品の文を入れない（人物名の軸ラベルは作品名だけにする）。
実行: python tools/fig_i27g.py --res <i27g.jsonl> --out <図の置き場> [--tag _wn]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "WenQuanYi Zen Hei", "sans-serif"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_expand(r, path):
    e = r["expand"]
    names = ["可逆（ネスト）\n元と同じ", "可逆 σ0.2\n元と同じ", "関連性\n元と同じ", "関連性\n値の帯 F1", "元の順\n値の帯 F1（上限）"]
    vals = [e["reversible_exact"], e["reversible_noisy_exact"], e["assoc_same"], e["assoc_f1"], e["nest_f1_ceiling"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURF)
    style(ax, f"中ほどの行の展開（{e['n']} 行）")
    b = ax.bar(range(len(vals)), vals, color=S1, width=0.55, edgecolor=SURF, linewidth=2)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(vals)), names, fontsize=8.5)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_unity(r, path):
    us = [u for u in r["unity"] if u["n_mentions"] > 0]
    labels = [u["work"][:8] if not u["work"][:1].isdigit() else f"作品 {chr(65 + i)}" for i, u in enumerate(us)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=SURF, gridspec_kw=dict(width_ratios=[3, 2]))
    ax = axes[0]
    style(ax, "名前を隠して、束ねた板から人物を当てる")
    x = range(len(us))
    w = 0.26
    for k, (key, lab, col) in enumerate((("acc", "束ねた板で当てる", S1), ("most_frequent", "いちばん多い人物", S2), ("chance", "偶然", S3))):
        ax.bar([i + (k - 1) * w for i in x], [u[key] for u in us], width=w, color=col, label=lab, edgecolor=SURF, linewidth=1.5)
    ax.set_xticks(list(x), labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax = axes[1]
    style(ax, "行の板どうしの近さ（余弦）")
    ax.bar([i - 0.2 for i in x], [u["within"] for u in us], width=0.4, color=S1, label="同じ人物", edgecolor=SURF, linewidth=1.5)
    ax.bar([i + 0.2 for i in x], [u["between"] for u in us], width=0.4, color=S2, label="違う人物", edgecolor=SURF, linewidth=1.5)
    ax.set_xticks(list(x), labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_completion(r, path):
    c = r["completion"]
    fr = sorted(c["plateau"], key=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), facecolor=SURF)
    ax = axes[0]
    style(ax, f"属性による展開（問い {c['n_targets']}）")
    ax.plot([float(f) for f in fr], [c["plateau"][f]["acc"] for f in fr], color=S1, lw=2, marker="o", ms=6, label="正解率")
    ax.plot([float(f) for f in fr], [c["plateau"][f]["coverage"] for f in fr], color=S2, lw=2, marker="o", ms=6, label="答えが候補にある")
    ax.set_xlabel("束ねに使ったほかの行の割合", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0, max(0.05, max(c["plateau"][f]["coverage"] for f in fr)) * 1.25)
    wins = sorted({int(k.split("|")[1]) for k in c["completion"]})
    sw = c.get("sweep", {})

    def get(mode, w, key):
        k = f"{mode}|{w}|足す β0.3"
        return (sw[k] if k in sw else c["completion"][f"{mode}|{w}"])[key]
    for ax, key, title in ((axes[1], "acc", "前後の行から補う: 正解率（1 位）"), (axes[2], "coverage", "前後の行から補う: 答えが候補にある")):
        style(ax, title)
        top = 0
        for mode, col in (("属性だけ", S3), ("展開した前後", S1), ("縮めた前後", S2)):
            ys = [c["completion"][f"{mode}|{w}"][key] if mode == "属性だけ" else get(mode, w, key) for w in wins]
            top = max(top, max(ys))
            ax.plot(wins, ys, color=col, lw=2, marker="o", ms=6, label=mode)
            if key == "coverage":                                          # 正解率の方は値が重なるので、凡例と表に任せる
                ax.text(wins[-1] + 0.08, ys[-1], f"{ys[-1]:.3f}", fontsize=8, color=INK, va="center")
        ax.set_xticks(wins, [f"前後 {w} 行" for w in wins])
        ax.set_xlim(wins[0] - 0.3, wins[-1] + 0.6)
        ax.set_ylim(0, max(0.05, top) * 1.3)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.text(0.99, 0.01, "前後の語の重み β = 0.3（結果に掃引あり）", ha="right", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    r = [json.loads(x) for x in open(a.res, encoding="utf-8") if x.strip()][-1]
    os.makedirs(a.out, exist_ok=True)
    fig_expand(r, os.path.join(a.out, f"i27g_expand{a.tag}.png"))
    fig_unity(r, os.path.join(a.out, f"i27g_unity{a.tag}.png"))
    fig_completion(r, os.path.join(a.out, f"i27g_completion{a.tag}.png"))


if __name__ == "__main__":
    main()
