"""見積もり E6 の図: 現代の雑談で、話者の見分け・隠し味の効き・口調の当たり（docs/images/est_modern.png）

results/est_modern.txt の最後の実行から作る。
実行: python tools/fig_est_modern.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
NAMES = {"polite": "丁寧さ\n（発言ごと）", "gender": "性別", "age3": "年代 3 つ", "fp4": "一人称 4 類"}


def main():
    rs = [json.loads(l) for l in open(os.path.join(ROOT, "results", "est_modern.txt"), encoding="utf-8") if l.strip()]
    last = {}
    for r in rs:
        last[(r["part"], r["label"], r.get("K"), r.get("cue"), r.get("cond"))] = r
    labels = [l for l in NAMES if ("A", l, 1, None, None) in last]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    a = ax[0]
    acc = [last[("A", l, 1, None, None)]["acc"] for l in labels]
    mac = [last[("A", l, 1, None, None)]["macro"] for l in labels]
    maj = [last[("A", l, 1, None, None)]["majority"] for l in labels]
    a.bar(x - 0.2, acc, 0.4, label="正解率", color="#37a")
    a.bar(x + 0.2, mac, 0.4, label="類ごとの正解率の平均", color="#e93")
    a.scatter(x - 0.2, maj, marker="_", s=600, c="k", label="いつも一番多い類", zorder=3)
    a.set_xticks(x, [NAMES[l] for l in labels])
    a.set_ylabel("%（知らない話者 51 人の発言 1 つ）")
    a.set_title("(a) 話者の見分け（数えた表）", fontsize=10)
    a.legend(fontsize=8)
    a.set_ylim(0, 105)
    a = ax[1]
    for i, (cue, nm, c) in enumerate((("oracle", "本当の類", "#137"), ("prev", "同じ話者の前の発言", "#9bd"))):
        v = [last[("B", l, None, cue, None)]["gain_acc"] for l in labels]
        a.bar(x + (i - 0.5) * 0.4, v, 0.4, label=nm, color=c)
        for xi, yi in zip(x, v):
            a.text(xi + (i - 0.5) * 0.4, yi + (0.02 if yi >= 0 else -0.06), f"{yi:+.2f}", ha="center", fontsize=8)
    a.axhline(0.5, ls="--", c="r", lw=1)
    a.text(len(labels) - 0.5, 0.51, "判断の基準 +0.5 点", ha="right", color="r", fontsize=8)
    a.axhline(0, c="k", lw=0.6)
    a.set_xticks(x, [NAMES[l] for l in labels])
    a.set_ylabel("隠し味あり − なし（次のサブワードの正解率、点）")
    a.set_title("(b) 隠し味の効き（3-gram、テスト 5,000 発言、seed 0）", fontsize=10)
    a.set_ylim(-0.4, 0.7)
    a.legend(fontsize=8)
    a = ax[2]
    for i, (cond, nm, c) in enumerate((("none", "隠し味なし", "#999"), ("mix1.0", "類の表だけで作る", "#137"))):
        r = [last[("C", l, None, None, cond)] for l in labels]
        a.bar(x + (i - 0.5) * 0.4, [q["style_acc"] for q in r], 0.4, yerr=[q["style_acc_se"] for q in r], capsize=3, label=nm, color=c)
    a.scatter(x, [last[("C", l, None, None, "none")]["judge_real_acc"] for l in labels], marker="*", s=120, c="g", zorder=3, label="本物の発言での当て直し（上限の目安）")
    a.scatter(x, [last[("C", l, None, None, "none")]["chance"] for l in labels], marker="_", s=600, c="k", zorder=3, label="偶然")
    a.set_xticks(x, [NAMES[l] for l in labels])
    a.set_ylabel("決めた類で当て直された割合（%、3 seed）")
    a.set_title("(c) 口調の当たり（当て直しは (a) の表）", fontsize=10)
    a.set_ylim(0, 105)
    a.legend(fontsize=7, loc="upper right")
    fig.suptitle("見積もり E6: 現代の雑談（RealPersonaChat・名大会話コーパス、知らない話者で測る、GPU なし）", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "images", "est_modern.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
