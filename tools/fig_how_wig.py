"""しくみの図: 光の 2DWIG が、会話文から丁寧さを見分けるまで（docs/images/how_wig.png）

例の 2 文（この図のために書いた文。データの文ではない）を、語尾・言い回しの袋 → 64 × 64 の光の場 → 板と回折 → レンズの後の明るさ → 2 つの窓、の順に並べる。
板は、この図のために CPU で小さく学習し直したもの（現代の雑談の丁寧さ、訓練 2 万発言・板 8 枚・800 ステップ。実験01 の本番は板 16 枚・5000 ステップ）。
実行: CUDA_VISIBLE_DEVICES= python tools/fig_how_wig.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
import exp01_wig_bag as e1  # noqa: E402

EXAMPLES = ["明日は雨が降るそうですね。", "明日、雨降るんだってさ。"]


def main():
    trx, try_, tex, tey, classes = e1.data("polite", max_train=20000, max_test=2000)
    ci = {c: i for i, c in enumerate(classes)}
    X, y = e1.bag(trx), torch.tensor([ci[c] for c in try_])
    net = e1.p2.Net2D(e1.L, len(classes), ["xy"] * 8, patches=16, seed=0)
    e1.p2.train(net, X, y, steps=800, lr=0.01, batch_size=128, to_input=e1.to_field, schedule="cosine")
    Xt, yt = e1.bag(tex), torch.tensor([ci[c] for c in tey])
    acc = e1.p2.evaluate(net, Xt, yt, to_input=e1.to_field)
    B = e1.bag(EXAMPLES)
    with torch.no_grad():
        u = e1.to_field(B)
        I = net.screen(u)
        S = torch.softmax(net(u), 1)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.5), gridspec_kw=dict(width_ratios=[1.3, 1, 1, 0.8]))
    for r, text in enumerate(EXAMPLES):
        f = e1.feats(text)
        ax[r, 0].axis("off")
        ax[r, 0].text(0, 0.95, f"会話文:\n「{text}」", fontsize=12, va="top")
        ax[r, 0].text(0, 0.6, "袋の中身（語尾と、となり合う 2 文字）:\n" + "、".join(x.replace("E", "語尾:") for x in f[:9]) + " …",
                      fontsize=8.5, va="top", wrap=True)
        ax[r, 0].text(0, 0.18, f"→ {len(f)} 個の特徴を、64 × 64 = 4096 の\n   マス目のどれかに番号で割り当てて数える", fontsize=9, va="top")
        ax[r, 1].imshow(u[r].abs().numpy(), cmap="magma")
        ax[r, 1].set_title("① 光の場に入れる\n（明るい点 = 特徴のあるマス）", fontsize=9)
        ax[r, 2].imshow(I[r].numpy() ** 0.5, cmap="viridis")
        det = net.det.sum(0).numpy() > 0
        ax[r, 2].contour(det, levels=[0.5], colors="w", linewidths=0.8)
        ax[r, 2].set_title("② 板 8 枚と回折を通り、レンズの後の明るさ\n（白い枠 = 答えを読む窓）", fontsize=9)
        for a in ax[r, 1:3]:
            a.set_xticks([])
            a.set_yticks([])
        ax[r, 3].barh(classes, S[r].numpy(), color=["#137", "#e93"])
        ax[r, 3].set_xlim(0, 1)
        ax[r, 3].set_title("③ 窓の明るさ → 答え", fontsize=9)
        ax[r, 3].set_xlabel("確率")
    fig.suptitle(f"光の 2DWIG のしくみ（この図のための小さな学習。テストの正解率 {acc:.1f}%）", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "images", "how_wig.png")
    fig.savefig(out, dpi=130)
    print(out, acc)


if __name__ == "__main__":
    main()
