"""実験17・18 の図（非公開の結果から。数字だけなので図は公開の docs/images/ にも写す）: 大きさと正解率、語尾の個性"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r17 = [json.loads(l) for l in open(os.path.join(PRIV, "results", "exp17.txt"), encoding="utf-8")][-1]["runs"]
r18 = {}
for l in open(os.path.join(PRIV, "results", "exp18.txt"), encoding="utf-8"):
    r18.update(json.loads(l)["runs"])
BLUE, AQUA, GRAY, TAN, INK, INK2, SURF = "#2a78d6", "#1baf7a", "#8a8984", "#c8a27a", "#0b0b0b", "#52514e", "#fcfcfb"
rows = [("光路 6×8\n9000 ステップ（実験17）", r17["光の 2DTG（4f 系）"], BLUE), ("光路 6×8\n18000 ステップ", r18["t6x8_18k"], BLUE),
        ("光路 8×16\n18000 ステップ", r18["t8x16_18k"], BLUE), ("分担 8 個（各 4×4）", r18["s8_4x4"], AQUA), ("分担 16 個（各 4×4）", r18["s16_4x4"], AQUA),
        ("GRU 256（実験17）", r17["GRU"], GRAY), ("GRU 512", r18["gru512"], GRAY), ("GRU 1024", r18["gru1024"], GRAY)]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.8), facecolor=SURF)
for a_ in (ax, bx):
    a_.set_facecolor(SURF)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
y = list(range(len(rows)))[::-1]
for yi, (name, v, c) in zip(y, rows):
    ax.barh(yi, v["acc"], color=c)
    size = (f"光 {v['params_light'] / 1e4:.0f} 万・デジタル {v['params_digital'] / 1e4:.0f} 万" if "params_light" in v else f"デジタル {v['params_digital'] / 1e4:.0f} 万")
    ax.text(v["acc"] + 0.3, yi, f"{v['acc']:.1f}%　{size}", va="center", fontsize=9, color=INK)
t3 = r17["3-gram（口調ごと）"]["acc"]
ax.axvline(t3, color=TAN, ls="--", lw=1.5)
ax.text(t3 + 0.2, y[-1] - 0.6, f"3-gram {t3:.1f}%", color=INK2, fontsize=9)
ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
ax.set_xlim(0, 52)
ax.set_xlabel("次の単位の正解率（%、試し 3 万位置）", color=INK2)
ax.set_title("大きくすると、次の単位の当たりは伸びるか", fontsize=12, color=INK)
cats = [("光路 6×8\n口調の帯", 24.8, BLUE), ("GRU 256\n口調の帯", 45.7, GRAY), ("GRU 1024\n口調の帯", 100 * r18["gru1024"]["style_hit"], GRAY),
        ("分担 16 個\n口調の帯", 100 * r18["s16_4x4"]["style_hit"], AQUA), ("3-gram\n口調ごとの表", 72.4, TAN),
        ("光路 6×8\n人格ごとの器", 100 * r18["t6x8_18k"]["style_hit"], "#1c5cab")]
for i, (n, v, c) in enumerate(cats):
    bx.bar(i, v, color=c, width=0.65)
    bx.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
bx.set_xticks(range(len(cats)), [c[0] for c in cats], fontsize=8.5)
bx.set_ylim(0, 105)
bx.set_ylabel("決めた口調の語尾になった割合（%、7 口調 × 15 文）", color=INK2)
bx.set_title("語尾の個性（お嬢様・老人・後輩・侍・俺様ほか）", fontsize=12, color=INK)
fig.tight_layout()
out = os.path.join(PRIV, "docs", "exp18_scale.png")
fig.savefig(out, dpi=130, facecolor=SURF)
shutil.copy(out, os.path.join(KOTO_ROOT, "docs", "images", "exp18_scale.png"))
print(out)
