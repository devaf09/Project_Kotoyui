import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf")
plt.rcParams["font.family"] = "IPAPGothic"
D = json.load(open(sys.argv[1], encoding="utf-8"))
runs = [k for k in D if "先生" in k]                     # 先生ありの回だけ、版ごとに色（決まった順）
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
def ma(v, k=5):
    return [sum(v[max(0, i - k + 1):i + 1]) / len(v[max(0, i - k + 1):i + 1]) for i in range(len(v))]
panels = [("中身なし", "中身の無い返事（12 往復あたり）"), ("繰り返し", "前と一字一句同じ行を言った回数"),
          ("受け", "相手が受けて話を続けた回数"), ("先生", "先生に変と言われた回数")]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.1), dpi=200, sharex=True)
fig.patch.set_facecolor("#fcfcfb")
for ax, (k, title) in zip(axs, panels):
    ax.set_facecolor("#fcfcfb")
    for ci, name in enumerate(runs):
        y = [r[k] for r in D[name]]; x = list(range(1, len(y) + 1))
        ax.plot(x, y, "o", ms=3, color=COL[ci], alpha=0.3, mec="none")
        ax.plot(x, ma(y), "-", lw=2, color=COL[ci], label=name.replace("Gemma 相手・先生", "").strip("（）") + f"（計 {sum(y)}）")
    ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    ax.set_xticks([1, 5, 10, 15, 20]); ax.set_ylim(0, 12.5); ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#b8b7b2")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.set_xlabel("会話の順（線は直近 5 会話の平均）", fontsize=8, color="#52514e")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right" if k != "中身なし" else "lower right")
fig.suptitle("Gemma が話し相手と先生の回（20 会話 × 12 往復、ボットは会話をまたいで同じ）", fontsize=10.5, x=0.01, ha="left", color="#0b0b0b")
fig.tight_layout(); fig.savefig(sys.argv[2], facecolor=fig.get_facecolor())
