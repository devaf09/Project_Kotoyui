"""I49zzd: 話の終わりに本体へ反映する形の数え（run26）の図。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

font = os.environ.get("KOTOYUI_FONT")
if font:
    fm.fontManager.addfont(font)
    plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
plt.rcParams["axes.unicode_minus"] = False

rates = ["10%", "30%", "100%"]
base = [.4057, .3912, .4002]
full = [.4542, .4438, .4502]
rows = {
    "外付けのまま": [[.4106, .4356, .4435], [.3991, .4298, .4383], [.4083, .4282, .4431]],
    "本体へ直に混ぜる": [[.4360, .4487, .4542], [.4241, .4395, .4438], [.4245, .4452, .4502]],
    "話の終わりに本体へ反映": [[.4342, .4514, .4623], [.4243, .4473, .4579], [.4273, .4493, .4587]],
}
cols = {"外付けのまま": "#4477aa", "本体へ直に混ぜる": "#999933", "話の終わりに本体へ反映": "#cc3311"}
fig, ax = plt.subplots(figsize=(8, 5))
x = range(3)
for name, v in rows.items():
    avg = [sum(r[i] for r in v) / 3 for i in x]
    for r in v:
        ax.plot(x, r, "o", color=cols[name], alpha=.35, ms=4)
    ax.plot(x, avg, "-o", color=cols[name], lw=2, label=name + "（3 通りの平均）")
b = sum(base) / 3
f = sum(full) / 3
ax.axhline(b, color="gray", ls=":")
ax.text(.25, b + .001, "本体だけ %.3f" % b, color="gray", va="bottom")
ax.axhline(f, color="black", ls="--", lw=1)
ax.text(-.05, f + .001, "本体が後半も聞く %.3f" % f, va="bottom")
ax.set_xticks(list(x))
ax.set_xticklabels(["介入 " + r for r in rates])
ax.set_ylabel("当たりの割合")
ax.set_ylim(.39, .47)
ax.set_title("教わった事を話の間は外付けに持ち、話の終わりに本体へ反映する\n（薄い点は並べ替え 1〜3 のそれぞれ）")
ax.legend(loc="lower right")
ax.grid(alpha=.3)
fig.tight_layout()
for out in ("docs/images/i49zzd_reflect.png",
            "/mnt/project-files/kotoyui_intervention/figs/i49zzd_reflect.png"):
    fig.savefig(out, dpi=170)
