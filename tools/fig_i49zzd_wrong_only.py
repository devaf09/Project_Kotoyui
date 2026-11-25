"""I49zzd: 外れた時だけ教える場面での反映（run38）の図。並べ替え 3 通りの平均と、それぞれの点。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def main():
    font = os.environ.get("KOTOYUI_FONT")
    if font:
        fm.fontManager.addfont(font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    rows = {
        "外付けのまま": [[.4037, .4087, .4258], [.3754, .4011, .4177], [.3897, .4070, .4254]],
        "本体へ直に混ぜる": [[.4317, .4431, .4487], [.4231, .4346, .4416], [.4233, .4347, .4436]],
        "話の終わりに本体へ反映": [[.4205, .4294, .4381], [.4062, .4262, .4353], [.4065, .4343, .4376]],
    }
    cols = {"外付けのまま": "#4477aa", "本体へ直に混ぜる": "#999933", "話の終わりに本体へ反映": "#cc3311"}
    always = [.4286, .4493, .4596]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = range(3)
    for name, v in rows.items():
        for r in v:
            ax.plot(x, r, "o", color=cols[name], alpha=.35, ms=4)
        ax.plot(x, [sum(r[i] for r in v) / 3 for i in x], "-o", color=cols[name], lw=2, label=name)
    ax.plot(x, always, "--", color="#cc3311", lw=1.2, label="（比べ）いつも教える時の反映")
    ax.axhline(.3990, color="gray", ls=":")
    ax.text(0.5, .4000, "本体だけ .399", color="gray", va="bottom", ha="center")
    ax.axhline(.4494, color="black", ls="--", lw=1)
    ax.text(-.05, .4504, "本体が後半も聞く .449", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["介入 10%\n（教えた問い 約 410）", "介入 30%\n（約 1,140）", "介入 100%\n（約 1,870）"])
    ax.set_ylabel("当たりの割合")
    ax.set_ylim(.37, .47)
    ax.set_title("外れた時だけ教える場面: 反映は効くが、直に混ぜる方が上\n（薄い点は並べ替え 1〜3 のそれぞれ）")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    for out in ("docs/images/i49zzd_wrong_only.png", "/mnt/project-files/kotoyui_intervention/figs/i49zzd_wrong_only.png"):
        fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main()
