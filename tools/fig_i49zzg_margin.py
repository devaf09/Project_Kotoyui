"""I49zzg: 1 位と 2 位の点の差ごとの当たり（run36a）と、器の向きで選び直した時の変わり方（run36b）の図。並べ替え 3 通りの合計。"""
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
    names = ["0\n（全く同じ）", "〜0.01", "〜0.05", "〜0.1", "〜0.2", "〜0.5", "それより上"]
    n = [340, 614, 1958, 1844, 2057, 2535, 10581]                 # 〜0.000001 と 〜0.0001 の 18 問は 〜0.01 に足した
    top1 = [38, 103, 373, 446, 563, 832, 6623]
    top2 = [39, 99, 377, 344, 477, 609, 593]
    wid = ["0", "〜0.01", "〜0.05", "〜0.1", "〜0.2", "〜0.5"]
    near = [(15, 19), (55, 62), (212, 255), (357, 475), (580, 768), (778, 1226)]      # (当たりに変わった, 外れに変わった)
    far = [(23, 14), (63, 51), (253, 188), (422, 405), (679, 714), (944, 1291)]
    changed = [160, 432, 1388, 2395, 3724, 5577]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5))
    x = range(len(names))
    a.bar([i - .2 for i in x], [t / m for t, m in zip(top1, n)], .4, color="#4477aa", label="1 位の候補が当たり")
    a.bar([i + .2 for i in x], [t / m for t, m in zip(top2, n)], .4, color="#ee7733", label="2 位の候補が当たり")
    for i, m in enumerate(n):
        a.text(i, .01, "%s 問" % format(m, ","), ha="center", fontsize=8, color="white")
    a.axvspan(-.5, 2.5, color="gray", alpha=.12)
    a.text(1, .5, "1 位と 2 位が五分\n（問いの 15%）", ha="center", fontsize=10)
    a.set_xticks(list(x))
    a.set_xticklabels(names, fontsize=9)
    a.set_xlabel("1 位と 2 位の点の差の割合")
    a.set_ylabel("当たりの割合")
    a.set_title("点の差が小さい問いでは、1 位と 2 位は同じくらい当たる")
    a.legend(loc="upper left", fontsize=9)
    a.grid(axis="y", alpha=.3)
    x = range(len(wid))
    b.bar([i - .2 for i in x], [u - d for u, d in near], .4, color="#cc3311", label="器の向きに一番近い候補を選ぶ")
    b.bar([i + .2 for i in x], [u - d for u, d in far], .4, color="#009988", label="器の向きに一番遠い候補を選ぶ（比べ役）")
    for i, c in enumerate(changed):
        b.text(i, 95, "答えが変わった\n%s 問" % format(c, ","), ha="center", fontsize=8)
    b.axhline(0, color="black", lw=1)
    b.set_ylim(-480, 140)
    b.set_xticks(list(x))
    b.set_xticklabels(wid)
    b.set_xlabel("同点と見なす幅（1 位との差の割合がこれ以下の候補から選び直す）")
    b.set_ylabel("当たりに変わった − 外れに変わった（問い）")
    b.set_title("器の向きで選び直すと答えは変わる。ただし近い方を選ぶと損で、遠い方が少し得")
    b.legend(loc="lower left", fontsize=9)
    b.grid(axis="y", alpha=.3)
    fig.tight_layout()
    for out in ("docs/images/i49zzg_margin.png", "/mnt/project-files/kotoyui_intervention/figs/i49zzg_margin.png"):
        fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main()
