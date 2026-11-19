"""文を波に変えて戻す小さな実験（自作の 10 文）。置き方はリポジトリの i48ab_place.Placer の「拡散」。
語 = D 全体に広がる決まった乱数の位相の符号 1 本。位置 k = k×SHIFT の巡回のずれ。文の波 = 全部の語の和。
戻し方 = 位置のずれを戻して、候補の語の符号との近さがいちばん大きい物を選ぶ（＋読めた語を引いて読み直す 2 回）。
使い方: python tools/i48_roundtrip.py [図の出力先.png]（数字は同じ名前 + .json）。字体は環境変数 KOTOYUI_FONT。"""
import sys, json
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48ab_place as PL

SENTS = ["今日 は 天気 が いい です", "友達 と 映画 を 見 ました", "猫 が 窓 の そば で 寝て います",
         "駅 まで 歩いて 十分 かかり ます", "明日 は 早く 起き ない と いけ ない", "この 本 は とても 面白 かった",
         "雨 が 降っ たら 傘 を 持って 行く", "先生 に 質問 を して も いい です か", "昨日 の 夜 は よく 眠れ なかった",
         "コーヒー を 飲み ながら 音楽 を 聞く"]
SENTS = [s.split() for s in SENTS]
VOC = sorted({w for s in SENTS for w in s})
class S:  # Placer が要る物だけ
    def __init__(self, D): self.D, self.shift = D, 7

def setup(D, V, seed):
    pl = PL.Placer(S(D), "拡散")
    off = seed * 100000
    L = np.stack([pl._code(off + i) for i in range(V)])          # 候補 V 個（先頭が本物の語、残りは使わない語）
    wid = {w: i for i, w in enumerate(VOC)}
    return pl, L, wid

def run_single(D, V, seed, rounds=2):
    pl, L, wid = setup(D, V, seed)
    tok = sen = n = 0
    for s in SENTS:
        ids = np.array([wid[w] for w in s])
        X = pl.encode([L[i] for i in ids])
        g = pl.decode(X, len(s), L, rounds)[-1]
        tok += int((g == ids).sum()); n += len(s); sen += int((g == ids).all())
    return tok / n, sen / len(SENTS)

def run_stack(D, V, seed, rounds=2):
    """10 文を、文ごとの鍵（乱数の位相の板）を掛けて 1 本の波に重ね、鍵で戻してから読む。"""
    pl, L, wid = setup(D, V, seed)
    r = np.random.default_rng(seed + 99)
    keys = np.exp(2j * np.pi * r.random((len(SENTS), D))).astype(np.complex64)
    X = np.zeros(D, dtype=np.complex64)
    for k, s in zip(keys, SENTS):
        X += k * pl.encode([L[wid[w]] for w in s])
    tok = sen = n = 0
    for k, s in zip(keys, SENTS):
        ids = np.array([wid[w] for w in s])
        g = pl.decode(np.conj(k) * X, len(s), L, rounds)[-1]
        tok += int((g == ids).sum()); n += len(s); sen += int((g == ids).all())
    return tok / n, sen / len(SENTS)

DS = [16, 32, 64, 128, 256, 512, 1024, 2048]
SEEDS = range(20)
res = {}
for name, fn, V in (("1 文を 1 本の波・候補 300", run_single, 300), ("1 文を 1 本の波・候補 3,000", run_single, 3000),
                    ("10 文を 1 本に重ねる・候補 300", run_stack, 300)):
    res[name] = {}
    for D in DS:
        a = np.array([fn(D, V, sd) for sd in SEEDS])
        res[name][D] = dict(語=round(float(a[:, 0].mean()), 4), 文=round(float(a[:, 1].mean()), 4))
    print(name, json.dumps(res[name], ensure_ascii=False))
# 読み直し無し（rounds=0）も 1 つ
res0 = {D: round(float(np.mean([run_single(D, 300, sd, 0)[0] for sd in SEEDS])), 4) for D in DS}
print("読み直し無し・候補 300・語", res0)
json.dump(dict(res=res, 読み直し無し=res0, 語の種類=len(VOC), 語の数=sum(len(s) for s in SENTS)), open((sys.argv[1] if len(sys.argv) > 1 else "roundtrip.png") + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- 図 ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FONT = os.environ.get("KOTOYUI_FONT")                        # 日本語の字体のファイル（無ければ既定の字体）
if FONT:
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.titlesize": 12, "axes.titlecolor": "#0b0b0b", "axes.titlelocation": "left", "font.size": 10})
BL, OR, AQ, GR = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
D = 256
pl, L, wid = setup(D, 300, 0)
s = SENTS[1]
X = pl.encode([L[wid[w]] for w in s])
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2), dpi=200)
a = ax[0, 0]
a.scatter(X.real, X.imag, s=9, color=BL, linewidths=0)
a.axhline(0, color=GR, lw=.6); a.axvline(0, color=GR, lw=.6); a.set_aspect("equal")
a.set_title("①「友達 と 映画 を 見 ました」を波にした物（256 個の点）")
a.set_xlabel("実部"); a.set_ylabel("虚部")
a = ax[0, 1]
a.scatter(np.arange(D), np.abs(X), s=9, color=BL, linewidths=0)
a.set_title("② 同じ波の、点ごとの強さ"); a.set_xlabel("波の中の位置（0〜255）"); a.set_ylabel("強さ"); a.set_ylim(0)
a = ax[1, 0]
k = 2                                                           # 3 番目の語（映画）
Y = pl.take(X, k * 7)
sim = np.real(Y @ np.conj(L / np.linalg.norm(L, axis=1, keepdims=True)).T)
c = wid["映画"]
m = np.ones(300, bool); m[c] = False
a.scatter(np.arange(300)[m], sim[m], s=9, color=GR, linewidths=0, label="ほかの候補（299 個）")
a.scatter([c], [sim[c]], s=42, color=OR, linewidths=0, label="正解「映画」")
a.annotate("映画", (c, sim[c]), textcoords="offset points", xytext=(8, -3), color="#0b0b0b")
a.axhline(0, color=GR, lw=.6)
a.set_title("③ 3 番目の位置を読んだ時の、候補 300 個との近さ"); a.set_xlabel("候補の番号"); a.set_ylabel("近さ")
a.legend(frameon=False, loc="upper right")
a = ax[1, 1]
for (name, col, dy) in (("1 文を 1 本の波・候補 300", BL, 6), ("1 文を 1 本の波・候補 3,000", AQ, -12), ("10 文を 1 本に重ねる・候補 300", OR, -12)):
    y = [res[name][d]["語"] for d in DS]
    a.plot(range(len(DS)), y, color=col, lw=2, marker="o", ms=5, label=name)
a.set_xticks(range(len(DS))); a.set_xticklabels(DS); a.set_ylim(0, 1.05); a.grid(axis="y", color="#e8e7e2", lw=.6)
a.set_title("④ 波の長さと、語を正しく戻せた割合（20 回の平均）"); a.set_xlabel("波の長さ（点の数）"); a.set_ylabel("戻せた割合")
a.legend(frameon=False, loc="lower right")
fig.tight_layout(pad=1.6)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "roundtrip.png")
print("saved")
