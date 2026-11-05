"""語板 v1 を光の器に学ばせ、確かめる（内容語、属性 約 2,000、画素 2,048）

持ち主の方針（2026-11-05）: 一括処理・融合（劣化あり）・消去・差分・逆回しを優先。前後の文から「的はずれな文」を見分ける学びが大事。
器: board_light.BatchDevice（口 1 つ、板 K 枚、画素 n）。板は 4 枚（2026-11-05 の比べで 8 枚とほぼ同じ精度、計算は半分）。
比べ: 板の位相を「画素ごと」に持つ器と、「少数の周波数成分の和」で持つ器（持ち主の案: 重みを周波数成分で持つ）。
確かめること:
  1. 学びの正確さ（全変動、帯ごと）
  2. 逆回し: 動的の帯（C）の属性 3 つから、クラス（D の最大）と語そのものを当てる
  3. 融合: クラスごとの親 = 子の明るさの和 / まとめて学ぶ / 板の平均
  4. 消去: 親から子を 1 つ消したとき、その子だけの属性が消えるか
  5. 崩した文を見分ける: 試しの文（属性を数えるのに使っていない文）の「名詞・動詞・役」と、
       (a) 役を入れ替えた組（リンゴが太郎を食べた）、(b) 名詞を別の語に替えた組 を、語板の明るさで見分ける（AUROC）。
       点 = 名詞の語板の「動詞/役」の明るさ（C の帯の中の割合）+ 動詞の語板の「主語:クラス」「目的語:クラス」の明るさ。
       比べ: 数えた割合そのもの（デジタル）。
結果: 非公開の置き場 ../Project_Kotoyui_private/data/boards/v1/light_v1_s<seed>.json
実行: python lexicon/content_light.py --dir <content_boards.json の置き場> --n_boards 5000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board_light as BL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(HERE, "..", "..", "Project_Kotoyui_private"))


class FourierDevice(BL.BatchDevice):
    """板の位相を、低い周波数成分 m 個の和で持つ器（重み = 板 1 枚あたり 2m）。"""

    def __init__(self, B, n, plates, m, seed=0):
        super().__init__(B, n=n, plates=plates, seed=seed)
        del self.phase
        g = torch.Generator().manual_seed(seed)
        self.coef = torch.nn.Parameter(torch.randn(B, plates, m, 2, generator=g) * 0.5)
        self.m = m

    @property
    def phase(self):
        c = torch.complex(self.coef[..., 0], self.coef[..., 1])            # (B, K, m)
        full = torch.zeros(self.B, self.plates, self.n // 2 + 1, dtype=torch.complex64)
        full[:, :, :self.m] = c
        return torch.fft.irfft(full, n=self.n, norm="ortho").permute(0, 2, 1) * math.sqrt(self.n)   # (B, n, K)


def train_dev(Q, n, K, steps, seed, m=None, lr=0.03, batch=500):
    """一括で学ぶ（batch 枚ずつ）。戻り値: 明るさ (B, A)、器の重みの数（1 枚あたり）。"""
    B, A = Q.shape
    out = np.zeros((B, A), dtype=np.float32)
    for s in range(0, B, batch):
        q = torch.tensor(Q[s:s + batch])
        dev = FourierDevice(len(q), n, K, m, seed=seed + s) if m else BL.BatchDevice(len(q), n=n, plates=K, seed=seed + s)
        opt = torch.optim.Adam(dev.parameters(), lr=lr)
        for _ in range(steps):
            W = dev.bright(A)
            p = torch.cat([W, (1 - W.sum(1, keepdim=True)).clamp(min=1e-9)], 1)
            loss = -(torch.cat([q, torch.zeros(len(q), 1)], 1) * torch.log(p + 1e-9)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out[s:s + batch] = dev.bright(A).numpy()
    return out, (K * 2 * m if m else K * n)


def band_ids(S):
    ids = defaultdict(list)
    for i, (b, _) in enumerate(S):
        ids[b].append(i)
    return ids


def band_prob(w, ids, b, j):
    x = w[ids[b]] + 1e-9
    return float(x[ids[b].index(j)] / x.sum()) if j in ids[b] else 0.0


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def odd_sentence_test(W, boards, S, SLOT, triples, cls_of, rng):
    """崩した文を見分ける。W: 明るさ（または正解の割合）。"""
    idx = {b["name"]: i for i, b in enumerate(boards)}
    ids = band_ids(S)
    nouns = [b["name"] for b in boards if b["pos"] == "名詞"]

    def score(noun, verb, role):
        s = 0.0
        if noun in idx:
            s += band_prob(W[idx[noun]], ids, "C", SLOT.get(("C", f"{verb}/{role}"), -1))
        if verb in idx and noun in cls_of:
            s += band_prob(W[idx[verb]], ids, "C", SLOT.get(("C", f"{'主語' if role == 'する' else '目的語'}:{cls_of[noun]}"), -1))
        return s

    T = [t for t in triples if t["noun"] in idx and t["verb"] in idx]
    pos = [score(t["noun"], t["verb"], t["role"]) for t in T]
    swap = [score(t["noun"], t["verb"], "される" if t["role"] == "する" else "する") for t in T]
    repl = [score(rng.choice(nouns), t["verb"], t["role"]) for t in T]
    return dict(組=len(T), 役の入れ替え=round(auroc(pos, swap), 3), 名詞の入れ替え=round(auroc(pos, repl), 3),
                点が0の割合=round(float(np.mean(np.array(pos) == 0)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(PRIV, "data", "boards", "v1"))
    ap.add_argument("--n_boards", type=int, default=5000)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--plates", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--fourier", type=int, nargs="*", default=[64, 256], help="周波数成分で持つ器の成分の数（比べ、--n_fourier 枚で）")
    ap.add_argument("--n_fourier", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = random.Random(a.seed)
    boards = json.load(open(os.path.join(a.dir, "content_boards.json"), encoding="utf-8"))[:a.n_boards]
    boards = [b for b in boards if b["attrs"]]
    meta = json.load(open(os.path.join(a.dir, "slots.json"), encoding="utf-8"))
    triples = json.load(open(os.path.join(a.dir, "test_triples.json"), encoding="utf-8"))
    for b in boards:                                                 # 融合・消去・逆回しの「機能」= いちばん明るいクラス
        d = b["attrs"].get("D")
        b["func"] = max(d, key=d.get).split(":", 1)[1] if d else "クラスなし"
    cls_of = {b["name"]: b["func"] for b in boards if b["pos"] == "名詞" and b["func"] != "クラスなし"}
    S, SLOT = BL.slots_of(boards)
    A = len(S)
    assert BL.VAL0 + A <= a.n, f"画素が足りない（属性 {A}）"
    Q = np.stack([BL.target(b, SLOT, A) for b in boards])
    print("語板", len(boards), "属性", A, flush=True)

    res = dict(設定=vars(a), 語板=len(boards), 属性=A, 名札=sum(len(v) for v in meta["slots"].values()))
    t0 = time.perf_counter()
    W, n_w = train_dev(Q, a.n, a.plates, a.steps, a.seed)
    res["学ぶ秒"] = round(time.perf_counter() - t0, 1)
    res["器の重み_1枚"] = n_w
    res["学びの正確さ_全変動"] = round(float(np.mean([BL.tv(W[i], b, S, SLOT) for i, b in enumerate(boards)])), 4)
    res["光の総量"] = round(float(W.sum(1).mean()), 3)
    print("学んだ", res["学ぶ秒"], res["学びの正確さ_全変動"], flush=True)
    res["逆回し"] = {q: BL.reverse_test(W, boards, S, SLOT, qv, rng) for q, qv in
                    [("C（動的）", ["C"]), ("A+C", ["A", "C"]), ("B+C", ["B", "C"])]}
    # 融合: クラスごと（子 5 つ以上）
    keep = defaultdict(list)
    for i, b in enumerate(boards):
        keep[b["func"]].append(i)
    sub = sorted(i for k, v in keep.items() if len(v) >= 5 and k != "クラスなし" for i in v[:40])
    sb = [boards[i] for i in sub]
    dev_dummy = BL.BatchDevice(len(sub), n=a.n, plates=a.plates)     # 板の平均の比べは、学んだ板が要るので小さく学び直す
    with torch.enable_grad():
        q = torch.tensor(Q[sub])
        opt = torch.optim.Adam(dev_dummy.parameters(), lr=0.03)
        for _ in range(a.steps):
            Wd = dev_dummy.bright(A)
            p = torch.cat([Wd, (1 - Wd.sum(1, keepdim=True)).clamp(min=1e-9)], 1)
            loss = -(torch.cat([q, torch.zeros(len(q), 1)], 1) * torch.log(p + 1e-9)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    res["融合"] = BL.fusion_test(dev_dummy, W[sub], sb, S, SLOT, Q[sub], a.n, a.steps, a.seed)
    res["消去"] = BL.erase_test(W[sub], sb, S)
    res["崩した文_光"] = odd_sentence_test(W, boards, S, SLOT, triples, cls_of, random.Random(1))
    res["崩した文_数えた割合"] = odd_sentence_test(Q, boards, S, SLOT, triples, cls_of, random.Random(1))
    # 周波数成分で持つ器（比べ）
    res["周波数成分の器"] = {}
    Qf = Q[:a.n_fourier]
    for m in a.fourier:
        t0 = time.perf_counter()
        Wf, n_wf = train_dev(Qf, a.n, a.plates, a.steps, a.seed, m=m)
        res["周波数成分の器"][f"成分{m}"] = dict(器の重み_1枚=n_wf, 学ぶ秒=round(time.perf_counter() - t0, 1),
                                           学びの正確さ_全変動=round(float(np.mean([BL.tv(Wf[i], b, S, SLOT) for i, b in enumerate(boards[:a.n_fourier])])), 4),
                                           光の総量=round(float(Wf.sum(1).mean()), 3))
        print("成分", m, res["周波数成分の器"][f"成分{m}"], flush=True)
    res["周波数成分の器"]["画素ごと（比べ）"] = dict(器の重み_1枚=n_w, 学びの正確さ_全変動=round(float(np.mean([BL.tv(W[i], b, S, SLOT) for i, b in enumerate(boards[:a.n_fourier])])), 4))
    json.dump(res, open(os.path.join(a.dir, f"light_v1_s{a.seed}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.save(os.path.join(a.dir, f"light_v1_bright_s{a.seed}.npy"), W.astype(np.float16))
    print(json.dumps({k: v for k, v in res.items() if k != "設定"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
