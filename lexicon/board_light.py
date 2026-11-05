"""特別な語板（品詞語板・機能語語板）を光の器に学ばせ、持ち主が優先する特性で確かめる（第 1 版: 辞書の属性だけ）

持ち主の方針（2026-11-05）: 光による一括処理、融合（劣化あり）、消去、差分検証、逆回しを優先する。デジタルの重みは試験中は OK、後半で減らす。
器: 介入の研究（intervention/i02_reverse.PortDevice）と同じ式（画素ごとの位相の板 + 回折 + レンズ）。口 1 つ。出口の明るさ = 属性の割合。
  一括処理: 語板 B 枚の板を 1 つの束（B, 画素, 板）にして、同じ光の計算で一度に学ぶ・通す。
  学び方: 正解 = 観点ごとの割合を観点の数で割って並べたもの（合計 1）。残りの光は暗い出口へ（I10 の fit_board と同じ）。
確かめること:
  1. 学びの正確さ: 学んだ明るさ（観点ごとに正規化）と正解の差（全変動距離、観点の平均）。
  2. 逆回し: 属性の組（一部の観点）を入れ、どの語板から来たかを当てる。光を出口から逆に入れて口に戻る量 = 順の明るさ（器の性質）なので、
     明るさの積で並べる。物差し = 1 位の語板の「機能」（品詞語板は品詞 1 段目）が正しい割合。
  3. 融合（劣化あり）: 同じ機能の子の語板から親を作る 3 通り。
       和: 子の明るさを足す（光を重ねる。劣化なし）。
       板の平均: 子の位相の板を平均した器を 1 台作る（器そのものを融合。劣化あり）。
       まとめて学ぶ: 子の属性を混ぜた正解で 1 台を学ぶ（比べ）。
     物差し = 親の本当の割合（子の割合の平均）との差。
  4. 消去: 親（和）から子を 1 つ消す。その子にしかない属性の値が親から消え、ほかの値が残るか。
  5. 差分検証: 同じ機能で 1 つの観点だけ違う語板の対。明るさの差がいちばん大きい観点が、本当に違う観点と一致するか。
結果: 非公開の置き場 ../Project_Kotoyui_private/data/boards/light_results.json（と学んだ明るさ）
実行: python lexicon/board_light.py --boards <pos_boards.json と func_boards.json の置き場>
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(HERE, "..", "..", "Project_Kotoyui_private"))
VAL0 = 8


class BatchDevice(torch.nn.Module):
    """口 1 つの光の器を B 台束ねたもの（式は i02_reverse.PortDevice と同じ。板の位相だけが B 台ぶん別）。"""

    def __init__(self, B, n=192, plates=8, gap=32.0, seed=0, phase=None):
        super().__init__()
        self.B, self.n, self.plates, self.L = B, n, plates, 2 * n
        fx = torch.fft.fftfreq(self.L)
        self.register_buffer("H", torch.exp(1j * gap * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * fx) ** 2)).to(torch.complex64))
        g = torch.Generator().manual_seed(seed)
        ph = math.pi * (torch.rand(B, n, plates, generator=g) < 0.5).float() if phase is None else phase.clone()
        self.phase = torch.nn.Parameter(ph)
        y = torch.arange(n, dtype=torch.float32)
        self.register_buffer("port", (torch.exp(1j * 2 * math.pi * y / n) / math.sqrt(n)).to(torch.complex64))

    def bright(self, A):
        u = torch.nn.functional.pad(self.port.expand(self.B, self.n), (0, self.L - self.n))
        for k in range(self.plates):
            u = torch.cat([u[:, :self.n] * torch.exp(1j * self.phase[:, :, k]), u[:, self.n:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * self.H, dim=-1, norm="ortho")
        out = torch.fft.fft(u[:, :self.n], dim=-1, norm="ortho")
        return out.abs()[:, VAL0:VAL0 + A] ** 2


def slots_of(boards):
    S = sorted({(v, x) for b in boards for v, xs in b["attrs"].items() for x in xs})
    return S, {s: i for i, s in enumerate(S)}


def target(b, SLOT, A):
    q = np.zeros(A, dtype=np.float32)
    nv = len(b["attrs"])
    for v, xs in b["attrs"].items():
        for x, p in xs.items():
            q[SLOT[(v, x)]] += p / nv
    return q


def train(Q, n, steps, seed, lr=0.03, phase=None):
    B, A = Q.shape
    dev = BatchDevice(B, n=n, seed=seed, phase=phase)
    q = torch.tensor(Q)
    opt = torch.optim.Adam(dev.parameters(), lr=lr)
    for _ in range(steps):
        W = dev.bright(A)
        p = torch.cat([W, (1 - W.sum(1, keepdim=True)).clamp(min=1e-9)], 1)
        loss = -(torch.cat([q, torch.zeros(B, 1)], 1) * torch.log(p + 1e-9)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return dev, dev.bright(A).numpy()


def per_view(w, S, views):
    """明るさを観点ごとに正規化した割合 {観点: 配列}。"""
    out = {}
    for v in views:
        ids = [i for i, (vv, _) in enumerate(S) if vv == v]
        x = w[ids] + 1e-9
        out[v] = (ids, x / x.sum())
    return out


def tv(w, b, S, SLOT):
    pv = per_view(w, S, b["attrs"])
    d = []
    for v, xs in b["attrs"].items():
        ids, p = pv[v]
        t = np.zeros(len(ids))
        for x, pr in xs.items():
            t[ids.index(SLOT[(v, x)])] = pr
        d.append(0.5 * np.abs(p - t).sum())
    return float(np.mean(d))


def label_of(b):
    return b.get("func") or b["name"].split("-")[0]


def reverse_test(W, boards, S, SLOT, query_views, rng, n_q=3):
    """逆回し: 語板ごとに、その語板の属性から query_views の観点の値を 1 つずつ（割合で）引いた問いを n_q 個作り、明るさの積で語板を当てる。"""
    Wn = W / W.sum(1, keepdims=True)
    logW = np.log(Wn + 1e-6)
    ok_lab, ok_word, n = 0, 0, 0
    for i, b in enumerate(boards):
        vs = [v for v in query_views if v in b["attrs"]]
        if not vs:
            continue
        for _ in range(n_q):
            ids = [SLOT[(v, rng.choices(list(b["attrs"][v]), weights=list(b["attrs"][v].values()))[0])] for v in vs]
            ll = logW[:, ids].sum(1)
            best = ll.max()
            top = [j for j in range(len(boards)) if ll[j] >= best - 1e-9]   # 同点は全部（属性が同じ語板どうし）
            ok_lab += np.mean([label_of(boards[j]) == label_of(b) for j in top])
            ok_word += (i in top) / len(top)
            n += 1
    return dict(問い=n, 機能が正しい=round(ok_lab / max(n, 1), 3), 語板が正しい=round(ok_word / max(n, 1), 3))


def fusion_test(dev, W, boards, S, SLOT, Q, n, steps, seed):
    groups = defaultdict(list)
    for i, b in enumerate(boards):
        groups[label_of(b)].append(i)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    A = len(S)
    res = defaultdict(list)
    names = list(groups)
    truth = np.stack([Q[groups[k]].mean(0) for k in names])          # 親の本当の割合（子の平均）
    with torch.no_grad():
        ph = torch.stack([dev.phase[groups[k]].mean(0) for k in names])   # 位相の板の平均（器の融合）
        avg_dev = BatchDevice(len(names), n=n, plates=ph.shape[2], phase=ph)
        W_avg = avg_dev.bright(A).numpy()
        # 位相は角度なので、平均は「単位円の上の平均の向き」でも試す
        z = torch.stack([torch.exp(1j * dev.phase[groups[k]]).mean(0) for k in names])
        W_circ = BatchDevice(len(names), n=n, plates=z.shape[2], phase=torch.angle(z)).bright(A).numpy()
    _, W_pool = train(truth, n, steps, seed + 7)
    W_sum = np.stack([W[groups[k]].sum(0) for k in names])

    def dist(Wp, t):
        a, b2 = Wp / (Wp.sum() + 1e-12), t / t.sum()
        return 0.5 * float(np.abs(a - b2).sum())

    for j, k in enumerate(names):
        res["和（光を重ねる）"].append(dist(W_sum[j], truth[j]))
        res["板の平均（位相をそのまま平均）"].append(dist(W_avg[j], truth[j]))
        res["板の平均（向きの平均）"].append(dist(W_circ[j], truth[j]))
        res["まとめて学ぶ"].append(dist(W_pool[j], truth[j]))
    out = {m: round(float(np.mean(v)), 3) for m, v in res.items()}
    out["親の数"] = len(names)
    out["子の数（平均）"] = round(float(np.mean([len(v) for v in groups.values()])), 1)
    out["光の総量（板の平均 / 子の平均）"] = round(float(W_circ.sum(1).mean() / W.sum(1).mean()), 3)
    return out


def erase_test(W, boards, S, thr=0.02):
    groups = defaultdict(list)
    for i, b in enumerate(boards):
        groups[label_of(b)].append(i)
    lost_ok, kept_ok, n_unique, n_shared = 0, 0, 0, 0
    examples = []
    for k, ids in groups.items():
        if len(ids) < 3:
            continue
        P = W[ids].sum(0)
        for e in ids:
            rest = [i for i in ids if i != e]
            P2 = W[rest].sum(0)
            own = {s for s, p in zip(S, W[e] / W[e].sum()) if p > thr}
            others = {s for i in rest for s, p in zip(S, W[i] / W[i].sum()) if p > thr}
            uniq, shared = own - others, own & others
            Pn, P2n = P / P.sum(), P2 / P2.sum()
            for s in uniq:
                n_unique += 1
                lost_ok += P2n[S.index(s)] < 0.25 * Pn[S.index(s)]   # 消えた（4 分の 1 未満に落ちた）
            for s in shared:
                n_shared += 1
                kept_ok += P2n[S.index(s)] > 0.5 * Pn[S.index(s)]    # 残った（半分以上）
            if uniq and len(examples) < 5:
                examples.append(dict(親=k, 消した=boards[e]["name"], その子だけの属性=[":".join(s) for s in sorted(uniq)]))
    return dict(その子だけの属性が消えた=round(lost_ok / max(n_unique, 1), 3), 共有の属性が残った=round(kept_ok / max(n_shared, 1), 3),
                数=dict(その子だけ=n_unique, 共有=n_shared), 例=examples)


def diff_test(W, boards, S):
    ok, n, examples = 0, 0, []
    for i, j in itertools.combinations(range(len(boards)), 2):
        a, b = boards[i], boards[j]
        if label_of(a) != label_of(b) or set(a["attrs"]) != set(b["attrs"]):
            continue
        dv = [v for v in a["attrs"] if a["attrs"][v] != b["attrs"][v]]
        if len(dv) != 1:
            continue
        pa, pb = per_view(W[i], S, a["attrs"]), per_view(W[j], S, b["attrs"])
        d = {v: 0.5 * float(np.abs(pa[v][1] - pb[v][1]).sum()) for v in a["attrs"]}
        guess = max(d, key=d.get)
        ok += guess == dv[0]
        n += 1
        if len(examples) < 5:
            examples.append(dict(対=f"{a['name']} / {b['name']}", 本当に違う観点=dv[0], 光で一番違う観点=guess))
    return dict(対=n, 正しい=round(ok / max(n, 1), 3), 例=examples)


def run(kind, boards, query_views, a):
    S, SLOT = slots_of(boards)
    A = len(S)
    assert VAL0 + A <= a.n, f"出口が足りない（属性 {A}、画素 {a.n}）"
    Q = np.stack([target(b, SLOT, A) for b in boards])
    t0 = time.perf_counter()
    dev, W = train(Q, a.n, a.steps, a.seed)
    t_train = time.perf_counter() - t0
    rng = random.Random(a.seed)
    res = dict(語板=len(boards), 属性=A, 学ぶ秒=round(t_train, 1),
               学びの正確さ_全変動=round(float(np.mean([tv(W[i], b, S, SLOT) for i, b in enumerate(boards)])), 3),
               光の総量=round(float(W.sum(1).mean()), 3),
               逆回し={"+".join(qv): reverse_test(W, boards, S, SLOT, qv, rng) for qv in query_views},
               融合=fusion_test(dev, W, boards, S, SLOT, Q, a.n, a.steps, a.seed),
               消去=erase_test(W, boards, S),
               差分検証=diff_test(W, boards, S))
    return res, W, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=os.path.join(PRIV, "data", "boards"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=192, help="画素（出口の数の上限）")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or a.boards
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    P = json.load(open(os.path.join(a.boards, "pos_boards.json"), encoding="utf-8"))
    F = json.load(open(os.path.join(a.boards, "func_boards.json"), encoding="utf-8"))
    results = {}
    results["品詞語板"], Wp, Sp = run("pos", P, [["活用型", "語種"], ["活用型", "書字", "長さ"], ["活用形", "語種", "書字"]], a)
    results["機能語語板"], Wf, Sf = run("func", F, [["品詞"], ["品詞", "活用形"], ["意味"], ["活用型", "活用形", "長さ"]], a)
    results["設定"] = dict(n=a.n, steps=a.steps, seed=a.seed)
    json.dump(results, open(os.path.join(out, f"light_results_s{a.seed}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(out, f"light_bright_s{a.seed}.npz"), pos=Wp, func=Wf,
                        pos_slots=np.array([":".join(s) for s in Sp]), func_slots=np.array([":".join(s) for s in Sf]))
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
