"""実験08: 光の共鳴グレインを直す — 語の区別がつく入れ方・像の面の板（実験07 の続き）

着想: 実験07 の光の共鳴（10 択 17〜19%）を見直すと、語をマス 1 つずつに置いた光 → 4f 系（周波数の面の板 = ずらしても同じ働きの畳み込み）
      → レンズの後の明るさ、という形では、**語 1 つのとき、どの語でも出力が同じ**になる（ずらした光はレンズの後の明るさが変わらない。手元で確かめ、差 1e-6 以下）。
      語が複数でも「語どうしのマスの距離」しか見えない。そこで、ずらしても同じ、を崩す 2 つの直し方と、その組み合わせを比べる。
方法（--input × --plates × --read）:
      入れ方 pixel  = 実験07 と同じ（語 = マス 1 つの明るさ）
             spread = 語ごとに決まった向きの平面波（64 × 64 の向きに黄金比の歩幅で散らす、SN2 の拡散と同じ）を重ね、決まったでたらめの位相の模様を掛ける
      像の面の板 none = 実験07 と同じ（周波数の面の板だけ）
                 xy   = 4f 系の各段の前に、像の面にも位相の板（64 × 64、学習する。光の部分）を置く
      読み方 hartley = レンズの後の明るさのハートレー係数 1 つ = 語 1 つ（実験07、符号の表なし）
             image   = レンズを通さず、最後の像の面の明るさのマス 1 つ = 語 1 つ（符号の表なし）
             codes   = 窓 256 個 + 学習する符号（語 × 窓、デジタル。比べる）
      比べる: --read lowrank = 光を使わない、学習するデジタルの低ランク（語の袋 → 512 → 語、約 420 万個）。同じ学び方で届く上の目安。
比べる（同じ 10 択、実験06・07 と同じ問の作り方・同じ seed で同じ問）: でたらめ、語の重なり、数え上げの共鳴（PMI）、重なり + PMI（kotoyui 0.1.0 の芯）。
材料: RealPersonaChat・JMultiWOZ（CC BY-SA 4.0）、会話ごとに教え 9 : 試し 1（kotoyui.load_conversations）。
      --layout svd（実験10）: 語のマスの並べ方を、よく出る順から「次の発言に一緒に出やすい語どうしを近くに」（正の PMI の上位 2 成分で 64 列 × 64 行）に変える。
      --probe（実験11）: 推し量る力を測る（表に無い組だけの問・決め手の語を似た語に入れ替える介入・取り除く）。probe() を参照。
結果: results/exp08.txt（実験10 は results/exp10.txt、実験11 は results/exp11.txt）
実行: python experiments/exp08_resonance_fix.py --input spread --plates xy --seed 0 --out results/exp08.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import kotoyui as K  # noqa: E402
import textlight  # noqa: E402

L = 64
NV = L * L - 1                     # 語の数（マス 0 = 直流 は使わない）


class Body(textlight.Multi4f):
    """Multi4f に、各段の前の像の面の位相の板（plates="xy"）と、最後のレンズを通さない読み（lens=False）を足したもの。"""

    def __init__(self, stages, width, seed, plates):
        super().__init__(stages, width, seed)
        self.plates = plates
        if plates == "xy":
            g = torch.Generator().manual_seed(seed + 7)
            self.iphase = torch.nn.Parameter(math.pi * (2 * torch.rand(stages, L, L, generator=g) - 1))

    def screen(self, u, lens=True):
        for s in range(self.stages):
            if self.plates == "xy":
                u = textlight.p2.apply_plate(u, self.iphase[s])
            U = torch.fft.fft2(u, norm="ortho")[:, None]
            filt = torch.polar(torch.exp(self.logamp[s].clamp(max=3)), self.phase[s])[None]
            v = torch.fft.ifft2(U * filt, norm="ortho")
            w = torch.complex(self.mix[s, :, 0], self.mix[s, :, 1])
            u = textlight.p2.normalize((v * w[None, :, None, None]).sum(1))
            if s < self.stages - 1:
                I = textlight.p2.intensity(u)
                u = textlight.p2.normalize((I / (I.mean((-2, -1), keepdim=True) + 1e-12)).clamp(min=0).sqrt().to(torch.complex64))
        I = textlight.p2.intensity(textlight.p2.lens(u) if lens else u)
        return I / (I.mean((-2, -1), keepdim=True) + 1e-12)


class ResonanceGrain(torch.nn.Module):
    """前の発言の語の袋 (B, 4096) → 光の場（pixel / spread）→ 4f 系 + 平方根（像の面の板 none / xy）→ 読み（hartley / image / codes）→ 次に出る語の対数確率。
    read="lowrank" は光を使わない比べる用（袋 → 512 → 語、デジタル）。"""

    def __init__(self, stages=4, width=4, seed=0, read="hartley", input="pixel", plates="none", K=256, rank=512):
        super().__init__()
        self.read, self.input = read, input
        if read == "lowrank":
            g = torch.Generator().manual_seed(seed)
            self.U = torch.nn.Parameter(torch.randn(L * L, rank, generator=g) / 8)
            self.V = torch.nn.Parameter(torch.randn(rank, L * L, generator=g) / rank ** 0.5)
        else:
            self.body = Body(stages, width, seed, plates)
        if input == "spread":
            pos = (torch.arange(L * L) * textlight.stride(L * L)) % (L * L)
            yy = torch.arange(L, dtype=torch.float32)
            ph = 2 * math.pi * ((pos % L).float()[:, None, None] * yy[None, None, :] + (pos // L).float()[:, None, None] * yy[None, :, None]) / L
            self.register_buffer("waves", torch.polar(torch.ones_like(ph), ph).view(L * L, L * L))   # 語 × マス（決まった平面波）
            gr = torch.Generator().manual_seed(12345)
            self.register_buffer("ref", torch.polar(torch.ones(L, L), 2 * math.pi * torch.rand(L, L, generator=gr)))
        if read == "codes":
            self.register_buffer("det", textlight.p2.window_grid(L, L, K, 4))
            self.codes = torch.nn.Parameter(torch.randn(L * L, K, generator=torch.Generator().manual_seed(seed)))
        self.gain = torch.nn.Parameter(torch.tensor(3.0))
        self.bias = torch.nn.Parameter(torch.zeros(L * L))

    def field(self, bag):
        a = bag / (bag.pow(2).mean(1, keepdim=True).sqrt() + 1e-6)
        if self.input == "spread":
            u = (a.to(torch.complex64) @ self.waves).view(-1, L, L) * self.ref
            return textlight.p2.normalize(u)
        return a.view(-1, L, L).to(torch.complex64)

    def forward(self, bag):
        """bag (B, 4096)（0/1）→ 語の対数確率 (B, 4096)。"""
        if self.read == "lowrank":
            v = (bag / bag.sum(1, keepdim=True).clamp(min=1)) @ self.U @ self.V
        else:
            I = self.body.screen(self.field(bag), lens=self.read != "image")
            if self.read == "codes":
                v = textlight.p2.code_scores(textlight.p2.read_windows(I, self.det), self.codes, 1.0)
            elif self.read == "image":
                v = I.flatten(1)
            else:
                Fq = torch.fft.fft2(I, norm="ortho")
                v = (Fq.real - Fq.imag).flatten(1)
        v = (v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True) + 1e-6)
        z = self.gain * v + self.bias
        z[:, 0] = -1e4                                              # マス 0 は語に使わない
        return F.log_softmax(z, 1)

    def params(self):
        if self.read == "lowrank":
            return 0, self.U.numel() + self.V.numel() + 1 + self.bias.numel()
        return sum(q.numel() for q in self.body.parameters()), self.gain.numel() + self.bias.numel() + (self.codes.numel() if self.read == "codes" else 0)


def probe(a, mat, cases, P, Q, wid, m, dev, log_uni, rng):
    """推し量る力を測る（実験11、--probe）。10 択（ほかの会話から）を、語の集まりのまま次の 3 通りで解き直す。
    unseen: 正解の返事と前の発言の間に、数えた表の組（2 回以上）が 1 つも無い問だけ（表を覚えるだけでは点が 0 になる問）。
    swap:   前の発言のうち、正解との PMI がいちばん効いている語（決め手の語）を、いちばん似た語に入れ替える（介入。似た語 = 次の発言に出る語の正の PMI が近い語）。
    drop:   決め手の語を取り除く（比べる）。
    方法: 語の重なり（A）・数え上げの PMI（B）・語板の光（G、kotoyui 0.2.0 の LightResonance）・この光路（L）。
    agree は入れ替えの前後で 1 番に選ぶ候補が同じ割合。"""
    rm_ = {i: w for w, i in wid.items()}
    M = torch.zeros(L * L, L * L)                                     # 似た語: 次の発言に出る語の正の PMI の行どうしの近さ
    for p_, q_ in zip(P, Q):
        M[torch.tensor(p_)[:, None], torch.tensor(q_)[None, :]] += 1.0
    tot, r_, c_ = M.sum(), M.sum(1, keepdim=True), M.sum(0, keepdim=True)
    M = torch.log((M * tot / (r_ @ c_).clamp(min=1)).clamp(min=1e-9)).clamp(min=0)
    Mn = M / (M.norm(dim=1, keepdim=True) + 1e-9)
    S = Mn @ Mn.T
    S.fill_diagonal_(-1)
    S[0] = -1
    S[:, 0] = -1
    nn_ = S.argmax(1).tolist()
    G = K.load_light_resonance(mat)
    cache = {}

    @torch.no_grad()
    def lift(uw):
        key = frozenset(uw)
        if key not in cache:
            l = [wid[w] for w in uw if w in wid]
            cache[key] = (m(dense1(l).to(dev))[0].cpu() - log_uni) if l else None
        return cache[key]

    def dense1(l):
        x = torch.zeros(1, L * L)
        x[0, l] = 1.0
        return x

    def sc(uw, cw_, how):
        if how == "A":
            return K.overlap(uw, cw_)
        if how == "B":
            return mat.resonance(uw, cw_)
        if how == "G":
            return G.resonance(uw, cw_)
        lf = lift(uw)
        cl = [wid[w] for w in cw_ if w in wid]
        return 0.0 if lf is None or not cl else float(lf[cl].clamp(min=0).mean())

    def pick(uw, cands, how):
        s = [sc(uw, c, how) + 1e-6 * rng.random() for c in cands]
        return max(range(len(s)), key=s.__getitem__)
    out = {}
    rows = []
    for src, ctx, gold, negs, same in cases:
        uw = set(K.analyze(ctx[-1])[0])
        cands = [set(K.analyze(t)[0]) for t in [gold] + negs]
        if not uw:
            continue
        key_w = [w for w in uw if w in wid]
        dec = max(key_w, key=lambda x: sum(mat.pmi(x, y) for y in cands[0])) if key_w else None
        rows.append((uw, cands, dec))
    for how in ("A", "B", "G", "L"):
        r = {}
        uns = [(uw, c) for uw, c, _ in rows if mat.resonance(uw, c[0]) == 0]
        r["unseen_n"] = len(uns)
        r["unseen_top1"] = sum(pick(uw, c, how) == 0 for uw, c in uns) / max(1, len(uns))
        base = [(uw, c, d) for uw, c, d in rows if d is not None and sum(mat.pmi(d, y) for y in c[0]) > 0]
        r["swap_n"] = len(base)
        r["base_top1"] = sum(pick(uw, c, how) == 0 for uw, c, _ in base) / len(base)
        for kind in ("swap", "drop"):
            ok = agree = 0
            for uw, c, d in base:
                uw2 = (uw - {d}) | ({rm_[nn_[wid[d]]]} if kind == "swap" else set())
                p0, p1 = pick(uw, c, how), pick(uw2, c, how)
                ok += p1 == 0
                agree += p0 == p1
            r[f"{kind}_top1"] = ok / len(base)
            r[f"{kind}_agree"] = agree / len(base)
        out[how] = r
        print("推し量り", how, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    ex = [(d, rm_[nn_[wid[d]]]) for _, _, d in rows[:40] if d is not None]
    out["swap_examples"] = ex[:20]
    print("入れ替えの例", ex[:20], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--read", default="hartley", choices=["hartley", "image", "codes", "lowrank"])
    ap.add_argument("--input", default="pixel", choices=["pixel", "spread"])
    ap.add_argument("--plates", default="none", choices=["none", "xy"])
    ap.add_argument("--layout", default="freq", choices=["freq", "svd"], help="語のマスの並べ方: freq = よく出る順（実験07・08）、svd = 次の発言に一緒に出やすい語どうしを近くに（実験10）")
    ap.add_argument("--probe", action="store_true", help="推し量る力を測る（実験11: 表に無い組の問・決め手の語を似た語に入れ替える介入）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=3000, help="試しの問の数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp08.txt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    mat, tg = K.load_tables()
    convs = K.load_conversations()
    # 語の表（よく出る中身の語 4,095）と、教えの組（前の発言の語 → 次の発言の語）
    freq = Counter()
    for _, turns in convs["train"]:
        for t in turns:
            freq.update(K.analyze(t)[0])
    vocab = [w for w, _ in freq.most_common(NV)]
    wid = {w: i + 1 for i, w in enumerate(vocab)}
    ids = lambda t: [wid[w] for w in K.analyze(t)[0] if w in wid]
    P, Q = [], []
    for _, turns in convs["train"]:
        A = [ids(t) for t in turns]
        for i in range(len(A) - 1):
            if A[i] and A[i + 1]:
                P.append(A[i])
                Q.append(A[i + 1])
    if a.layout == "svd":                                       # 語のマスの並べ方（実験10）: 次の発言に一緒に出やすい語どうしを近くのマスに
        M = torch.zeros(L * L, L * L)
        for p_, q_ in zip(P, Q):
            M[torch.tensor(p_)[:, None], torch.tensor(q_)[None, :]] += 1.0
        M = M + M.T
        tot = M.sum()
        r_ = M.sum(1, keepdim=True)
        M = torch.log((M * tot / (r_ @ r_.T).clamp(min=1)).clamp(min=1e-9)).clamp(min=0)   # 正の PMI（対称）
        U, _, _ = torch.svd_lowrank(M[1:, 1:], q=4, niter=4)        # 語 4,095 の 2 次元の座標（上位 2 成分）
        o1 = torch.argsort(U[:, 0])                                  # 1 成分目で 64 列に分け、列の中を 2 成分目で並べる
        pos = torch.zeros(NV, dtype=torch.long)
        for c in range(L):
            col = o1[c * L:(c + 1) * L]
            col = col[torch.argsort(U[col, 1])]
            for r, w in enumerate(col.tolist()):
                pos[w] = c * L + r
        rm = [0] + pos.tolist()                                      # 元の番号（語 w は w + 1）→ 新しいマス
        free = sorted(set(range(1, L * L)) - set(rm[1:]))            # マス 0 に当たった語は、1 つ空いたマスへ（マス 0 は使わない）
        rm = [0] + [m_ if m_ != 0 else free.pop() for m_ in rm[1:]]
        assert len(set(rm[1:])) == NV and 0 not in rm[1:]
        P = [[rm[i] for i in p_] for p_ in P]
        Q = [[rm[i] for i in q_] for q_ in Q]
        wid = {w: rm[i] for w, i in wid.items()}
        print(f"語のマスを並べ替えた（正の PMI の上位 2 成分、{time.perf_counter() - t0:.0f} 秒）", flush=True)
    uni = torch.zeros(L * L)
    for q in Q:
        uni[q] += 1.0 / len(q)
    log_uni = torch.log((uni + 1) / (uni + 1).sum())
    print(f"語 {len(vocab)}・教えの組 {len(P)}（{time.perf_counter() - t0:.0f} 秒）", flush=True)

    def dense(lists, normalize=False):
        x = torch.zeros(len(lists), L * L)
        for r, l in enumerate(lists):
            x[r, l] = 1.0 / len(l) if normalize else 1.0
        return x
    m = ResonanceGrain(a.stages, a.width, a.seed, a.read, a.input, a.plates).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    t1 = time.perf_counter()
    for k in range(a.steps):
        idx = torch.randint(len(P), (256,), generator=g).tolist()
        x = dense([P[i] for i in idx]).to(dev)
        y = dense([Q[i] for i in idx], True).to(dev)
        loss = -(y * m(x)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{a.steps} 損失 {loss.item():.3f} {time.perf_counter() - t1:.0f} 秒", flush=True)
    train_s = time.perf_counter() - t1
    m.eval()
    # 試し: 実験06 と同じ作り方の 10 択
    rng = random.Random(a.seed)
    pool = [t for _, turns in convs["test"] for t in turns]
    cases = []
    for src, turns in convs["test"]:
        for i in range(3, len(turns), 4):
            negs = []
            while len(negs) < 9:
                t2 = rng.choice(pool)
                if t2 != turns[i] and t2 not in turns:
                    negs.append(t2)
            same = [t for j, t in enumerate(turns) if abs(j - i) > 1 and t != turns[i]]
            cases.append((src, turns[max(0, i - 3):i], turns[i], negs, rng.sample(same, min(9, len(same)))))
    rng.shuffle(cases)
    cases = cases[:a.n]
    lp_cache = {}

    @torch.no_grad()
    def light_lp(u):
        if u not in lp_cache:
            l = ids(u)
            lp_cache[u] = (m(dense([l]).to(dev))[0].cpu() - log_uni) if l else None
        return lp_cache[u]

    def light(u, cand):
        lift = light_lp(u)
        cl = ids(cand)
        if lift is None or not cl:
            return 0.0
        return float(lift[cl].clamp(min=0).mean())

    def score(ctx, cand, how):
        u = ctx[-1]
        uw, _ = K.analyze(u)
        cw_, _ = K.analyze(cand)
        sc = 0.0
        if "A" in how:
            sc += K.overlap(uw, cw_)
        if "B" in how:
            sc += mat.resonance(uw, cw_)
        if "L" in how:
            sc += light(u, cand)
        return sc
    rec = dict(exp="08", **vars(a), n_vocab=len(vocab), n_pairs=len(P), train_s=train_s, train_loss=loss.item(),
               params_light=m.params()[0], params_digital=m.params()[1], runs=[], version=K.__version__, torch=torch.__version__)
    for hard in (False, True):
        for how, name in (("R", "でたらめ"), ("A", "語の重なり"), ("B", "数え上げの共鳴（PMI）"), ("AB", "重なり + PMI（0.1.0 の芯）"),
                          ("L", "光の共鳴"), ("AL", "重なり + 光の共鳴"), ("ABL", "重なり + PMI + 光の共鳴")):
            top1 = 0
            by = {}
            for src, ctx, gold, negs, same in cases:
                cands = [gold] + (same if hard else negs)
                scs = [rng.random() for _ in cands] if how == "R" else [score(ctx, c, how) + 1e-6 * rng.random() for c in cands]
                ok = all(scs[0] > s for s in scs[1:])
                top1 += ok
                b = by.setdefault(src, [0, 0])
                b[0] += ok
                b[1] += 1
            r = dict(hard=hard, how=how, name=name, top1=top1 / len(cases), by_source={k: v[0] / v[1] for k, v in by.items()})
            rec["runs"].append(r)
            print("同じ会話から" if hard else "ほかの会話から", name, round(r["top1"], 3), {k: round(v, 3) for k, v in r["by_source"].items()}, flush=True)
    if a.probe:
        rec["probe"] = probe(a, mat, cases, P, Q, wid, m, dev, log_uni, rng)
    rec["total_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
