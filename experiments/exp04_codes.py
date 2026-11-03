"""実験04: デジタルの符号を減らす — 2 段で読む（方法 1）と、符号を学習しない（方法 2）

着想: 持ち主の依頼（2026-10-27「デジタル符号の問題は大きい。方法 1 と 2 をやってほしい」）。次の言葉を当てる光の部品は、答え 4096 種類 × 窓 256 個 =
      約 105 万個の符号（デジタル）を光路 1 本ごとに持つ（4f 系で学ぶ数の 73%、板の分担の 2DTG で 97%）。docs/ideas.md 5 節。
方法（光路の中身はどれも実験03 の光路 1 本と同じ: 拡散 → 4f 系 6 段 × 幅 8 + 平方根 → レンズ → 窓、9000 ステップ）:
  fixed       : 方法 2a。符号 4096 × 256 をでたらめのまま固定し、学習しない（学習するのは光の部分と温度だけ）。
  proto       : 方法 2b。符号を学習しない代わりに、答えごとの「窓の明るさの模様の平均（典型）」にそろえ直す（300 ステップごと、勾配は使わない。SN2 実験23 の典型）。
  two_freq    : 方法 1。答え 4096 種類を 64 組 × 64 番に分け、光路 A が組を窓 64 個（1 組 1 窓）で、光路 B が「組を入口の帯で教えた上で」組の中の番号を窓 64 個で読む。
                P(答え) = P(組) × P(番号 | 組)。デジタルは窓の背景光と倍率だけ。組の分け方: 出る回数の順に 64 個ずつ（よく出る単位どうしが同じ組）。
  pixel       : 方法 3（持ち主の案 2026-10-27「スクリーンの明るさを 2 次元フーリエ変換して符号の表と照らし合わせる」から）。
                64 × 64 のスクリーンの画素は 4096 個 = 答えの種類と同じ数なので、画素 1 つを答え 1 つの窓にする（符号の表なし）。
  hartley     : 方法 3。スクリーンの明るさを 2 次元フーリエ変換し、ハートレー変換（実部 − 虚部。明るさが実数なので 4096 個の独立な実数）の係数 1 つを答え 1 つにする。
                符号の表の代わりに、決まった模様（フーリエの縞）を答えの符号に使う形。光ではレンズがフーリエ変換をするので、明るさを測って光を出し直し、
                もう 1 枚のレンズを通せば作れる。どちらもデジタルは倍率 1 個と答えごとの背景光 4096 個だけ。
  two_clust   : 方法 1。組の分け方: 前に来る単位の分布が似た単位を同じ組に（64 個ずつにそろえた k-means）。
  比べる: 学習する符号（実験03 の光路 1 本、同じテスト）、GRU 28.33%・3-gram 30.88%（実験02）。
データ・テスト: 実験02・03 と同じ（現代の雑談、知らない話者 51 人、テスト 3 万位置、SentencePiece 4096、直前 4 単位）。口調のスイッチなし。
指標: 正解率・CE・訓練の正解率（2 万位置）。学習する数は光の部分・デジタルの部分（学習するもの）・持つだけの表（学習しない符号）に分ける。
      2 段で読む方式の正解率は、64 組すべてについて P(組) × P(番号 | 組) を出した、ちょうどの値。
結果: results/exp04.txt
実行: python experiments/exp04_codes.py --method two_clust --seed 0 --device cuda --out results/exp04.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp02_split_style as e2  # noqa: E402
import exp03_4f as e3  # noqa: E402
import textlight  # noqa: E402

p2 = textlight.p2
NG = 64


class PlaneNet(torch.nn.Module):
    """拡散 → 4f 系 + 平方根 → レンズの後の明るさ 64 × 64 = 4096 個を、そのまま（pixel）またはハートレー変換して（hartley）答えの点数にする。"""

    def __init__(self, n, V, C, stages, width, seed, read="pixel"):
        super().__init__()
        self.read = read
        self.field = textlight.SpreadField(n, V)
        self.body = textlight.Multi4f(stages, width, seed)
        self.gain = torch.nn.Parameter(torch.tensor(3.0))
        self.bias = torch.nn.Parameter(torch.zeros(C))

    def forward(self, X):
        I = self.body.screen(self.field(X))                         # (B, 64, 64)、平均 1
        if self.read == "hartley":
            Fq = torch.fft.fft2(I, norm="ortho")
            v = (Fq.real - Fq.imag).flatten(1)
            v = torch.cat([torch.zeros_like(v[:, :1]), v[:, 1:]], 1)   # 直流（全体の明るさ、答え 0 = 未知語の場所）は使わない
        else:
            v = torch.log(I.flatten(1) + 1e-3)                          # 明るさの対数（レンズの中心だけが極端に明るいのをならす）
        v = v[:, :self.bias.numel()]
        v = (v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True) + 1e-6)    # 答えの点数の大きさをそろえる（レンズの中心は特に明るいため）
        return self.gain * v + self.bias

    def params(self):
        return self.field.w.numel() + sum(q.numel() for q in self.body.parameters()), 1 + self.bias.numel()


class WindowNet(torch.nn.Module):
    """拡散 → 4f 系 + 平方根 → 窓 n_out 個（1 種類 1 窓）。デジタルは倍率 1 個と窓ごとの背景光 n_out 個だけ。"""

    def __init__(self, n, V, n_out, stages, width, seed):
        super().__init__()
        self.field = textlight.SpreadField(n, V)
        self.body = textlight.Multi4f(stages, width, seed)
        self.register_buffer("det", p2.window_grid(textlight.L, textlight.L, n_out, 4))
        self.gain = torch.nn.Parameter(torch.tensor(5.0))
        self.bias = torch.nn.Parameter(torch.zeros(n_out))

    def forward(self, X):
        return self.gain * p2.read_windows(self.body.screen(self.field(X)), self.det) + self.bias

    def params(self):
        return self.field.w.numel() + sum(q.numel() for q in self.body.parameters()), 1 + self.bias.numel()


def grouping(kind, X, y, V, seed=0):
    """答え（0..V-1）→ (組, 組の中の番号)。どちらの分け方も 64 個ずつ。"""
    if kind == "freq":
        order = torch.bincount(y, minlength=V).argsort(descending=True)
    else:
        # 前に来る単位の分布（直前の単位 → 答え の数）を 32 次元に縮め、64 個ずつにそろえた k-means
        C = torch.zeros(V, V)
        C.index_put_((y, X[:, -1]), torch.ones(len(y)), accumulate=True)
        C = torch.log1p(C)
        C = C - C.mean(1, keepdim=True)
        U, S, _ = torch.linalg.svd(C, full_matrices=False)
        E = U[:, :32] * S[:32]
        E = E / (E.norm(dim=1, keepdim=True) + 1e-6)
        g = torch.Generator().manual_seed(seed)
        cent = E[torch.randperm(V, generator=g)[:NG]].clone()
        for _ in range(8):
            D = torch.cdist(E, cent)                                   # (V, 64)
            assign = [-1] * V
            cap = [0] * NG
            left = V
            for i in D.flatten().argsort().tolist():                   # 近い組から、64 個まで埋める
                t, c = divmod(i, NG)
                if assign[t] < 0 and cap[c] < V // NG:
                    assign[t] = c
                    cap[c] += 1
                    left -= 1
                    if left == 0:
                        break
            assign = torch.tensor(assign)
            cent = torch.stack([E[assign == c].mean(0) for c in range(NG)])
        order = torch.cat([(assign == c).nonzero().squeeze(1) for c in range(NG)])
    grp = torch.empty(V, dtype=torch.long)
    idx = torch.empty(V, dtype=torch.long)
    pos = torch.empty(V, dtype=torch.long)
    pos[order] = torch.arange(V)
    grp, idx = pos // (V // NG), pos % (V // NG)
    return grp, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="two_clust", choices=["fixed", "proto", "two_freq", "two_clust", "pixel", "hartley"])
    ap.add_argument("--stages", type=int, default=6)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--max_train", type=int, default=2000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp04.txt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    (X, y, s), (Xt, yt, st), sp, tr_u = e2.build(a.max_train, a.seed)
    V = sp.get_piece_size()
    n = e2.N_CTX
    ti = torch.randperm(len(y), generator=torch.Generator().manual_seed(1))[:20000]
    rec = dict(exp="04", **vars(a), n_train=len(y), n_test=len(yt))
    if a.method in ("pixel", "hartley"):
        m = PlaneNet(n, V + 2, V, a.stages, a.width, a.seed, a.method).to(dev)
        train_s = e3.fit(m, X, y, a.steps, dev, a.seed)
        m.eval()
        LP = e3.logp_of(m, Xt, dev)
        acc, ce = e3.evals(LP, yt)
        tr_acc, _ = e3.evals(e3.logp_of(m, X[ti], dev), y[ti])
        light, dig = m.params()
        rec.update(test_acc=acc, test_ce=ce, train_acc=tr_acc, params_light=light, params_digital_learned=dig, table_stored=0, train_s=train_s)
    elif a.method in ("fixed", "proto"):
        m = textlight.Text4f(n, V + 2, V, a.K, a.stages, a.width, a.seed).to(dev)
        m.codes.requires_grad_(False)
        if a.method == "proto":
            S = torch.zeros(V, a.K, device=dev)
            cnt = torch.zeros(V, device=dev)
            opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=0.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
            g = torch.Generator().manual_seed(a.seed)
            t1 = time.perf_counter()
            for k in range(a.steps):
                i = torch.randint(len(y), (256,), generator=g)
                xb, yb = X[i].to(dev), y[i].to(dev)
                w = p2.read_windows(m.body.screen(m.field(xb)), m.det)
                loss = F.cross_entropy(p2.code_scores(w, m.codes, 1.0) * m.temp, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
                with torch.no_grad():                   # 答えごとの窓の模様の平均を貯め、300 ステップごとに符号をそろえ直す
                    S.index_add_(0, yb, p2.unit_rows(w.detach()))
                    cnt.index_add_(0, yb, torch.ones_like(yb, dtype=torch.float))
                    if (k + 1) % 300 == 0:
                        seen = cnt > 0
                        m.codes[seen] = S[seen] / cnt[seen, None]
                        S *= 0.5
                        cnt *= 0.5
                if (k + 1) % 1000 == 0:
                    print(f"{k + 1}/{a.steps} 損失 {loss.item():.3f} {time.perf_counter() - t1:.0f} 秒", flush=True)
            train_s = time.perf_counter() - t1
        else:
            train_s = e3.fit(m, X, y, a.steps, dev, a.seed)
        m.eval()
        LP = e3.logp_of(m, Xt, dev)
        acc, ce = e3.evals(LP, yt)
        tr_acc, _ = e3.evals(e3.logp_of(m, X[ti], dev), y[ti])
        light, _ = m.params()
        rec.update(test_acc=acc, test_ce=ce, train_acc=tr_acc, params_light=light, params_digital_learned=1, table_stored=m.codes.numel(), train_s=train_s)
    else:
        grp, idx = grouping(a.method.split("_")[1], X, y, V, a.seed)
        rec["group_sizes_max"] = int(torch.bincount(grp).max())
        # 光路 A: 文脈 → 組（窓 64 個）
        mA = WindowNet(n, V + 2, NG, a.stages, a.width, a.seed).to(dev)
        tA = e3.fit(mA, X, grp[y], a.steps, dev, a.seed)
        # 光路 B: 文脈 + 組（入口の 1 本の帯。組の番号は V+2.. の単位として入れる）→ 組の中の番号（窓 64 個）
        mB = WindowNet(n + 1, V + 2 + NG, NG, a.stages, a.width, a.seed + 1).to(dev)
        withg = lambda Z, gg: torch.cat([(V + 2 + gg)[:, None], Z], 1)
        tB = e3.fit(mB, withg(X, grp[y]), idx[y], a.steps, dev, a.seed + 1)
        mA.eval()
        mB.eval()

        def full_logp(Z):
            """全部の答えの対数確率 (N, V) = log P(組) + log P(番号 | 組)（64 組すべてで光路 B を通す）。"""
            LA = e3.logp_of(mA, Z, dev)                                     # (N, 64)
            out = torch.empty(len(Z), V)
            for gi in range(NG):
                LB = e3.logp_of(mB, withg(Z, torch.full((len(Z),), gi)), dev)   # (N, 64)
                members = (grp == gi).nonzero().squeeze(1)
                out[:, members] = LA[:, gi:gi + 1] + LB[:, idx[members]]
            return out
        LP = full_logp(Xt)
        acc, ce = e3.evals(LP, yt)
        tr_acc, _ = e3.evals(full_logp(X[ti]), y[ti])
        accA, _ = e3.evals(e3.logp_of(mA, Xt, dev), grp[yt])
        accB, _ = e3.evals(e3.logp_of(mB, withg(Xt, grp[yt]), dev), idx[yt])
        la, da = mA.params()
        lb, db = mB.params()
        rec.update(test_acc=acc, test_ce=ce, train_acc=tr_acc, group_acc=accA, index_acc_given_group=accB,
                   params_light=la + lb, params_digital_learned=da + db, table_stored=0, train_s=tA + tB)
    rec.update(total_s=time.perf_counter() - t0, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
