"""実験02: 光の分担の 2DTG を現代の雑談で学び、口調のスイッチ（丁寧・くだけた）を入れて文を作る — M1 の土台

着想: 持ち主の判断（2026-10-26）で M1 を「光の 2DTG ＋ 口調のスイッチ ＋ 2DWIG」に組み替えた。見積もり E6 で、
      丁寧さを決めて作ると口調の当たりが 52 → 93%（数えた表）。光の分担の 2DTG（SN2-Parts-2D 実験70 の最良の形）で同じことができるか。
データ: 現代の雑談（RealPersonaChat・名大会話コーパス）。知らない話者 51 人の対話をテスト（est_modern.split と同じ）。
        単位は SentencePiece 4096（見積もり E6 で訓練の発言だけで学んだもの）。会話文の中で直前 4 単位 → 次の 1 単位（終わり = 」も当てる）。
方法（--method）:
  split     : 光の分担の 2DTG（直前の単位の番号の余りで 16 個、各 BandNet = 板 ① 8 枚 × 区画 16 + 窓 32 の符号読み、担当の決め打ち）
  split_tag : 同じ。入力の先頭にもう 1 本の帯を足し、口調（丁寧・くだけた）を決まった向きの平面波で入れる（口調のスイッチ。計算はほぼ同じ）
  gru / gru_tag : 比べる相手。GRU（埋め込み 128・隠れ 256）。_tag は口調を入力の先頭の単位として足す
  tri / tri_tag : 比べる相手。後退つき 3-gram（Witten-Bell）。_tag は口調ごとの表で作る（見積もり E6 と同じ）
指標: 次の単位の正解率・CE（テストの話者の発言から 3 万位置）、口調の当たり（口調を決めて 100 文 × 2 口調を作り、文末で丁寧さを判定、
      判定は答えの定義そのもの = est_modern._POLITE）、」で閉じた割合、同じ単位の繰り返し。作った文は ../Project_Kotoyui_private/docs/samples/m1_exp02_<method>.md。
      学習する数は光の部分とデジタルの部分（符号）に分けて数える。
結果: results/exp02.txt
実行: python experiments/exp02_split_style.py --method split_tag --seed 0 --device cuda --out results/exp02.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import est_flavor_counts as efc  # noqa: E402
import est_modern  # noqa: E402
import textlight  # noqa: E402

N_CTX = 4
STYLES = ["丁寧", "くだけた"]


def build(max_train, seed):
    """位置ごとの (直前 4 単位, 次の単位, 口調) の訓練・テストと、SentencePiece。"""
    us = est_modern.load()
    tr_d, _, te_d, test_spk = est_modern.split(us, 0)
    tr_u = [u for u in us if (u["src"], u["dialog"]) in tr_d]
    te_u = [u for u in us if (u["src"], u["dialog"]) in te_d and u["spk"] in test_spk]
    sp = efc.train_spm([u["text"] for u in tr_u], 4096, "modern")

    def pos(uu):
        X, y, s = [], [], []
        for u in uu:
            seq = [1] * N_CTX + sp.encode(u["text"]) + [2]
            st = STYLES.index(u["polite"])
            for t in range(N_CTX, len(seq)):
                X.append(seq[t - N_CTX:t])
                y.append(seq[t])
                s.append(st)
        return torch.tensor(X, dtype=torch.long), torch.tensor(y), torch.tensor(s)
    X, y, s = pos(tr_u)
    Xt, yt, st = pos(te_u)
    rng = torch.Generator().manual_seed(seed)
    if len(y) > max_train:
        i = torch.randperm(len(y), generator=rng)[:max_train]
        X, y, s = X[i], y[i], s[i]
    i = torch.randperm(len(yt), generator=torch.Generator().manual_seed(0))[:30000]   # テストは seed によらず同じ
    return (X, y, s), (Xt[i], yt[i], st[i]), sp, tr_u


def with_tag(X, s, V):
    """入力の先頭に口調の単位（番号 V + 口調）を足す。"""
    return torch.cat([(V + s)[:, None], X], 1)


class GRU(torch.nn.Module):
    def __init__(self, V, C, tag=False):
        super().__init__()
        self.emb = torch.nn.Embedding(V + (2 if tag else 0), 128)
        self.rnn = torch.nn.GRU(128, 256, batch_first=True)
        self.out = torch.nn.Linear(256, C)

    def forward(self, X):
        return self.out(self.rnn(self.emb(X))[0][:, -1])


def generate(step_probs, style, rng, topk=20, max_len=40):
    """1 単位ずつ選んで 1 文を作る（est_flavor_style.generate と同じ選び方）。step_probs(直前 4 単位, 口調) → 確率 (V,)。"""
    seq = [1] * N_CTX
    for _ in range(max_len):
        p = step_probs(seq[-N_CTX:], style).copy()
        p[[0, 1]] = 0
        if seq[-1] == seq[-2] and seq[-1] > 2:
            p[seq[-1]] = 0
        top = np.argsort(-p)[:topk]
        y = int(rng.choice(top, p=p[top] / p[top].sum()))
        if y == 2:
            return seq[N_CTX:], True
        seq.append(y)
    return seq[N_CTX:], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="split_tag", choices=["split", "split_tag", "gru", "gru_tag", "tri", "tri_tag"])
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max_train", type=int, default=2000000)
    ap.add_argument("--n_style", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp02.txt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    (X, y, s), (Xt, yt, st), sp, tr_u = build(a.max_train, a.seed)
    V = sp.get_piece_size()
    tag = a.method.endswith("_tag")
    print(f"訓練 {len(y)} 位置・テスト {len(yt)} 位置（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    light = digital = 0
    t1 = time.perf_counter()
    if a.method.startswith("split"):
        n = N_CTX + (1 if tag else 0)
        model = textlight.Split2DTG(n, V + 2, V, a.G, a.layers, a.patches, a.K, a.seed).to(dev)
        Xin = with_tag(X, s, V) if tag else X
        textlight.train_split(model, Xin, y, a.steps, seed=a.seed, dev=dev, log=lambda m: print(m, flush=True))
        light, digital = model.params()
        model.eval()

        @torch.no_grad()
        def logp_batch(XX, ss):
            Z = with_tag(XX, ss, V) if tag else XX
            return torch.cat([F.log_softmax(model.logits(Z[i:i + 2000].to(dev)), 1).cpu() for i in range(0, len(Z), 2000)])
    elif a.method.startswith("gru"):
        model = GRU(V, V, tag).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps * 4)
        g = torch.Generator().manual_seed(a.seed)
        Xin = with_tag(X, s, V) if tag else X
        for _ in range(a.steps * 4):
            i = torch.randint(len(y), (256,), generator=g)
            loss = F.cross_entropy(model(Xin[i].to(dev)), y[i].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        digital = sum(p.numel() for p in model.parameters())
        model.eval()

        @torch.no_grad()
        def logp_batch(XX, ss):
            Z = with_tag(XX, ss, V) if tag else XX
            return torch.cat([F.log_softmax(model(Z[i:i + 2000].to(dev)), 1).cpu() for i in range(0, len(Z), 2000)])
    else:
        tabs = {None: efc.Tri(V)}
        if tag:
            tabs.update({k: efc.Tri(V) for k in range(len(STYLES))})
        for u in tr_u:
            seq = [1, 1] + sp.encode(u["text"]) + [2]
            tabs[None].add(seq)
            if tag:
                tabs[STYLES.index(u["polite"])].add(seq)
        for t in tabs.values():
            t.finish()

        def logp_batch(XX, ss):
            out = torch.empty(len(XX), V)
            for i in range(len(XX)):
                tb = tabs[int(ss[i])] if tag else tabs[None]
                out[i] = torch.from_numpy(np.log(np.maximum(tb.prob(int(XX[i, -2]), int(XX[i, -1])), 1e-12)))
            return out
    train_s = time.perf_counter() - t1
    LP = logp_batch(Xt, st)
    acc = 100.0 * (LP.argmax(1) == yt).float().mean().item()
    # 訓練の正解率（2 万位置。覚えすぎ・覚えきれないの目安）
    ti = torch.randperm(len(y), generator=torch.Generator().manual_seed(1))[:20000]
    train_acc = 100.0 * (logp_batch(X[ti], s[ti]).argmax(1) == y[ti]).float().mean().item()
    ce = -LP[torch.arange(len(yt)), yt].mean().item()
    # 口調の当たり: 口調を決めて文を作り、文末で丁寧さを判定する
    res_style, lines = {}, [f"# 実験02 の例: {a.method}（seed {a.seed}、口調ごとに先頭の 8 文。選び直していない）", "",
                            "現代の雑談の訓練の発言で学んだモデルが作った文（データの文そのものではない）。判定は文末の です・ます。", ""]
    rng = np.random.default_rng(a.seed)
    closed = reps = total = 0
    for si, sty in enumerate(STYLES):
        hit = 0
        lines += [f"## 決めた口調: {sty}", ""]
        for k in range(a.n_style):
            def step_probs(ctx, style, si=si):
                lp = logp_batch(torch.tensor([ctx]), torch.tensor([si]))[0]
                return np.exp(lp.numpy().astype(np.float64))
            ids, ok = generate(step_probs, si, rng)
            text = sp.decode(ids)
            judged = "丁寧" if est_modern._POLITE.search(text) else "くだけた"
            hit += judged == sty
            closed += ok
            reps += (len(ids) - len(set(ids))) / max(1, len(ids))
            total += 1
            if k < 8:
                lines.append(f"- 「{text}」{'' if ok else '（閉じず）'} — 判定: {judged}{' ✓' if judged == sty else ''}")
        res_style[sty] = 100.0 * hit / a.n_style
        lines.append("")
    os.makedirs("docs/samples", exist_ok=True)
    if a.seed == 0:
        with open(f"../Project_Kotoyui_private/docs/samples/m1_exp02_{a.method}.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    rec = dict(exp="02", **vars(a), n_train=len(y), n_test=len(yt), test_acc=acc, test_ce=ce, train_acc=train_acc,
               style_acc=sum(res_style.values()) / len(res_style), style_per=res_style,
               closed=100.0 * closed / total, repeat=100.0 * reps / total,
               params_light=light, params_digital=digital, train_s=train_s, total_s=time.perf_counter() - t0,
               version=textlight.p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
