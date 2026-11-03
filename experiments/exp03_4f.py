"""実験03: 4f 系 ＋ 光の非線形（平方根）で次の言葉を当てる — 光路 1 本・分担の 2DTG・足し合わせ（現代の雑談）

着想: 持ち主の案（2026-10-27「4f 系の非線形ならいい結果になるかも」）。実験02・02c で、板を重ねる分担の 2DTG は GRU に 3.3〜4.6 点負け、
      訓練の正解率もテストと同じくらい（25.5%）で、覚えきれていなかった（容量が足りない）。SN2-Parts-2D では、4f 系 + 平方根の光路が
      単語の課題で 3-gram 並み（36.1%）、性格の違う光路の足し合わせで GRU を超えた（37.4%、SN2 実験84・85）。
方法（どれも textlight.py。SN2 の Word4f・Multi4f・拡散の入れ方の書き直し）:
  f4_single : 光路 1 本。拡散（位置ごとの模様を掛けて面全体に重ねる）→ 4f 系 6 段 × 幅 8 + 平方根 → レンズ → 窓 256 個 → 符号読み。9000 ステップ
  f4_split  : 分担の 2DTG（担当 4、直前の単位の番号の余りで決め打ち）。中身は 4f 系 4 段 × 幅 4 + 平方根。各 9000 ステップ
  fuse_light: f4_single と f4_split の対数確率の重み付き和（光だけの足し合わせ）
  fuse_tri  : 上に 3-gram を足した足し合わせ（光 + 数えた表。性格の違うものを足すと伸びる、SN2 実験83）
  重みはテストを 2 つに分け、前半で選んで後半で測る・後半で選んで前半で測る（SN2 の exp81_fuse.py と同じ。テスト全部の値になる）。
  --tag で、入力の先頭に口調（丁寧・くだけた）の帯を足す（口調のスイッチ、実験02 と同じ）。3-gram は口調ごとの表になる。
データ・テスト: 実験02 と同じ（現代の雑談、知らない話者 51 人、テスト 3 万位置。SentencePiece 4096、直前 4 単位）。
比べる（実験02・02c、同じテスト）: 板の分担の 2DTG 25.03%（9000 ステップ）、GRU 28.33%、3-gram 30.88%。
指標: 正解率・CE・訓練の正解率（2 万位置）、--tag では口調の当たり（f4_single と fuse_light で文を作る）。学習する数は光とデジタルに分ける。
結果: results/exp03.txt、例 ../Project_Kotoyui_private/docs/samples/m1_exp03_*.md
実行: python experiments/exp03_4f.py --tag --seed 0 --device cuda --out results/exp03.txt
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import est_flavor_counts as efc  # noqa: E402
import est_modern  # noqa: E402
import exp02_split_style as e2  # noqa: E402
import textlight  # noqa: E402


def fit(model, X, y, steps, dev, seed):
    """光路 1 本を学習する（Adam、学習率 0.01 → 0、256 件ずつ）。"""
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for k in range(steps):
        i = torch.randint(len(y), (256,), generator=g)
        loss = F.cross_entropy(model(X[i].to(dev)), y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if (k + 1) % 1000 == 0:
            print(f"光路 1 本 {k + 1}/{steps} 損失 {loss.item():.3f} {time.perf_counter() - t0:.0f} 秒", flush=True)
    return time.perf_counter() - t0


@torch.no_grad()
def logp_of(fn, Z, dev, bs=1000):
    return torch.cat([F.log_softmax(fn(Z[i:i + bs].to(dev)).float(), 1).cpu() for i in range(0, len(Z), bs)])


def evals(LP, y):
    return 100.0 * (LP.argmax(1) == y).float().mean().item(), -LP[torch.arange(len(y)), y].mean().item()


def fuse(parts, y, grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5)):
    """対数確率の重み付き和 → 正規化。重みは半分で選び、もう半分で測る（入れ替えて両方）。戻り値: 正解率・CE・選んだ重み 2 組。"""
    n = len(y)
    halves = [torch.arange(0, n // 2), torch.arange(n // 2, n)]
    pred = torch.empty(n, dtype=torch.long)
    nll = torch.empty(n)
    chosen = []
    for a, b in ((0, 1), (1, 0)):
        ia, ib = halves[a], halves[b]
        best = None
        for w in itertools.product(grid, repeat=len(parts)):
            if sum(w) == 0:
                continue
            lp = F.log_softmax(sum(wi * P[ia] for wi, P in zip(w, parts)), 1)
            ce = -lp[torch.arange(len(ia)), y[ia]].mean().item()
            if best is None or ce < best[0]:
                best = (ce, w)
        w = best[1]
        chosen.append(w)
        lp = F.log_softmax(sum(wi * P[ib] for wi, P in zip(w, parts)), 1)
        pred[ib] = lp.argmax(1)
        nll[ib] = -lp[torch.arange(len(ib)), y[ib]]
    return 100.0 * (pred == y).float().mean().item(), nll.mean().item(), chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="store_true", help="口調のスイッチ（入力の先頭に口調の帯）")
    ap.add_argument("--stages", type=int, default=6)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--split_stages", type=int, default=4)
    ap.add_argument("--split_width", type=int, default=4)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--max_train", type=int, default=2000000)
    ap.add_argument("--n_style", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp03.txt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    (X, y, s), (Xt, yt, st), sp, tr_u = e2.build(a.max_train, a.seed)
    V = sp.get_piece_size()
    n = e2.N_CTX + (1 if a.tag else 0)
    Xin = e2.with_tag(X, s, V) if a.tag else X
    Xtin = e2.with_tag(Xt, st, V) if a.tag else Xt
    ti = torch.randperm(len(y), generator=torch.Generator().manual_seed(1))[:20000]
    print(f"訓練 {len(y)} 位置・テスト {len(yt)} 位置（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    rec = dict(exp="03", **vars(a), n_train=len(y), n_test=len(yt), version=textlight.p2.__version__, torch=torch.__version__)
    parts, models = [], {}
    # 1. 光路 1 本
    m1 = textlight.Text4f(n, V + 2, V, a.K, a.stages, a.width, a.seed).to(dev)
    t1 = fit(m1, Xin, y, a.steps, dev, a.seed)
    m1.eval()
    LP1 = logp_of(m1, Xtin, dev)
    acc, ce = evals(LP1, yt)
    tr_acc, _ = evals(logp_of(m1, Xin[ti], dev), y[ti])
    rec["f4_single"] = dict(acc=acc, ce=ce, train_acc=tr_acc, params_light=m1.params()[0], params_digital=m1.params()[1], train_s=t1)
    print("光路 1 本", rec["f4_single"], flush=True)
    parts.append(LP1)
    models["f4_single"] = lambda Z: m1(Z)
    # 2. 分担の 2DTG（中身は 4f 系）
    m4 = textlight.Split4f(n, V + 2, V, a.G, a.K, a.split_stages, a.split_width, a.seed).to(dev)
    t4 = textlight.train_split(m4, Xin, y, a.steps, seed=a.seed, dev=dev, log=lambda msg: print(msg, flush=True))
    m4.eval()
    LP4 = logp_of(m4.logits, Xtin, dev)
    acc, ce = evals(LP4, yt)
    tr_acc, _ = evals(logp_of(m4.logits, Xin[ti], dev), y[ti])
    lp4, dg4 = m4.params()
    rec["f4_split"] = dict(acc=acc, ce=ce, train_acc=tr_acc, params_light=lp4, params_digital=dg4, train_s=t4)
    print("分担の 2DTG", rec["f4_split"], flush=True)
    parts.append(LP4)
    models["f4_split"] = lambda Z: m4.logits(Z)
    # 3. 3-gram（比べる・足し合わせの相手。--tag なら口調ごとの表）
    tabs = {None: efc.Tri(V)}
    if a.tag:
        tabs.update({k: efc.Tri(V) for k in range(len(e2.STYLES))})
    for u in tr_u:
        seq = [1, 1] + sp.encode(u["text"]) + [2]
        tabs[None].add(seq)
        if a.tag:
            tabs[e2.STYLES.index(u["polite"])].add(seq)
    for t in tabs.values():
        t.finish()
    LPT = torch.stack([torch.from_numpy(np.log(np.maximum(
        (tabs[int(st[i])] if a.tag else tabs[None]).prob(int(Xt[i, -2]), int(Xt[i, -1])), 1e-12))).float() for i in range(len(yt))])
    acc, ce = evals(LPT, yt)
    rec["tri"] = dict(acc=acc, ce=ce)
    # 4. 足し合わせ
    acc, ce, w = fuse(parts, yt)
    rec["fuse_light"] = dict(acc=acc, ce=ce, weights=w)
    acc, ce, w = fuse(parts + [LPT], yt)
    rec["fuse_tri"] = dict(acc=acc, ce=ce, weights=w)
    print("足し合わせ", rec["fuse_light"], rec["fuse_tri"], flush=True)
    # 5. 口調の当たり（--tag のとき）: 光路 1 本と、光だけの足し合わせ（重みは前半で選んだもの）で文を作る
    if a.tag:
        wl = rec["fuse_light"]["weights"][0]
        gens = {"f4_single": [(1.0, models["f4_single"])],
                "fuse_light": [(wl[0], models["f4_single"]), (wl[1], models["f4_split"])]}
        rng = np.random.default_rng(a.seed)
        for name, comps in gens.items():
            per, closed, lines = {}, 0, [f"# 実験03 の例: {name}（口調のスイッチあり、seed {a.seed}、口調ごとに先頭の 8 文。選び直していない）", "",
                                          "現代の雑談の訓練の発言で学んだ光の部品が作った文（データの文そのものではない）。判定は文末の です・ます。", ""]
            for si, sty in enumerate(e2.STYLES):
                hit = 0
                lines += [f"## 決めた口調: {sty}", ""]
                for k in range(a.n_style):
                    def step_probs(ctx, style, si=si):
                        Z = e2.with_tag(torch.tensor([ctx]), torch.tensor([si]), V).to(dev)
                        with torch.no_grad():
                            lp = F.log_softmax(sum(w * F.log_softmax(f(Z).float(), 1) for w, f in comps), 1)[0]
                        return np.exp(lp.cpu().numpy().astype(np.float64))
                    ids, ok = e2.generate(step_probs, si, rng)
                    text = sp.decode(ids)
                    judged = "丁寧" if est_modern._POLITE.search(text) else "くだけた"
                    hit += judged == sty
                    closed += ok
                    if k < 8:
                        lines.append(f"- 「{text}」{'' if ok else '（閉じず）'} — 判定: {judged}{' ✓' if judged == sty else ''}")
                per[sty] = 100.0 * hit / a.n_style
                lines.append("")
            rec[f"style_{name}"] = dict(per=per, avg=sum(per.values()) / len(per), closed=100.0 * closed / (a.n_style * 2))
            if a.seed == 0:
                with open(f"../Project_Kotoyui_private/docs/samples/m1_exp03_{name}.md", "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
    rec["total_s"] = time.perf_counter() - t0
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
