"""介入の研究 実験I27n: 似方を関わりの種類ごとに分けて測り、どの関わりが係り先の手がかりになるかを見る（試験の一覧 B1 の続き）

I27l の 4 回目は、似方を「同じ連体の修飾を受けたか」1 種類だけで測った。ここでは名詞の関わりを種類ごとの板に分ける:
  受けた修飾（acl / amod の M'）・する（主語になった述語）・される（目的語になった述語）・で・に（obl の述語）・の（「X の 名詞」の X）。
  種類ごとに似方（余弦。問いの M と試しの行の分は除く）を測り、候補に似た名詞に M が係った数 × 似方 × λ_種類 を根拠に足す。
作り方は I27l と同じ（make_items を使う。Web 小説は _wn、非公開の置き場だけ）。
速くするための形: 読む幅 = 全部のときは、波の答えは根拠の合計だけで決まる（I27k の read_wave で、最後のいちばん明るい仮説）。
  そこで問いごとに「同じ組の数」「種類ごとの広げた根拠」「ほかの作品の同じ組」を 1 回だけ数え、λ の組み合わせを手早く回す。
λ の決め方: 作品を偶数番・奇数番の 2 組に分け、片方で λ の組み合わせを選び、もう片方で測る（入れ替えて 2 回、合わせて報告）。
実行: python intervention/i27n_relation_sim.py --cache <解析済み JSON> --tag wn --out <jsonl> [--jobs 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27l_wave_scale as L  # noqa: E402

NOUN = ("NOUN", "PROPN")
TYPES = ("受けた修飾", "する", "される", "で・に", "の")


def line_rels(toks):
    """行の中の (名詞の見出し, 種類, 相手の見出し)。"""
    out = []
    for i, w in enumerate(toks):
        h = w["head"]
        if h == i or not 0 <= h < len(toks):
            continue
        hw = toks[h]
        if w["pos"] in ("ADJ", "VERB") and w["dep"] in ("acl", "amod") and hw["pos"] in NOUN:
            out.append((hw["lemma"], "受けた修飾", w["lemma"]))
        if w["pos"] in NOUN and w["dep"] in ("nsubj", "obj", "obl") and hw["pos"] in ("VERB", "ADJ"):
            out.append((w["lemma"], {"nsubj": "する", "obj": "される", "obl": "で・に"}[w["dep"]], hw["lemma"]))
        if w["pos"] in NOUN and w["dep"] == "nmod" and hw["pos"] in NOUN:
            out.append((hw["lemma"], "の", w["lemma"]))
    return out


def prep_rel(works):
    for w in works:
        w["_rel"] = {t: defaultdict(Counter) for t in TYPES}
        w["_rel_line"] = []
        for ln in w["lines"]:
            own = Counter()
            for n, t, k in line_rels(ln["toks"]):
                w["_rel"][t][n][k] += 1
                own[(n, t, k)] += 1
            w["_rel_line"].append(own)


def sim_rel(w, li, a, b, t, m):
    """種類 t だけの似方。相手が問いの M のもの（M に係られた・M の主語になった など）と、試しの行の分は除く。"""
    own = w["_rel_line"][li]
    va, vb = w["_rel"][t].get(a, {}), w["_rel"][t].get(b, {})
    if not va or not vb:
        return 0.0

    def g(v, wd, k):
        return v.get(k, 0) - own.get((wd, t, k), 0)
    dot = sum(g(va, a, k) * g(vb, b, k) for k in va if k != m and k in vb)
    if dot <= 0:
        return 0.0
    na = sum(g(va, a, k) ** 2 for k in va if k != m) ** 0.5
    nb = sum(g(vb, b, k) ** 2 for k in vb if k != m) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def totals(works, x):
    """問い 1 つの根拠の合計: 同じ組（2）、種類ごとの広げた根拠（種類 × 2）、ほかの作品の同じ組（2）。"""
    w = works[x["work"]]
    li = x["line"]
    key = "修飾:" + x["m"]
    own = w["_boards"][li]
    cnt = {}                                                    # 名詞 → ほかの行で M が係った数
    for n, c in w["_vec"].items():
        v = c.get(key, 0) - own.get(n, {}).get(key, 0)
        if v > 0:
            cnt[n] = v
    same = [cnt.get(c, 0) for c in x["cand"]]
    spread = {}
    for t in TYPES:
        row = []
        for c in x["cand"]:
            s = 0.0
            for n, v in cnt.items():
                if n in x["cand"]:
                    continue
                sm = sim_rel(w, li, c, n, t, x["m"])
                if sm > 0:
                    s += sm * v
            row.append(s)
        spread[t] = row
    cross = L.cross_row(works, x, 1.0, 0.0)
    return same, spread, cross


_W = {}


def _tot(k):
    return totals(_W["works"], _W["items"][k])


def answer(tt, lam, mu, eps=0.3):
    same, spread, cross = tt
    amp = [eps + same[0], same[1]]
    got = any(same)
    for t, l in lam.items():
        if l:
            for k in range(2):
                amp[k] += l * spread[t][k]
            got = got or any(spread[t])
    if mu:
        for k in range(2):
            amp[k] += mu * cross[k]
        got = got or any(cross)
    return (0 if amp[0] ** 2 >= amp[1] ** 2 else 1), not got


def score(items, tots, ks, lam, mu):
    groups = defaultdict(list)
    for k in ks:
        groups[items[k]["distract"]].append(k)
    r = {}
    for g, gk in sorted(groups.items()):
        ok = sil = sil_ok = 0
        for k in gk:
            x = items[k]
            a, s = answer(tots[k], lam, mu)
            hit = x["cand"][a] == x["gold"]
            ok += hit
            sil += s
            sil_ok += s and hit
        n = len(gk)
        r[g] = dict(n=n, 波=round(ok / n, 4), 静か=round(sil / n, 4),
                    静か以外=round((ok - sil_ok) / max(1, n - sil), 4))
    r["平均"] = round(sum(v["波"] for v in r.values()) / len(r), 4)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", default="wn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_work", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    items = L.make_items(works, a.seed, a.per_work)
    prep_rel(works)
    L.SHARED = L.shared(works)
    print("items", len(items), flush=True)
    _W.update(works=works, items=items)
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.jobs) as pool:
            tots = pool.map(_tot, range(len(items)), chunksize=max(1, len(items) // (a.jobs * 20)))
    else:
        tots = [_tot(k) for k in range(len(items))]
    allk = list(range(len(items)))
    rec = dict(exp="I27n", tag=a.tag, works=len(works), lines=sum(len(w["lines"]) for w in works), n=len(items), fixed={}, learned={})
    zero = {t: 0 for t in TYPES}
    rec["fixed"]["同じ組だけ"] = score(items, tots, allk, zero, 0)
    rec["fixed"]["同じ組だけ + ほかの作品 μ1"] = score(items, tots, allk, zero, 1.0)
    for t in TYPES:
        for lam in (1.0, 3.0):
            rec["fixed"][f"{t} λ{lam:g}"] = score(items, tots, allk, {**zero, t: lam}, 0)
            rec["fixed"][f"{t} λ{lam:g} + ほかの作品 μ0.3"] = score(items, tots, allk, {**zero, t: lam}, 0.3)
    rec["fixed"]["全種類 λ1"] = score(items, tots, allk, {t: 1.0 for t in TYPES}, 0)
    rec["fixed"]["全種類 λ1 + ほかの作品 μ0.3"] = score(items, tots, allk, {t: 1.0 for t in TYPES}, 0.3)
    # 片方の作品で λ を選び、もう片方で測る
    grid = (0.0, 1.0, 3.0)
    halves = [[k for k in allk if items[k]["work"] % 2 == p] for p in (0, 1)]
    picked = []
    for p in (0, 1):
        tr, te = halves[p], halves[1 - p]
        best = None
        for combo in itertools.product(grid, repeat=len(TYPES)):
            for mu in (0.0, 0.3, 1.0):
                lam = dict(zip(TYPES, combo))
                s = score(items, tots, tr, lam, mu)["平均"]
                if best is None or s > best[0]:
                    best = (s, lam, mu)
        picked.append(dict(学んだ組=("偶数" if p == 0 else "奇数"), λ=best[1], μ=best[2], 学んだ組の平均=best[0],
                           測った組=score(items, tots, te, best[1], best[2])))
        print("picked", picked[-1], flush=True)
    rec["learned"] = picked
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
