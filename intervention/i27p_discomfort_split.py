"""介入の研究 実験I27p: 違和感で語板を分ける（試験の一覧 B の残り。I1 の形の文章版）

I27l の 3 回目は、関連の多い語板（よく出る語）を前後の品詞で別の板に分けたが、ほとんど変わらなかった（分ける語を数で選んだ）。
ここでは分ける語を「違和感」で選ぶ:
  - 違和感の測り方（自分で作った問いだけを使う。答えは文から作れる）: I27l と同じ作り方で、試しとは別の seed の問いを作る（試しの問いと同じ行は使わない）。
    似た語へ広げるとき（受けた修飾の似方、I27n）、名詞 n から来た票が、正解の候補と紛らわしい候補のどちらに多く入ったかを数える。
    違和感(n) = 紛らわしい方に入った票 / 票の合計（票が 3 以上の名詞だけ）。
  - 分け方: 違和感の高い名詞の板を、行の中の前後（前の語の品詞 > 後ろの語。助詞・助動詞なら字）で別の板にする（I27l の 3 回目と同じ手がかり）。
    分けた板は、似方も M の数えも、分けた単位で持つ。候補（問いの H・D）は分けない。
  - 比べ: 分けない / 数で選んで分ける（同じ語数）/ 違和感で選んで分ける / 違和感の高い語を広げから外す。
読む幅 = 全部、作品の中だけの広げ（受けた修飾 λ3）と、I27n で学んだ組み合わせ（受けた修飾 3・する 1・で・に 1 + ほかの作品 μ1）の 2 通りで測る。
Web 小説は _wn、結果は非公開の置き場だけ。
実行: python intervention/i27p_discomfort_split.py --cache <解析済み JSON> --out <jsonl> [--jobs 3]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27l_wave_scale as L  # noqa: E402
import i27n_relation_sim as N  # noqa: E402

NOUN = ("NOUN", "PROPN")
TYPES = N.TYPES


def ctx(toks, i):
    pv = toks[i - 1]["pos"] if i > 0 else "頭"
    nx = toks[i + 1] if i + 1 < len(toks) else None
    nv = "尾" if nx is None else (nx["text"] if nx["pos"] in ("ADP", "AUX", "SCONJ", "PART") else nx["pos"])
    return f"{pv}>{nv}"


def line_facts(toks):
    """行の中の (名詞の位置, 種類, 相手) と、(名詞の位置, 係った M) の 2 つ（I27a の話の板の 修飾: と同じ決まり。見出しだけ）。"""
    rels, mods = [], []
    for i, w in enumerate(toks):
        h = w["head"]
        if h == i or not 0 <= h < len(toks):
            continue
        hw = toks[h]
        if w["pos"] in ("ADJ", "VERB") and w["dep"] in ("acl", "amod") and hw["pos"] in NOUN:
            rels.append((h, "受けた修飾", w["lemma"]))
            mods.append((h, w["lemma"]))
        if w["pos"] in NOUN and w["dep"] in ("nsubj", "obj", "obl") and hw["pos"] in ("VERB", "ADJ"):
            rels.append((i, {"nsubj": "する", "obj": "される", "obl": "で・に"}[w["dep"]], hw["lemma"]))
            mods.append((i, hw["lemma"]))
        if w["pos"] in NOUN and w["dep"] == "nmod" and hw["pos"] in NOUN:
            rels.append((h, "の", w["lemma"]))
            mods.append((h, w["lemma"]))
    return rels, mods


def build(works, split_by_work):
    """作品ごとに、(分けた) 板: 名詞の鍵 → 種類 → 相手の数え、名詞の鍵 → M の数え。行ごとの分も持つ（試しの行を引くため）。"""
    for wi, w in enumerate(works):
        sp = split_by_work.get(wi, set())
        rel = {t: defaultdict(Counter) for t in TYPES}
        mc = defaultdict(Counter)
        rel_line, mc_line = [], []
        for ln in w["lines"]:
            toks = ln["toks"]
            rels, mods = line_facts(toks)

            def key(i):
                k = toks[i]["lemma"]
                return f"{k}|{ctx(toks, i)}" if k in sp else k
            ro, mo = Counter(), Counter()
            for i, t, k in rels:
                rel[t][key(i)][k] += 1
                ro[(key(i), t, k)] += 1
            for i, m in mods:
                mc[key(i)][m] += 1
                mo[(key(i), m)] += 1
            rel_line.append(ro)
            mc_line.append(mo)
        # 候補（分けない語）用の、分けない板も持つ
        rel0 = {t: defaultdict(Counter) for t in TYPES}
        for t in TYPES:
            for k, c in rel[t].items():
                rel0[t][k.split("|")[0]].update(c)
        w["_p"] = dict(rel=rel, rel0=rel0, mc=mc, rel_line=rel_line, mc_line=mc_line)


def sim(w, li, a, b, t, m):
    """候補 a（分けない板）と名詞の鍵 b（分けた板かもしれない）の、種類 t の似方。"""
    P = w["_p"]
    own = P["rel_line"][li]
    va, vb = P["rel0"][t].get(a, {}), P["rel"][t].get(b, {})
    if not va or not vb:
        return 0.0
    own_a = Counter()
    for (k, tt, x), v in own.items():
        if tt == t and k.split("|")[0] == a:
            own_a[x] += v

    def ga(k):
        return va.get(k, 0) - own_a.get(k, 0)

    def gb(k):
        return vb.get(k, 0) - own.get((b, t, k), 0)
    dot = sum(ga(k) * gb(k) for k in va if k != m and k in vb)
    if dot <= 0:
        return 0.0
    na = sum(ga(k) ** 2 for k in va if k != m) ** 0.5
    nb = sum(gb(k) ** 2 for k in vb if k != m) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def totals(works, x, drop=frozenset(), votes=None):
    """I27n の totals と同じ形（同じ組・種類ごとの広げ・ほかの作品）。drop の名詞は広げに使わない。votes があれば、名詞ごとの票を足す。"""
    w = works[x["work"]]
    P = w["_p"]
    li, m = x["line"], x["m"]
    cnt = {}
    for n, c in P["mc"].items():
        v = c.get(m, 0) - P["mc_line"][li].get((n, m), 0)
        if v > 0:
            cnt[n] = v
    same = [sum(v for n, v in cnt.items() if n.split("|")[0] == c) for c in x["cand"]]
    spread = {}
    for t in TYPES:
        if t in ("される", "の"):                     # I27n で選ばれなかった種類は数えない（速さのため）
            spread[t] = [0.0, 0.0]
            continue
        row = [0.0, 0.0]
        for n, v in cnt.items():
            base = n.split("|")[0]
            if base in x["cand"] or base in drop:
                continue
            for k, c in enumerate(x["cand"]):
                sm = sim(w, li, c, n, t, m)
                if sm > 0:
                    row[k] += sm * v
                    if votes is not None and t == "受けた修飾":
                        votes[n.split("|")[0]][c == x["gold"]] += sm * v
        spread[t] = row
    return same, spread, L.cross_row(works, x, 1.0, 0.0)


_W = {}


def _tot(k):
    return totals(_W["works"], _W["items"][k], _W.get("drop", frozenset()))


def run_tots(works, items, jobs, drop=frozenset()):
    _W.update(works=works, items=items, drop=drop)
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            return pool.map(_tot, range(len(items)), chunksize=max(1, len(items) // (jobs * 20)))
    return [_tot(k) for k in range(len(items))]


def _votes(k):
    v = defaultdict(lambda: [0.0, 0.0])
    x = _W["items"][k]
    totals(_W["works"], x, votes=v)
    return x["work"], {n: tuple(c) for n, c in v.items()}


SETTINGS = {"受けた修飾 λ3（作品の中だけ）": ({"受けた修飾": 3.0}, 0.0),
            "I27n で学んだ組（受けた修飾 3・する 1・で・に 1 + μ1）": ({"受けた修飾": 3.0, "する": 1.0, "で・に": 1.0}, 1.0)}


def evaluate(items, tots):
    out = {}
    for name, (lam, mu) in SETTINGS.items():
        lam = {t: lam.get(t, 0.0) for t in TYPES}
        out[name] = N.score(items, tots, range(len(items)), lam, mu)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_work", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--min_votes", type=float, default=3.0)
    ap.add_argument("--top", type=float, default=0.02, help="分ける語の割合（作品ごと、票のある名詞のうち）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    items = L.make_items(works, a.seed, a.per_work)
    L.SHARED = L.shared(works)
    build(works, {})
    # 違和感を測る問い（別の seed、試しの問いと同じ行は除く）
    test_lines = {(x["work"], x["line"]) for x in items}
    probe = [x for x in L.make_items(works, a.seed + 1, a.per_work) if (x["work"], x["line"]) not in test_lines]
    print("items", len(items), "probe", len(probe), flush=True)
    _W.update(works=works, items=probe)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.jobs) as pool:
        vs = pool.map(_votes, range(len(probe)), chunksize=max(1, len(probe) // (a.jobs * 20)))
    votes = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for wi, d in vs:
        for n, (bad, good) in d.items():
            votes[wi][n][0] += bad
            votes[wi][n][1] += good
    freq = [Counter(t["lemma"] for ln in w["lines"] for t in ln["toks"] if t["pos"] in NOUN) for w in works]
    by_disc, by_freq = {}, {}
    disc_rows = []
    for wi in range(len(works)):
        cand = {n: c for n, c in votes[wi].items() if sum(c) >= a.min_votes}
        k = max(1, int(len(cand) * a.top)) if cand else 0
        disc = sorted(cand, key=lambda n: (-cand[n][0] / sum(cand[n]), n))[:k]
        by_disc[wi] = set(disc)
        by_freq[wi] = {n for n, _ in Counter({n: freq[wi][n] for n in cand}).most_common(k)}
        disc_rows += [cand[n][0] / sum(cand[n]) for n in cand]
    nd = sum(len(v) for v in by_disc.values())
    rec = dict(exp="I27p", works=len(works), n=len(items), probe=len(probe), 分けた語の数=nd,
               違和感の平均=round(sum(disc_rows) / max(1, len(disc_rows)), 4),
               選んだ語の違和感の平均=round(sum(votes[wi][n][0] / sum(votes[wi][n]) for wi in by_disc for n in by_disc[wi]) / max(1, nd), 4),
               重なり=sum(len(by_disc[wi] & by_freq[wi]) for wi in by_disc), 結果={})
    print(rec, flush=True)
    rec["結果"]["分けない"] = evaluate(items, run_tots(works, items, a.jobs))
    print("分けない", rec["結果"]["分けない"], flush=True)
    tots = []
    for wi in range(len(works)):
        ks = [k for k, x in enumerate(items) if x["work"] == wi]
        sub = [items[k] for k in ks]
        tots_w = run_tots(works, sub, a.jobs, frozenset(by_disc[wi]))
        tots += list(zip(ks, tots_w))
    tots = [t for _, t in sorted(tots)]
    rec["結果"]["違和感の高い語を広げから外す"] = evaluate(items, tots)
    print("外す", rec["結果"]["違和感の高い語を広げから外す"], flush=True)
    for name, sp in (("数で選んで分ける", by_freq), ("違和感で選んで分ける", by_disc)):
        build(works, sp)
        rec["結果"][name] = evaluate(items, run_tots(works, items, a.jobs))
        print(name, rec["結果"][name], flush=True)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
