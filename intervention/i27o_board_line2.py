"""介入の研究 実験I27o: 想定属性板から 1 行を作る（I27m）の続き。言い直しを減らし、話の流れで述語を選ぶ

I27m の行は読めるが、すぐ隣の行の言い直しになりやすかった（元の行との内容語 F1 0.041、すぐ前の行を写すと 0.047）。I27m の NOTES の次の手を 4 つ入れる:
  (a) 形式名詞（こと・もの・気 など）を主語にしない。
  (b) 隣の行（前後 1 行）に既にある「主語 + 述語」の組は、重みを下げる（言い直しを避ける）。
  (c) 話の流れの板: 行の述語（行の根）から次の行の述語への移り（P_前 → P）を、全作品の行の並びで数える（試しの行に触れる 2 つの移りは除く）。
      主語と述語の組の重み × (κ + 移りの割合 P(P | P_前)) で選ぶ。
  (d) 修飾の中の語も持つ: 連体の修飾を、修飾の語に係る語（「〜になる」の「〜」）まで含めた形で覚える。
物差し（3,000 か所）: 元の行との内容語 F1、隣の行（前後 1 行の大きい方）との F1（言い直しの度合い、低い方が新しい）、
  述語が元の行の述語と同じ割合、主語が元の行の主語と同じ割合。目安: すぐ前の行を写す、窓の中でいちばん多い述語、前の行の述語から移りのいちばん多い述語。
Web 小説は _wn。例の md は非公開の置き場だけに置く。本文はコードに書かない。
実行: python intervention/i27o_board_line2.py --cache <解析済み JSON> --stats <jsonl> [--out <例の md> --n 10]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27m_board_line as M  # noqa: E402
from i27l_wave_scale import FORMAL2  # noqa: E402

NOUN, PRED = M.NOUN, M.PRED


def subtree_span(toks, i, ks):
    """修飾の語 i に係る語を含めた、i までの連続した範囲の表記（句読点・かぎかっこは落とす）。"""
    seen, st = {i}, [i]
    while st:
        j = st.pop()
        for c in ks.get(j, []):
            if c not in seen and c < i + 3:
                seen.add(c)
                st.append(c)
    lo = min(seen)
    hi = i
    while hi + 1 < len(toks) and toks[hi + 1]["dep"] in ("aux", "fixed", "mark") and toks[hi + 1]["pos"] != "PUNCT":
        hi += 1                                                  # 修飾の語のすぐ後ろに続く助動詞（「〜てい」「た」）まで
    return "".join(toks[k]["text"] for k in range(lo, hi + 1) if toks[k]["pos"] != "PUNCT" and toks[k]["text"] not in M.SKIP)


def line_attrs2(toks, inner):
    if not inner:
        return M.line_attrs(toks)
    ks = M.kids(toks)
    out = []
    for n, a, f, p in M.line_attrs(toks):
        out.append((n, a, f, p))
    # 修飾の形を、修飾の中の語を含めた形に置き換える
    rep = {}
    for i, t in enumerate(toks):
        if t["pos"] in PRED and t["dep"] in ("acl", "amod"):
            rep[M.mod_surface(toks, i, ks)] = subtree_span(toks, i, ks)
    return [(n, a, rep.get(f, f) if a.startswith("修飾:") else f, p) for n, a, f, p in out]


def build_board2(lines, li, win, inner):
    w = defaultdict(Counter)
    form = defaultdict(Counter)
    part = defaultdict(Counter)
    text = {}
    for lj in range(max(0, li - win), min(len(lines), li + win + 1)):
        if lj == li:
            continue
        toks = lines[lj]["toks"]
        for t in toks:
            if t["pos"] in NOUN:
                text.setdefault(t["lemma"], t["text"])
        seen = set()
        for n, a, f, p in line_attrs2(toks, inner):
            if (n, a) in seen:
                continue
            seen.add((n, a))
            wt = 1.0 + 0.1 / abs(lj - li)
            w[n][a] += wt
            if f:
                form[a][f] += wt
            if p:
                part[(n, a.split(":")[0])][p] += wt
    return w, form, part, text


def root_pred(toks):
    for t in toks:
        if t["dep"] == "ROOT" and t["pos"] in PRED:
            return t["lemma"]
    return None


def subj_of(toks):
    r = [i for i, t in enumerate(toks) if t["dep"] == "ROOT"]
    for t in toks:
        if t["dep"] == "nsubj" and r and t["head"] == r[0]:
            return t["lemma"]
    return None


def flow_board(works):
    """述語の移り: P_前 → P の数え（全作品、行の並び）。"""
    tr = defaultdict(Counter)
    roots = []
    for w in works:
        rp = [root_pred(ln["toks"]) for ln in w["lines"]]
        roots.append(rp)
        for a, b in zip(rp, rp[1:]):
            if a and b:
                tr[a][b] += 1
    return tr, roots


def pairs_in(toks):
    return {(n, a.split(":", 1)[1]) for n, a, f, p in M.line_attrs(toks) if a.startswith("する:")}


def choose(board, *, skip_formal, avoid, flow, prev_p, tr, kappa=0.05, avoid_w=0.2, joint=False):
    """主語と述語の組を選ぶ。返り値 (主語, 述語)。
    joint でなければ I27m と同じ（主語 = 板の重みの合計がいちばん大きい名詞、述語 = その する: の 1 位）。
    joint なら、組ごとに 主語の重み × 述語の重み × (避ける・流れの係数) のいちばん大きい組。"""
    w = board[0]
    if flow == "先":
        return choose_flow_first(w, skip_formal, avoid, prev_p, tr, avoid_w)
    if not joint:
        subj = Counter({n: sum(c.values()) for n, c in w.items()
                        if any(a.startswith("する:") for a in c) and not (skip_formal and n in FORMAL2)})
        subj = +subj
        if not subj:
            return None, None
        s = max(subj, key=lambda k: (subj[k], k))
        preds = Counter({a.split(":", 1)[1]: v for a, v in w[s].items() if a.startswith("する:")})
        return s, max(preds, key=lambda k: (preds[k], k))
    best = None
    tot = sum(tr.get(prev_p, {}).values()) if (flow and prev_p) else 0
    subj_w = {n: sum(c.values()) for n, c in w.items()}
    for n, c in w.items():
        if skip_formal and n in FORMAL2:
            continue
        for a, v in c.items():
            if not a.startswith("する:"):
                continue
            p = a.split(":", 1)[1]
            s = subj_w[n] * v                       # I27m: 話題の重い主語の、重い述語（主語の重み × 述語の重み）
            if avoid and (n, p) in avoid:
                s *= avoid_w
            if tot:
                s *= kappa + tr[prev_p].get(p, 0) / tot
            key = (s, n, p)
            if best is None or key > best:
                best = key
    return (best[1], best[2]) if best else (None, None)


def choose_flow_first(w, skip_formal, avoid, prev_p, tr, avoid_w, kappa2=0.1):
    """(e) 述語を先に話の流れの板で選ぶ: P(P | P_前) × (κ2 + 窓の板での述語の重みの割合)。主語は、その述語を する: に持つ名詞の重い方。"""
    pw = Counter()
    for n, c in w.items():
        for a, v in c.items():
            if a.startswith("する:"):
                pw[a.split(":", 1)[1]] += v
    subj_w = {n: sum(c.values()) for n, c in w.items() if not (skip_formal and n in FORMAL2)}
    if not subj_w:
        return None, None
    nxt = tr.get(prev_p, {}) if prev_p else {}
    tot = sum(nxt.values())
    mx = max(pw.values()) if pw else 1
    best = None
    cands = set(pw) | {p for p, v in nxt.items() if v > 0}
    for p in cands:
        f = (nxt.get(p, 0) / tot if tot else 0.0) + 1e-3
        s = f * (kappa2 + pw.get(p, 0) / mx)
        if best is None or (s, p) > best:
            best = (s, p)
    if best is None:
        return None, None
    p = best[1]
    sub = [(subj_w[n] * c.get("する:" + p, 0) * (avoid_w if avoid and (n, p) in avoid else 1.0), n)
           for n, c in w.items() if n in subj_w and c.get("する:" + p, 0)]
    if not sub:                                                   # 窓の板にその述語の主語が無ければ、話題のいちばん重い主語
        sub = [(subj_w[n] * (avoid_w if avoid and (n, p) in avoid else 1.0), n) for n, c in w.items()
               if n in subj_w and any(a.startswith("する:") for a in c)]
    if not sub:
        return None, None
    return max(sub)[1], p


def expand2(board, s, p):
    """主語と述語が決まった後の並べ方（I27m の expand と同じ）。"""
    w, form, part, text = board

    def top(c):
        return max(c, key=lambda k: (c[k], k)) if c else None
    pick = dict(主語=s, 述語=p)
    objs = +Counter({n: c.get("される:" + p, 0) for n, c in w.items() if n != s})
    obls = +Counter({n: c.get("で・に:" + p, 0) for n, c in w.items() if n != s})
    o, b = top(objs), top(obls)
    mods = Counter({a.split(":", 1)[1]: v for a, v in w[s].items() if a.startswith("修飾:")})
    m_of = s
    if not mods and o:
        mods = Counter({a.split(":", 1)[1]: v for a, v in w[o].items() if a.startswith("修飾:")})
        m_of = o
    m = top(mods)

    def pt(n, role, dflt):
        return top(part.get((n, role), Counter())) or dflt
    head = []
    mform = top(form["修飾:" + m]) if m else None
    if m and m_of == s:
        head.append(mform or m)
    head.append(text.get(s, s) + pt(s, "する", "が"))
    if b:
        head.append(text.get(b, b) + pt(b, "で・に", "に"))
    if o:
        if m and m_of == o:
            head.append(mform or m)
        head.append(text.get(o, o) + "を")
    head.append((top(form["する:" + p]) or p) + "。")
    pick.update(目的語=o, 補語=b, 修飾=m)
    return "".join(head), pick, mform


VARIANTS = {
    "I27m（元の形）": dict(skip_formal=False, avoid=False, flow=False, inner=False),
    "(a) 形式名詞を主語にしない": dict(skip_formal=True, avoid=False, flow=False, inner=False),
    "(a) 組の積で選ぶ": dict(skip_formal=True, avoid=False, flow=False, inner=False, joint=True),
    "(a)+(b) 隣の行の組を避ける": dict(skip_formal=True, avoid=True, flow=False, inner=False),
    "(a)+(c) 話の流れの板": dict(skip_formal=True, avoid=False, flow=True, inner=False),
    "(a)+(b)+(c)": dict(skip_formal=True, avoid=True, flow=True, inner=False),
    "(a)+(b)+(c)+(d) 修飾の中の語も": dict(skip_formal=True, avoid=True, flow=True, inner=True),
    "(a)+(b)+(e) 述語を先に話の流れで": dict(skip_formal=True, avoid=True, flow="先", inner=False),
    "(a)+(b)+(e)+(d)": dict(skip_formal=True, avoid=True, flow="先", inner=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--win", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_stats", type=int, default=3000)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default=None, help="例の md（非公開）")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    tr_all, roots = flow_board(works)
    rng = random.Random(a.seed)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(a.win, len(w["lines"]) - a.win)
             if any(t["dep"] == "nsubj" for t in w["lines"][li]["toks"])]
    rng.shuffle(sites)
    sites = sites[:a.n_stats]
    lemma_of = {}                                                     # 表記 → 見出し（修飾の中の語を内容語として数えるため）
    for w in works:
        for ln in w["lines"]:
            for t in ln["toks"]:
                if t["pos"] in M.CONTENT:
                    lemma_of.setdefault(t["text"], t["lemma"])
    res = {k: Counter() for k in VARIANTS}
    base = Counter()
    examples = []
    for si, (wi, li) in enumerate(sites):
        lines = works[wi]["lines"]
        toks = lines[li]["toks"]
        ref = M.content_lemmas(toks)
        nb = [M.content_lemmas(lines[lj]["toks"]) for lj in (li - 1, li + 1)]
        ref_p, ref_s = root_pred(toks), subj_of(toks)
        prev_p = roots[wi][li - 1]
        # 試しの行に触れる移りは数えから除く
        tr = tr_all
        sub = []
        if prev_p and roots[wi][li]:
            sub.append((prev_p, roots[wi][li]))
        if roots[wi][li] and li + 1 < len(roots[wi]) and roots[wi][li + 1]:
            sub.append((roots[wi][li], roots[wi][li + 1]))
        for x, y in sub:
            tr[x][y] -= 1
        avoid = pairs_in(lines[li - 1]["toks"]) | pairs_in(lines[li + 1]["toks"])
        base["n"] += 1
        base["すぐ前の行 F1"] += M.f1(ref, nb[0])
        base["すぐ前の行 隣F1"] += 1.0
        wb = M.build_board(lines, li, a.win)[0]
        pc = Counter()
        for n, c in wb.items():
            for at, v in c.items():
                if at.startswith("する:"):
                    pc[at.split(":", 1)[1]] += v
        if pc:
            base["窓でいちばん多い述語 述語一致"] += pc.most_common(1)[0][0] == ref_p
        if prev_p and tr.get(prev_p):
            base["前の述語からの移りの 1 位 述語一致"] += max(tr[prev_p], key=lambda k: (tr[prev_p][k], k)) == ref_p
        base["前の行の述語を写す 述語一致"] += prev_p == ref_p
        ex = {}
        for name, v in VARIANTS.items():
            board = build_board2(lines, li, a.win, v["inner"])
            s, p = choose(board, skip_formal=v["skip_formal"], avoid=avoid if v["avoid"] else None,
                          flow=v["flow"], prev_p=prev_p, tr=tr, joint=v.get("joint", v["avoid"] or v["flow"]))
            if not s:
                continue
            line, pick, mform = expand2(board, s, p)
            g = Counter(x for x in pick.values() if x)
            if v["inner"] and mform:
                for ch in mform_split(mform, lemma_of):           # 修飾の中の語（見出しが取れたもの、修飾の語そのものは除く）
                    if ch not in g:
                        g[ch] += 1
            r = res[name]
            r["n"] += 1
            r["F1"] += M.f1(ref, g)
            r["隣F1"] += max(M.f1(nb[0], g), M.f1(nb[1], g))
            r["述語一致"] += p == ref_p
            r["主語一致"] += s == ref_s
            r["主語が形式名詞"] += s in FORMAL2
            r["組が隣の行にある"] += (s, p) in avoid
            ex[name] = line
        for x, y in sub:
            tr[x][y] += 1
        if a.out and len(examples) < a.n and len(ex) == len(VARIANTS):
            examples.append((wi, li, ex))
    rec = dict(exp="I27o", win=a.win, n=base["n"],
               目安={k: round(v / base["n"], 4) for k, v in base.items() if k != "n"},
               結果={k: dict(n=r["n"], **{m: round(r[m] / max(1, r["n"]), 4) for m in r if m != "n"}) for k, r in res.items()})
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.out:
        out = ["# I27o 想定属性板から作った 1 行、言い直しを避けて話の流れで選ぶ（非公開）", "",
               "▶ が作り方ごとの行、（元）が抜いた元の行。前後は 2 行ずつ。", ""]
        for j, (wi, li, ex) in enumerate(examples, 1):
            lines = works[wi]["lines"]
            out += [f"## {j}. 作品 {works[wi]['name']} {li} 行目", ""]
            for lj in range(li - 2, li):
                out.append(f"- 　 {lines[lj]['text']}")
            out.append(f"- （元）{lines[li]['text']}")
            for lj in range(li + 1, li + 3):
                out.append(f"- 　 {lines[lj]['text']}")
            out.append("")
            for k, line in ex.items():
                out.append(f"- ▶ {k}: **{line}**")
            out.append("")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(out))


def mform_split(mform, lemma_of):
    """修飾の形の中にある内容語の見出し（最長一致で、作品に出た表記から拾う）。"""
    out, i = [], 0
    while i < len(mform):
        for j in range(min(len(mform), i + 8), i, -1):
            if mform[i:j] in lemma_of and j - i >= 2:
                out.append(lemma_of[mform[i:j]])
                i = j
                break
        else:
            i += 1
    return out


if __name__ == "__main__":
    main()
