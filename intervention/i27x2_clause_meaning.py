"""介入の研究 実験I27x2: つないだ後ろの節にも意味の反応を掛ける（気づきの器）

持ち主（2026-11-08）: 「１から始めてくれる？」（次の予定の 1: I27x の後ろの節の意味が弱いのを直す）
I27x では、後ろの節の枠（目的語・補語）の意味の差 |1 − 裏づけ| で、代名詞（何・それ など）は固有名詞と同じく
組の裏づけなしで通っていた（I27s の propn に PRON が入っている）。このため「何を〜して」のような節が残った。
I27x2 の気づきの器（後ろの節）:
  1. 枠の語が代名詞なら、組の裏づけも見る（固有名詞だけ免除）。
  2. 節の述語と前の節の述語の組（内流）の裏づけを、主語と述語の組と同じ形の差 |1 − 裏づけ| にして足す。
  3. 目的語と補語を両方埋めるなら、その 3 つ（目的語・補語・述語）がほかの作品の同じ行に一緒に出たか（双の裏づけ）。
     出ていなければ |1 − 裏づけ| の差（「口に髪をして」のように、組ごとには裏づけがあっても一緒には無い）。
  3 と 4 は 1 つ目の節にも掛ける（全部の節で同じ気づき）。
  4. 問いの語の期待: 語がほかの作品で問いの行（？・か で終わる）に出る割合 r。作る行は言い切りなので差 |r − 0|（「何をして」）。
物差し: I27x と同じものに、意味も通る割合（I27s と同じ閾値の物差し: 枠 ≥3 かつ 組 ≥1、ほかの作品で）を全部の節で足す。
  代名詞を免除しない物差しも並べる。
Web 小説は _wn。結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i27x2_clause_meaning.py --cache <解析済み JSON> --stats <jsonl> --out <例の md>
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27m_board_line as M  # noqa: E402
import i27o_board_line2 as O  # noqa: E402
import i27q_book_insert as Q  # noqa: E402
import i27s_fused_generate as S  # noqa: E402
import i27x_chain_clauses as X  # noqa: E402
from board_ops import Board, bundle, light, react  # noqa: E402

SAT_FLOW = 1.0
SAT_JOINT = 1.0
NOUNS = ("NOUN", "PRON", "PROPN")


def joint_rows(w):
    """行ごとの板: ("双", 目的語, 補語, 述語) と、("名行", 語) / ("問", 語)（問いの行に出た）。"""
    rows = []
    for ln, la in zip(w["lines"], w["_attrs"]):
        b = Board()
        objs, comps = {}, {}
        for n, a, f, p in la:
            role, x = a.split(":", 1)
            if role == "される":
                objs.setdefault(x, set()).add(n)
            elif role == "で・に":
                comps.setdefault(x, set()).add(n)
        for x in set(objs) & set(comps):
            for o in objs[x]:
                for c in comps[x]:
                    b[("双", o, c, x)] += 1
        t = ln["text"].rstrip("」』）) 　")
        q = t.endswith(("？", "?", "か"))
        for lem in {tk["lemma"] for tk in ln["toks"] if tk["pos"] in NOUNS}:
            b[("名行", lem)] += 1
            if q:
                b[("問", lem)] += 1
        rows.append(b)
    return rows


def ask_rate(ch, n):
    d = light(ch.JG, ("名行", n), minus=ch.jown)
    return light(ch.JG, ("問", n), minus=ch.jown) / d if d else 0.0


def attested_strict(st, n, a):
    """組の裏づけ（代名詞は免除しない。固有名詞だけ免除）。"""
    fr = light(st.G, ("枠", a), sat=S.SAT_FRAME, minus=st.own)
    pr = 1.0 if n in st.propn_only else light(st.G, ("組", n, a), sat=S.SAT_PAIR, minus=st.own)
    return fr * pr


def slot_strict(st, n, a, e_fill):
    return st.bright(n, a) - react(1.0, attested_strict(st, n, a))[0] - react(e_fill, 1.0)[0]


def clause_opts2(ch, s, p, k=3):
    st = ch.st
    G, own = st.G, st.own
    out = {}
    for r, ro in S.ROLES:
        e = S.ratio(G, own, ("有", ro[:-1], p), ("出", p))
        c = Counter({n: v.get(ro + p, 0) for n, v in st.wb.items() if n != s})
        out[r] = [(None, -react(e, 0.0)[0])] + [(n, slot_strict(st, n, ro + p, e) - react(ask_rate(ch, n), 0.0)[0]) for n in S.cands(c, k)]
    return out


def first_react(ch, pick):
    """1 つ目の節にも 3・4 の気づき（問いの語の期待、目的語と補語の双の裏づけ）。"""
    d = sum(react(ask_rate(ch, pick[k]), 0.0)[0] for k in ("目的語", "補語") if pick.get(k))
    if pick.get("目的語") and pick.get("補語"):
        d += react(1.0, light(ch.JG, ("双", pick["目的語"], pick["補語"], pick["述語"]), sat=SAT_JOINT, minus=ch.jown))[0]
    return d


def propose2(ch, first, RB, use_chain=True):
    """I27x の propose と同じ形で、後ろの節に 1・2 の気づきを足す。"""
    st = ch.st
    pre, pick = first
    pre -= first_react(ch, pick)
    clauses = [dict(pick)]
    score = pre
    used = {pick["述語"]}
    if not use_chain:
        return score, clauses
    for k in range(2, X.MAXC + 1):
        e = ch.e_clause(k)
        p_prev = clauses[-1]["述語"]
        if ch.te(p_prev) is None:
            break
        best = None
        for fl, q in ch.next_preds(p_prev, used):
            flow_seen = light(ch.CG, ("内流", p_prev, q), sat=SAT_FLOW, minus=ch.cown)
            base = (1.0 - math.exp(-fl / 0.05)) + st.bright(pick["主語"], "する:" + q) \
                - react(1.0, attested_strict(st, pick["主語"], "する:" + q))[0] - react(1.0, flow_seen)[0]
            opts = clause_opts2(ch, pick["主語"], q)
            for (o, so), (b, sb) in itertools.product(opts["目的語"], opts["補語"]):
                if o is not None and o == b:
                    continue
                c = dict(主語=pick["主語"], 述語=q, 目的語=o, 補語=b, 修飾=None, 目的語の修飾=None)
                v = base + so + sb - X.penalty(RB, [c])
                if o is not None and b is not None:
                    v -= react(1.0, light(ch.JG, ("双", o, b, q), sat=SAT_JOINT, minus=ch.jown))[0]
                if best is None or v > best[0]:
                    best = (v, c)
        if best is None:
            break
        add = best[0] - react(e, 1.0)[0]
        stop = -react(e, 0.0)[0]
        if add <= stop:
            score += stop
            break
        score += add
        clauses.append(best[1])
        used.add(best[1]["述語"])
    return score, clauses


def meaning_all(st, clauses, free):
    """物差し: 全部の節の枠が、ほかの作品で 枠 ≥3 かつ 組 ≥1（free の語は組を免除）。"""
    G, own = st.G, st.own
    res = {}
    for j, c in enumerate(clauses):
        for slot, role in (("主語", "する:"), ("目的語", "される:"), ("補語", "で・に:")):
            n = c.get(slot)
            if not n or (slot == "主語" and j > 0):
                continue
            a = role + c["述語"]
            fr = G[("枠", a)] - own[("枠", a)]
            pr = G[("組", n, a)] - own[("組", n, a)]
            res[(j, slot)] = fr >= 3 and (n in free or pr >= 1)
        if j == 0:
            for slot, of in (("修飾", "主語"), ("目的語の修飾", "目的語")):
                m, n = c.get(slot), c.get(of)
                if m and n:
                    a = "修飾:" + m
                    res[(0, slot)] = (G[("枠", a)] - own[("枠", a)]) >= 3 and (n in free or G[("組", n, a)] - own[("組", n, a)] >= 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_stats", type=int, default=2000)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--only", default="", help="この名前を含む組み方だけ回す（例: 回数 6）")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("ja_ginza")
    works = json.load(open(a.cache, encoding="utf-8"))
    Q.prep(works)
    tr, roots = O.flow_board(works)
    rows = [S.line_rows(w) for w in works]
    frows = [S.frame_rows(w) for w in works]
    own = [bundle(list(enumerate(fr))) for fr in frows]
    G = Board()
    for b in own:
        G.update(b)
    crow = []
    for w in works:
        b = bundle(list(enumerate(X.clause_rows(w))))
        b.update(bundle(list(enumerate(X.te_rows(w)))))
        crow.append(b)
    CG = Board()
    for b in crow:
        CG.update(b)
    nexts = X.index_next(CG)
    jown = [bundle(list(enumerate(joint_rows(w)))) for w in works]
    JG = Board()
    for b in jown:
        JG.update(b)
    propn = {t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] in ("PROPN", "PRON")}
    propn_only = {t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] == "PROPN"}
    rng = random.Random(a.seed)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(10, len(w["lines"]) - 10)
             if any(t["dep"] == "nsubj" for t in w["lines"][li]["toks"])]
    rng.shuffle(sites)
    sites = sites[:a.n_stats]
    names = [("I27x・回数 2", False, 2), ("I27x2 後ろの節に意味の反応・回数 2", True, 2),
             ("I27x・回数 6", False, 6), ("I27x2 後ろの節に意味の反応・回数 6", True, 6)]
    if a.only:
        names = [x for x in names if a.only in x[0]]
    res = {nm: Counter() for nm, _, _ in names}
    slot_ok = {nm: Counter() for nm, _, _ in names}
    slot_n = {nm: Counter() for nm, _, _ in names}
    examples = []
    for si, (wi, li) in enumerate(sites):
        w = works[wi]
        lines = w["lines"]
        ref = M.content_lemmas(lines[li]["toks"])
        nb = [M.content_lemmas(lines[lj]["toks"]) for lj in (li - 1, li + 1)]
        prev_p = roots[wi][li - 1]
        sub = [(x, y) for x, y in ((prev_p, roots[wi][li]), (roots[wi][li], roots[wi][li + 1])) if x and y]
        for x, y in sub:
            tr[x][y] -= 1
        avoid = O.pairs_in(lines[li - 1]["toks"]) | O.pairs_in(lines[li + 1]["toks"])
        n = len(lines)
        BB = bundle([(lj, rows[wi][lj]) for lj in range(n) if lj != li], weight=lambda lj: Q.slope("直線", abs(lj - li), n))
        st = S.Site(BB, G, own[wi], propn)
        st.BB_text = w["_text"]
        st.propn_only = propn_only
        board = (st.wb, st.form, st.part, w["_text"])
        ch = X.Chain(st, CG, crow[wi])
        ch.nexts = nexts
        ch.JG, ch.jown = JG, jown[wi]
        pairs = S.pair_scores(board, avoid, prev_p, tr)
        ex = {}
        for nm, strict, budget in names:
            if not pairs:
                continue
            X.propose = propose2 if strict else _orig_propose      # ring は X.propose を呼ぶ
            (line, cl, chk, doc), tries = X.ring(nlp, ch, pairs, budget, True)
            mn = meaning_all(st, cl, propn)
            mn2 = meaning_all(st, cl, propn_only)
            later = cl[1:]
            g = Counter(x for c in cl for k, x in c.items() if x and not (k == "主語" and c is not cl[0]))
            r = res[nm]
            r["n"] += 1
            r["全部戻った"] += all(chk.values())
            r["意味も通る"] += all(chk.values()) and all(mn.values())
            r["意味も通る（代名詞も裏づけ）"] += all(chk.values()) and all(mn2.values())
            r["後ろの節の意味が通る"] += all(v for (j, _), v in mn2.items() if j > 0)
            r["後ろの節の代名詞"] += sum(1 for c in later for s_ in ("目的語", "補語") if c.get(s_) in propn and c.get(s_) not in propn_only)
            r["後ろの節の双の裏づけ"] += all(JG[("双", c["目的語"], c["補語"], c["述語"])] - jown[wi][("双", c["目的語"], c["補語"], c["述語"])] >= 1
                                        for c in later if c.get("目的語") and c.get("補語"))
            r["解析し直した回数"] += tries
            r["F1"] += M.f1(ref, g)
            r["隣F1"] += max(M.f1(nb[0], g), M.f1(nb[1], g))
            r["内容語"] += S.n_content(doc)
            r["文字数"] += len(line)
            r["節の数"] += len(cl)
            for (j, k), v in mn2.items():
                kk = f"{'1 つ目' if j == 0 else '2 つ目から'}の{k}"
                slot_n[nm][kk] += 1
                slot_ok[nm][kk] += v
            ex[nm] = (line, cl, [f"{j + 1}:{k}" for (j, k), v in chk.items() if not v] + [f"{j + 1}:{k}（意味）" for (j, k), v in mn2.items() if not v])
        X.propose = _orig_propose
        for x, y in sub:
            tr[x][y] += 1
        if len(examples) < a.n and len(ex) == len(names):
            examples.append((wi, li, ex))
        if si % 200 == 0:
            print(si, flush=True)
    rec = dict(exp="I27x2", n=len(sites), 結果={})
    for nm, r in res.items():
        rec["結果"][nm] = dict(n=r["n"], **{m: round(r[m] / max(1, r["n"]), 4) for m in r if m != "n"},
                             意味が通った割合_代名詞も裏づけ={k: round(slot_ok[nm][k] / slot_n[nm][k], 4) for k in slot_n[nm]})
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = ["# I27x2 後ろの節に意味の反応を掛けた行（非公開）", "",
           "（元）が抜いた元の行。[崩れ] は振り返りで戻らなかった節:枠と、意味の物差し（代名詞も裏づけ）で通らなかった節:枠。", ""]
    for j, (wi, li, ex) in enumerate(examples, 1):
        lines = works[wi]["lines"]
        out += [f"## {j}. 作品 {works[wi]['name']} {li} 行目", "",
                f"- 　 {lines[li - 1]['text']}", f"- （元）{lines[li]['text']}", f"- 　 {lines[li + 1]['text']}", ""]
        orig = "・".join(f"{n}|{at}" for n, at, f_, p_ in works[wi]["_attrs"][li]) or "—"
        out.append(f"- （元を縮めた形）{orig}")
        for nm, (line, cl, bad) in ex.items():
            comp = " ／ ".join("・".join(f"{k}={v}" for k, v in c.items() if v and not (k == "主語" and c is not cl[0])) for c in cl)
            out.append(f"- ▶ {nm}: **{line}**　[崩れ: {'、'.join(bad) or 'なし'}]")
            out.append(f"  - 縮めた形（選んだ枠）: {comp}")
        out.append("")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


_orig_propose = X.propose

if __name__ == "__main__":
    main()
