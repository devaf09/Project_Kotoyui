"""介入の研究 実験I27x: 節をつないで行を作る（試行の環・段 1、振り返りの回数 2 回と 6 回）

I27s の行は述語 1 つ（元の行は述語 3.9 個）。持ち主（2026-11-08）が了承した「介入の設計」の 1 番目:
  節をつなぐ所で「試す → 振り返り → 気づきの器 → 反応の板」の環を回す。
部品（docs/thinking_parts.md の名前）:
  場の板   作品全体を直線の坂で束ねた板（I27s と同じ）。
  汎用板   ほかの作品から: 述語が枠を取る割合、主語と述語の組の裏づけ、行が k 個以上の述語を持つ割合（節の期待 e_k）、
           述語のて形（「見て」など、表記の板）、行の中の述語の流れ（前の述語 → 次の述語）。
  気づきの器 節を足すか: 足せば |e_k − 1|、足さなければ |e_k − 0|（枠の期待と同じ形）。節の述語・枠も I27s と同じ差。
  振り返り 作った行を解析し直し、節ごとに: 述語が見つかるか、最後の述語が根か、前の述語が後ろの述語に係るか、
           枠（主語・目的語・補語・修飾）が係るか。
  反応の板 崩れた枠・述語の差を束ね、同じ鍵を使う次の見立てを暗くする（I27s と同じ）。出力ごとに消す（段 1）。
  決め手   点 = Σ 明るさ − Σ |差| の勝者。
段 1 の回数（振り返りの回数）は機器のスペック: 2 回（エッジ）と 6 回（PC）を比べる。
物差し（I27s と同じ 2,000 か所）: 全部の係りが戻った割合、F1、隣F1、内容語の数、文字数、節の数。
Web 小説は _wn。結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i27x_chain_clauses.py --cache <解析済み JSON> --stats <jsonl> --out <例の md>
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27m_board_line as M  # noqa: E402
import i27o_board_line2 as O  # noqa: E402
import i27q_book_insert as Q  # noqa: E402
import i27r_check_repair as R  # noqa: E402
import i27s_fused_generate as S  # noqa: E402
from board_ops import Board, bundle, light, react, winner  # noqa: E402

PRED_POS = ("VERB", "ADJ")
CLAUSE_DEP = ("ROOT", "advcl", "conj")
MAXC = 3


def te_rows(w):
    """行ごとの「て形」の板: ("て", 原形, 表記)。述語の語のすぐあとが て・で の所から。"""
    rows = []
    for ln in w["lines"]:
        b = Board()
        t = ln["toks"]
        for i in range(len(t) - 1):
            if t[i]["pos"] in PRED_POS and t[i + 1]["text"] in ("て", "で"):
                b[("て", t[i]["lemma"], t[i]["text"] + t[i + 1]["text"])] += 1
        rows.append(b)
    return rows


def clause_rows(w):
    """行ごとの節の数の板: ("節", k) は述語が k 個以上ある行。("内流", 前の述語, 次の述語) は行の中の述語の並び。"""
    rows = []
    for ln in w["lines"]:
        b = Board()
        ps = [t["lemma"] for t in ln["toks"] if t["pos"] in PRED_POS and t["dep"] in CLAUSE_DEP]
        b[("行",)] += 1
        for k in range(1, MAXC + 1):
            if len(ps) >= k:
                b[("節", k)] += 1
        for x, y in zip(ps, ps[1:]):
            b[("内流", x, y)] += 1
            b[("内流計", x)] += 1
        rows.append(b)
    return rows


class Chain:
    def __init__(self, st, CG, cown):
        self.st, self.CG, self.cown = st, CG, cown

    def e_clause(self, k):
        n = light(self.CG, ("行",), minus=self.cown)
        return light(self.CG, ("節", k), minus=self.cown) / n if n else 0.0

    def te(self, p):
        c = Counter({k[2]: v for k, v in self.CG.items() if k[0] == "て" and k[1] == p})
        return max(c, key=lambda x: (c[x], x)) if c else None

    def next_preds(self, p, used, k=6):
        tot = light(self.CG, ("内流計", p), minus=self.cown)
        if not tot:
            return []
        c = [(light(self.CG, key, minus=self.cown) / tot, key[2]) for key in self.nexts.get(p, []) if key[2] not in used]
        c.sort(reverse=True)
        return c[:k]


def index_next(CG):
    nx = defaultdict(list)
    for k in CG:
        if k[0] == "内流":
            nx[k[1]].append(k)
    return nx


def clause_opts(st, s, p, k=3):
    """2 つ目からの節: 目的語・補語だけ（修飾は 1 つ目の節に任せる）。"""
    G, own = st.G, st.own
    out = {}
    for r, ro in S.ROLES:
        e = S.ratio(G, own, ("有", ro[:-1], p), ("出", p))
        c = Counter({n: v.get(ro + p, 0) for n, v in st.wb.items() if n != s})
        out[r] = [(None, -react(e, 0.0)[0])] + [(n, st.slot(n, ro + p, e)) for n in S.cands(c, k)]
    return out


def keys_of(clauses):
    """反応の板の鍵: 節ごとの述語と枠の語。"""
    ks = {}
    for j, c in enumerate(clauses):
        ks[(j, "述語")] = ("述語", c["述語"])
        for k in ("目的語", "補語", "修飾", "目的語の修飾"):
            if c.get(k):
                ks[(j, k)] = (k, c[k])
        if j == 0:
            ks[(0, "主語")] = ("組", c["主語"], c["述語"])
    return ks


def penalty(RB, clauses):
    return sum(light(RB, k) for k in set(keys_of(clauses).values()))


def propose(ch, first, RB, use_chain=True):
    """見立て: 1 つ目の節（I27s の候補の点 − 反応の板）に、節の期待で次の節を足していく。"""
    st = ch.st
    pre, pick = first
    clauses = [dict(pick)]
    score = pre
    used = {pick["述語"]}
    if not use_chain:
        return score, clauses
    for k in range(2, MAXC + 1):
        e = ch.e_clause(k)
        p_prev = clauses[-1]["述語"]
        if ch.te(p_prev) is None:
            break
        best = None
        for fl, q in ch.next_preds(p_prev, used):
            base = (1.0 - pow(2.718281828, -fl / 0.05)) + st.bright(pick["主語"], "する:" + q) - react(1.0, st.attested(pick["主語"], "する:" + q))[0]
            opts = clause_opts(st, pick["主語"], q)
            for (o, so), (b, sb) in itertools.product(opts["目的語"], opts["補語"]):
                if o is not None and o == b:
                    continue
                c = dict(主語=pick["主語"], 述語=q, 目的語=o, 補語=b, 修飾=None, 目的語の修飾=None)
                v = base + so + sb - penalty(RB, [c])
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


def render_chain(ch, clauses):
    st = ch.st
    line, surf = S.render2(st, clauses[0])
    if len(clauses) == 1:
        return line, [surf]
    head = line[:-1]                                   # 「。」を外す
    p0 = clauses[0]["述語"]
    pf = max(st.form["する:" + p0], key=lambda x: (st.form["する:" + p0][x], x)) if st.form["する:" + p0] else p0
    if head.endswith(pf):
        head = head[: -len(pf)]
    parts = [head + ch.te(p0)]
    surfs = [surf]
    for j, c in enumerate(clauses[1:], 1):
        sf, seg = {}, []
        if c["補語"]:
            sf["補語"] = st.BB_text.get(c["補語"], c["補語"])
            part = st.part.get((c["補語"], "で・に"), Counter())
            seg.append(sf["補語"] + (max(part, key=lambda x: (part[x], x)) if part else "に"))
        if c["目的語"]:
            sf["目的語"] = st.BB_text.get(c["目的語"], c["目的語"])
            seg.append(sf["目的語"] + "を")
        last = j == len(clauses) - 1
        if last:
            fm = st.form["する:" + c["述語"]]
            seg.append(max(fm, key=lambda x: (fm[x], x)) if fm else c["述語"])
        else:
            seg.append(ch.te(c["述語"]))
        parts.append("".join(seg))
        surfs.append(sf)
    return "、".join(parts) + "。", surfs


def check_chain(nlp, line, clauses, surfs):
    """振り返り: 節ごとの係りが戻るか。"""
    doc = nlp(line)
    roots = [t for t in doc if t.dep_ == "ROOT"]
    preds, pos = [], 0
    toks = list(doc)
    for c in clauses:
        hit = next((t for t in toks[pos:] if t.lemma_ == c["述語"] and t.pos_ in ("VERB", "ADJ", "NOUN", "AUX")), None)
        preds.append(hit)
        if hit is not None:
            pos = hit.i + 1
    res = {}
    n = len(clauses)
    for j, (c, pr) in enumerate(zip(clauses, preds)):
        if pr is None:
            res[(j, "述語")] = False
            continue
        if j == n - 1:
            res[(j, "述語")] = len(roots) == 1 and pr.i == roots[0].i
        else:
            later = {x.i for x in preds[j + 1:] if x is not None}
            res[(j, "述語")] = pr.head.i in later
        for slot, deps in (("目的語", ("obj",)), ("補語", ("obl",))):
            if c.get(slot):
                res[(j, slot)] = any(t.dep_ in deps and t.head.i == pr.i for t in R.finds(doc, surfs[j][slot]))
    pis = {x.i for x in preds if x is not None}
    s0 = surfs[0]["主語"]
    res[(0, "主語")] = any(t.dep_ in ("nsubj", "dislocated") and t.head.i in pis for t in R.finds(doc, s0))
    if clauses[0].get("修飾"):
        tg = {x.i for x in R.finds(doc, s0)}
        res[(0, "修飾")] = any(t.lemma_ == clauses[0]["修飾"] and t.dep_ in ("acl", "amod") and t.head.i in tg for t in doc)
    if clauses[0].get("目的語の修飾") and clauses[0].get("目的語"):
        tg = {x.i for x in R.finds(doc, surfs[0]["目的語"])}
        res[(0, "目的語の修飾")] = any(t.lemma_ == clauses[0]["目的語の修飾"] and t.dep_ in ("acl", "amod") and t.head.i in tg for t in doc)
    return res, doc


def ring(nlp, ch, pairs, budget, use_chain):
    """試行の環（段 1）: 見立て → 試す → 振り返り → 気づき → 反応の板 → 見立て。budget 回まで。"""
    st = ch.st
    firsts = []
    top = pairs[0][0] if pairs and pairs[0][0] > 0 else 1.0
    for v, s, p in pairs:
        firsts += S.combos(st, s, p, pb=1.0 - pow(2.718281828, -v / top))
    firsts.sort(key=lambda x: -x[0])
    firsts = firsts[:60]
    RB = Board()
    parsed, seen = [], set()
    while len(parsed) < budget:
        props = []
        for f in firsts:
            sc, cl = propose(ch, (f[0] - penalty(RB, [f[1]]), f[1]), RB, use_chain)
            props.append((sc, cl))
        props = [(sc, cl) for sc, cl in props if render_chain(ch, cl)[0] not in seen]
        if not props:
            break
        sc, cl = max(props, key=lambda x: (x[0], repr(x[1])))
        line, surfs = render_chain(ch, cl)
        seen.add(line)
        chk, doc = check_chain(nlp, line, cl, surfs)
        diffs = {k: react(1.0, float(v))[0] for k, v in chk.items()}
        parsed.append((sc - sum(diffs.values()), (line, cl, chk, doc)))
        ks = keys_of(cl)
        RB.update(bundle([(0, Board({ks[k]: d for k, d in diffs.items() if d and k in ks}))]))
        if not any(diffs.values()):
            break                                       # 差が無い = 止める
    return winner(parsed), len(parsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_stats", type=int, default=2000)
    ap.add_argument("--n", type=int, default=12)
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
        b = bundle(list(enumerate(clause_rows(w))))
        b.update(bundle(list(enumerate(te_rows(w)))))
        crow.append(b)
    CG = Board()
    for b in crow:
        CG.update(b)
    nexts = index_next(CG)
    propn = {t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] in ("PROPN", "PRON")}
    rng = random.Random(a.seed)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(10, len(w["lines"]) - 10)
             if any(t["dep"] == "nsubj" for t in w["lines"][li]["toks"])]
    rng.shuffle(sites)
    sites = sites[:a.n_stats]
    names = [("I27s 1 節・回数 2", False, 2), ("I27s 1 節・回数 6", False, 6), ("I27x 節をつなぐ・回数 2", True, 2), ("I27x 節をつなぐ・回数 6", True, 6)]
    res = {nm: Counter() for nm, _, _ in names}
    slot_ok = {nm: Counter() for nm, _, _ in names}
    slot_n = {nm: Counter() for nm, _, _ in names}
    orig = Counter()
    examples = []
    for si, (wi, li) in enumerate(sites):
        w = works[wi]
        lines = w["lines"]
        ref = M.content_lemmas(lines[li]["toks"])
        orig["内容語"] += sum(ref.values())
        orig["文字数"] += len(lines[li]["text"])
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
        board = (st.wb, st.form, st.part, w["_text"])
        ch = Chain(st, CG, crow[wi])
        ch.nexts = nexts
        pairs = S.pair_scores(board, avoid, prev_p, tr)
        ex = {}
        for nm, use_chain, budget in names:
            if not pairs:
                continue
            (line, cl, chk, doc), tries = ring(nlp, ch, pairs, budget, use_chain)
            g = Counter(x for c in cl for k, x in c.items() if x and not (k == "主語" and c is not cl[0]))
            r = res[nm]
            r["n"] += 1
            r["全部戻った"] += all(chk.values())
            r["解析し直した回数"] += tries
            r["F1"] += M.f1(ref, g)
            r["隣F1"] += max(M.f1(nb[0], g), M.f1(nb[1], g))
            r["内容語"] += S.n_content(doc)
            r["文字数"] += len(line)
            r["節の数"] += len(cl)
            for (j, k), v in chk.items():
                kk = f"{'1 つ目' if j == 0 else '2 つ目から'}の{k}"
                slot_n[nm][kk] += 1
                slot_ok[nm][kk] += v
            ex[nm] = (line, [f"{j + 1}:{k}" for (j, k), v in chk.items() if not v])
        for x, y in sub:
            tr[x][y] += 1
        if len(examples) < a.n and len(ex) == len(names):
            examples.append((wi, li, ex))
        if si % 200 == 0:
            print(si, flush=True)
    rec = dict(exp="I27x", n=len(sites), 元の行=dict(内容語=round(orig["内容語"] / len(sites), 3), 文字数=round(orig["文字数"] / len(sites), 2)),
               節の期待={k: round(Chain(None, CG, Board()).e_clause(k), 3) for k in range(1, MAXC + 1)}, 結果={})
    for nm, r in res.items():
        rec["結果"][nm] = dict(n=r["n"], **{m: round(r[m] / max(1, r["n"]), 4) for m in r if m != "n"},
                             枠ごとに戻った割合={k: round(slot_ok[nm][k] / slot_n[nm][k], 4) for k in slot_n[nm]})
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = ["# I27x 節をつないで作った行（非公開）", "", "（元）が抜いた元の行。[崩れ] は振り返りで戻らなかった節:枠。", ""]
    for j, (wi, li, ex) in enumerate(examples, 1):
        lines = works[wi]["lines"]
        out += [f"## {j}. 作品 {works[wi]['name']} {li} 行目", "",
                f"- 　 {lines[li - 1]['text']}", f"- （元）{lines[li]['text']}", f"- 　 {lines[li + 1]['text']}", ""]
        for nm, (line, bad) in ex.items():
            out.append(f"- ▶ {nm}: **{line}**　[崩れ: {'、'.join(bad) or 'なし'}]")
        out.append("")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
