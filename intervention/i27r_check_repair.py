"""介入の研究 実験I27r: 作った行を解析し直して確かめ（逆回しのチェック）、崩れた枠を差し替える（介入）

I27q の方法 1（作品全体・直線の坂の板）で作った行は、崩れたもの（解析の切れ端が主語、助詞の取り違え）が多かった。
持ち主（2026-11-08）: 「介入による自分チェックや文法解析の逆回しはうまくいっていないってことなのかなぁ」→ 2 段を足す:
  1. 逆回しのチェック: 作った行を GiNZA で解析し直し、選んだ語の係りが戻るかを見る。
     主語 → 述語（nsubj）、目的語 → 述語（obj）、補語 → 述語（obl）、修飾 → その名詞（acl / amod）、述語が行の根。戻らなかった枠 = 崩れた枠。
  1b. 意味のチェック（違和感）: 枠の組（主語 + する:述語 など）が、ほかの作品の板にもあるか（固有名詞は枠だけ）。無い枠も崩れた枠として直す（4 つ目の読み方）。
  2. 介入: 崩れた枠ごとに直す。修飾・補語・目的語が崩れたらその枠を外す。主語・述語が崩れたら、次の組（主語と述語の組の 2 位以下）に替える。
     先に、切れ端の名詞（かな 1 字・作品に 2 回未満）は主語・目的語・補語にしない（違和感の前さばき）。解析し直しは 1 行につき最大 6 回。
物差し（2,000 か所、行を 1 つ抜いてそこに作る）: 全部の係りが戻った行の割合、枠ごとの戻った割合、元の行との内容語 F1、隣の行との F1。
Web 小説は _wn。例の md と結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i27r_check_repair.py --cache <解析済み JSON> --stats <jsonl> --out <例の md>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27m_board_line as M  # noqa: E402
import i27o_board_line2 as O  # noqa: E402
import i27q_book_insert as Q  # noqa: E402
from i27l_wave_scale import FORMAL2  # noqa: E402

KANA1 = re.compile(r"^[ぁ-んァ-ンー]$")


def ranked_pairs(board, avoid, prev_p, tr, frag, kappa=0.05, avoid_w=0.2, k=8):
    """I27o の joint と同じ点で、主語と述語の組を上から k 個。"""
    w = board[0]
    subj_w = {n: sum(c.values()) for n, c in w.items()}
    tot = sum(tr.get(prev_p, {}).values()) if prev_p else 0
    out = []
    for n, c in w.items():
        if n in FORMAL2 or (frag and n in frag):
            continue
        for a, v in c.items():
            if not a.startswith("する:"):
                continue
            p = a.split(":", 1)[1]
            s = subj_w[n] * v
            if avoid and (n, p) in avoid:
                s *= avoid_w
            if tot:
                s *= kappa + tr[prev_p].get(p, 0) / tot
            out.append((s, n, p))
    out.sort(reverse=True)
    return [(n, p) for _, n, p in out[:k]]


def render(board, s, p, drop=(), frag=None):
    """I27o の expand2 と同じ並べ方。drop の枠（修飾・補語・目的語）は外す。frag の名詞は目的語・補語にしない。"""
    w, form, part, text = board

    def top(c):
        return max(c, key=lambda k: (c[k], k)) if c else None
    ok = (lambda n: n != s and not (frag and n in frag))
    objs = +Counter({n: c.get("される:" + p, 0) for n, c in w.items() if ok(n)})
    obls = +Counter({n: c.get("で・に:" + p, 0) for n, c in w.items() if ok(n)})
    o = None if "目的語" in drop else top(objs)
    b = None if "補語" in drop else top(obls)
    mods = Counter({a.split(":", 1)[1]: v for a, v in w[s].items() if a.startswith("修飾:")})
    m_of = s
    if not mods and o:
        mods = Counter({a.split(":", 1)[1]: v for a, v in w[o].items() if a.startswith("修飾:")})
        m_of = o
    m = None if "修飾" in drop else top(mods)

    def pt(n, role, dflt):
        return top(part.get((n, role), Counter())) or dflt
    head, surf = [], {}
    mform = (top(form["修飾:" + m]) or m) if m else None
    if m and m_of == s:
        head.append(mform)
    surf["主語"] = text.get(s, s)
    head.append(surf["主語"] + pt(s, "する", "が"))
    if b:
        surf["補語"] = text.get(b, b)
        head.append(surf["補語"] + pt(b, "で・に", "に"))
    if o:
        if m and m_of == o:
            head.append(mform)
        surf["目的語"] = text.get(o, o)
        head.append(surf["目的語"] + "を")
    pf = top(form["する:" + p]) or p
    head.append(pf + "。")
    pick = dict(主語=s, 述語=p, 目的語=o, 補語=b, 修飾=m)
    return "".join(head), pick, surf, mform, m_of


def finds(doc, surface):
    """表記が surface で終わる名詞の語（複合語は最後の語）。同じ表記が 2 つあればどちらも。"""
    return [t for t in doc if t.pos_ in ("NOUN", "PROPN", "PRON") and t.text and surface.endswith(t.text)]


def check(nlp, line, pick, surf, m_of):
    """逆回しのチェック: 枠ごとに、意図した係りが戻ったか。"""
    doc = nlp(line)
    res = {}
    roots = [t for t in doc if t.dep_ == "ROOT"]
    preds = [t for t in doc if t.lemma_ == pick["述語"] and t.pos_ in ("VERB", "ADJ", "NOUN", "AUX")]
    pred = next((t for t in preds if t.dep_ == "ROOT"), preds[-1] if preds else None)
    res["述語"] = bool(pred is not None and len(roots) == 1 and pred.i == roots[0].i)     # 根が 2 つ = 行が 2 つに割れた
    sjs = [t for t in finds(doc, surf["主語"]) if pred is not None and t.dep_ in ("nsubj", "dislocated") and t.head.i == pred.i]
    res["主語"] = bool(sjs)
    for slot, deps in (("目的語", ("obj",)), ("補語", ("obl",))):
        if pick.get(slot):
            res[slot] = any(pred is not None and t.dep_ in deps and t.head.i == pred.i for t in finds(doc, surf[slot]))
    if pick.get("修飾"):
        tg = finds(doc, surf["主語"] if m_of == pick["主語"] else surf.get("目的語", "\0"))
        res["修飾"] = any(t.lemma_ == pick["修飾"] and t.dep_ in ("acl", "amod") and t.head.i in {x.i for x in tg} for t in doc)
    return res, doc


def meaning(pick, m_of, wi, G, propn):
    """意味のチェック（違和感）: 枠の組が、ほかの作品にもあるか。固有名詞は組ではなく枠（役 + 述語）だけを見る。"""
    pair, frame, own_pair, own_frame = G["pair"], G["frame"], G["own_pair"][wi], G["own_frame"][wi]
    res = {}
    for slot, role in (("主語", "する:"), ("目的語", "される:"), ("補語", "で・に:")):
        n = pick.get(slot)
        if not n:
            continue
        a = role + pick["述語"]
        fr = frame[a] - own_frame[a]
        pr = pair[(n, a)] - own_pair[(n, a)]
        res[slot] = fr >= 3 and (n in propn or pr >= 1)
    if pick.get("修飾"):
        a = "修飾:" + pick["修飾"]
        n = m_of
        res["修飾"] = (frame[a] - own_frame[a]) >= 3 and (n in propn or pair[(n, a)] - own_pair[(n, a)] >= 1)
    return res


def frag_nouns(w):
    cnt = Counter(t["lemma"] for ln in w["lines"] for t in ln["toks"] if t["pos"] in ("NOUN", "PROPN"))
    return {n for n in set(cnt) if KANA1.match(n) or cnt[n] < 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_stats", type=int, default=2000)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max_parse", type=int, default=6)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("ja_ginza")
    works = json.load(open(a.cache, encoding="utf-8"))
    Q.prep(works)
    tr, roots = O.flow_board(works)
    frags = [frag_nouns(w) for w in works]
    from collections import defaultdict
    G = dict(pair=Counter(), frame=Counter(), own_pair=[], own_frame=[])
    propn = set()
    for w in works:
        op, of = Counter(), Counter()
        for la in w["_attrs"]:
            for n, at, f, pp in la:
                op[(n, at)] += 1
                of[at] += 1
        for ln in w["lines"]:
            for t in ln["toks"]:
                if t["pos"] in ("PROPN", "PRON"):
                    propn.add(t["lemma"])
        G["pair"].update(op)
        G["frame"].update(of)
        G["own_pair"].append(op)
        G["own_frame"].append(of)
    rng = random.Random(a.seed)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(10, len(w["lines"]) - 10)
             if any(t["dep"] == "nsubj" for t in w["lines"][li]["toks"])]
    rng.shuffle(sites)
    sites = sites[:a.n_stats]
    res = {k: Counter() for k in ("I27q（チェックなし）", "チェックで選び直す（前さばきなし）", "前さばき + チェックで直す", "前さばき + 文法と意味のチェックで直す")}
    slot_ok = {k: Counter() for k in res}
    slot_n = {k: Counter() for k in res}
    slot_ok2 = {k: Counter() for k in res}
    slot_n2 = {k: Counter() for k in res}
    examples = []
    for si, (wi, li) in enumerate(sites):
        w = works[wi]
        lines = w["lines"]
        toks = lines[li]["toks"]
        ref = M.content_lemmas(toks)
        nb = [M.content_lemmas(lines[lj]["toks"]) for lj in (li - 1, li + 1)]
        prev_p = roots[wi][li - 1]
        sub = [(x, y) for x, y in ((prev_p, roots[wi][li]), (roots[wi][li], roots[wi][li + 1])) if x and y]
        for x, y in sub:
            tr[x][y] -= 1
        avoid = O.pairs_in(lines[li - 1]["toks"]) | O.pairs_in(lines[li + 1]["toks"])
        board = Q.book_board(w, li, "直線")
        ex = {}
        for name in res:
            frag = frags[wi] if name.startswith("前さばき") else None
            pairs = ranked_pairs(board, avoid, prev_p, tr, frag)
            if not pairs:
                continue
            tries, final = 0, None
            log = []
            if name == "I27q（チェックなし）":
                s, p = pairs[0]
                line, pick, surf, mform, m_of = render(board, s, p)
                chk, _ = check(nlp, line, pick, surf, m_of)
                final = (line, pick, mform, chk, m_of)
                tries = 1
            else:
                pi, drop = 0, set()
                best = None
                while tries < a.max_parse and pi < len(pairs):
                    s, p = pairs[pi]
                    line, pick, surf, mform, m_of = render(board, s, p, drop, frag)
                    chk, _ = check(nlp, line, pick, surf, m_of)
                    if "意味" in name:
                        mn = meaning(pick, m_of, wi, G, propn)
                        chk = {k: v and mn.get(k, True) for k, v in chk.items()}
                    tries += 1
                    log.append((line, {k: v for k, v in chk.items() if not v}))
                    sc = sum(chk.values()) / len(chk)
                    if best is None or sc > best[0]:
                        best = (sc, (line, pick, mform, chk, m_of))
                    if all(chk.values()):
                        break
                    if not chk["主語"] or not chk["述語"]:      # 主語・述語が崩れたら、次の組へ
                        pi, drop = pi + 1, set()
                    else:                                         # 修飾・補語・目的語が崩れたら、その枠を外す
                        drop |= {k for k, v in chk.items() if not v}
                final = best[1]
            line, pick, mform, chk, final_mof = final
            g = Counter(x for x in pick.values() if x)
            r = res[name]
            r["n"] += 1
            r["全部戻った"] += all(chk.values())
            r["解析し直した回数"] += tries
            r["F1"] += M.f1(ref, g)
            r["隣F1"] += max(M.f1(nb[0], g), M.f1(nb[1], g))
            r["主語が切れ端"] += pick["主語"] in frags[wi]
            mn = meaning(pick, None if not pick.get("修飾") else (pick["主語"] if final_mof == pick["主語"] else pick.get("目的語")), wi, G, propn)
            r["意味も通る"] += all(chk.values()) and all(mn.values())
            for k, v in mn.items():
                slot_n2[name][k] += 1
                slot_ok2[name][k] += v
            for k, v in chk.items():
                slot_n[name][k] += 1
                slot_ok[name][k] += v
            ex[name] = (line, chk, log)
        for x, y in sub:
            tr[x][y] += 1
        if len(examples) < a.n and len(ex) == len(res) and not all(ex["I27q（チェックなし）"][1].values()):
            examples.append((wi, li, ex))
        if si % 200 == 0:
            print(si, flush=True)
    rec = dict(exp="I27r", n=len(sites), 結果={})
    for name, r in res.items():
        rec["結果"][name] = dict(n=r["n"], **{m: round(r[m] / max(1, r["n"]), 4) for m in r if m != "n"},
                               枠ごとに戻った割合={k: round(slot_ok[name][k] / slot_n[name][k], 4) for k in slot_n[name]},
                               枠ごとに意味が通る割合={k: round(slot_ok2[name][k] / slot_n2[name][k], 4) for k in slot_n2[name]})
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = ["# I27r 作った行を解析し直して確かめ、崩れた枠を直す（非公開）", "",
           "チェックなしで崩れた行だけを選んだ。（元）が抜いた元の行。[崩れた枠] は解析し直して係りが戻らなかった枠。", ""]
    for j, (wi, li, ex) in enumerate(examples, 1):
        lines = works[wi]["lines"]
        out += [f"## {j}. 作品 {works[wi]['name']} {li} 行目", ""]
        out.append(f"- 　 {lines[li - 1]['text']}")
        out.append(f"- （元）{lines[li]['text']}")
        out.append(f"- 　 {lines[li + 1]['text']}")
        out.append("")
        for name, (line, chk, log) in ex.items():
            bad = [k for k, v in chk.items() if not v]
            out.append(f"- ▶ {name}: **{line}**　[崩れた枠: {'、'.join(bad) or 'なし'}]")
            if log and len(log) > 1:
                out.append("  - 直した道筋: " + " → ".join(f"{x}（{'、'.join(b) or 'なし'}）" for x, b in log))
        out.append("")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
