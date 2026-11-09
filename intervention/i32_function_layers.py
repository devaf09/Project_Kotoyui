"""介入の研究 実験I32: 機能語の層（つなぎ・述部の形）を板にして、展開で選ぶ

持ち主（2026-11-09）: 「日本語の接続部分の学習が圧倒的に足りていないか、たぶんない。作らないといけないかも」
I27x2 の展開は、中の節がいつも て形、最後がいつも「いちばん多い表記」だった。
I32 で足す板（どれもほかの作品で数え、自分の作品は引く。手の一覧は使わない）:
  つなぎの板   行の中で 述語 p が 述語 q に係る（advcl）所の、p のあとの機能語の列の末尾（て・ながら・から・けど・ば・
               たら・連用 など）。("繋", p, q, つなぎ)・("繋p", p, つなぎ)・("繋g", つなぎ) と、表記 ("繋形", p, つなぎ, 表記)。
  述部の形の板 根の述語の表記（「見ていた」「見ます」）と、その形の印（た・丁寧・否定・相）。
               期待は、同じ作品の前後の行（地の文か台詞かを分けて）で印が出る割合。
選び方（気づきの器・反応の板は I27x2 と同じ形）:
  中の節のつなぎ  Witten-Bell で (p, q) → p → 全体 と混ぜた確からしさ。反応の板の鍵 ("繋", p, つなぎ) で暗くする。
  最後の形        表記の確からしさ × 前後の行の印の期待との一致。反応の板の鍵 ("終", p, 表記)。
I32c 助詞を述語と一緒に選ぶ（持ち主の 2・案 3）:
  補語の助詞を、名詞ごとの板 ("助", 名詞, 役, 助詞)（場の板）だけでなく、述語ごとの期待 ("助述", 役:述語, 助詞)（ほかの作品）
  と一緒に選ぶ。点 = log P(助詞|名詞) + log P(助詞|述語) − log P(助詞)（どちらも Witten-Bell で全体に混ぜる）。
物差し: I27x2 と同じもの（全部戻った・意味も通る など）に、
  つなぎの種類（いろいろ使えるか、元の行の型の割合との差）と、最後の形の印が元の行と合う割合を足す。
Web 小説は _wn。結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i32_function_layers.py --cache <解析済み JSON> --stats <jsonl> --out <例の md>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27m_board_line as M  # noqa: E402
import i27o_board_line2 as O  # noqa: E402
import i27q_book_insert as Q  # noqa: E402
import i27s_fused_generate as S  # noqa: E402
import i27x_chain_clauses as X  # noqa: E402
import i27x2_clause_meaning as X2  # noqa: E402
from board_ops import Board, bundle, light  # noqa: E402

PRED = ("VERB", "ADJ")
FUNDEP = ("aux", "mark", "case", "cop", "fixed", "cc")
CONNPOS = ("SCONJ", "ADP", "PART", "CCONJ")
VOICE = ("する", "れる", "られる", "せる", "させる")
# 型は物差しのためだけ（選ぶときには使わない）
TYPE = {"て": "順", "連用": "順", "てから": "順", "に": "目的", "ため": "理由", "から": "理由", "ので": "理由",
        "し": "並べ", "たり": "並べ", "が": "逆接", "けど": "逆接", "けれど": "逆接", "のに": "逆接", "ても": "逆接",
        "でも": "逆接", "ながらも": "逆接", "ながら": "同時", "つつ": "同時", "と": "条件", "ば": "条件",
        "たら": "条件", "なら": "条件", "ず": "否定"}
MARKS = ("た", "丁寧", "否定", "相")


def ctype(c):
    return TYPE.get(c, "他")


def chain(kids, p):
    """述語 p のすぐ後ろに続く機能語の列。p の子と、列の語に fixed で付く語（「て」+「いる」）も含める。"""
    cand = {}
    stack = [p["i"]]
    while stack:
        x = stack.pop()
        for k in kids[x]:
            if k["i"] > p["i"] and k["dep"] in FUNDEP and k["pos"] != "PUNCT" and k["i"] not in cand:
                if x == p["i"] or k["dep"] == "fixed":
                    cand[k["i"]] = k
                    stack.append(k["i"])
    out, nxt = [], p["i"] + 1
    while nxt in cand:
        out.append(cand[nxt])
        nxt += 1
    return out


def layers(kids, p):
    """述語 p の層: つなぎ（列の末尾の助詞・接続助詞）、形（その前の助動詞の原形の列）、表記。"""
    ch = chain(kids, p)
    i = 0
    while i < len(ch) and ch[i]["lemma"] in VOICE and ch[i]["pos"] == "AUX":
        i += 1
    rest = ch[i:]
    j = len(rest)
    while j > 0 and rest[j - 1]["pos"] in CONNPOS:
        j -= 1
    conn = "".join(k["text"] for k in rest[j:])
    form = rest[:j]
    if not conn and form and form[-1]["text"] in ("たら", "だら", "なら", "たり", "だり", "ずに", "ず", "に", "ば"):
        conn, form = form[-1]["text"], form[:-1]
    lem = [k["lemma"] for k in form]
    marks = {"た": "た" in lem, "丁寧": "ます" in lem or "です" in lem, "否定": "ない" in lem or "ぬ" in lem, "相": "てる" in lem or ("て" in lem and ("いる" in lem or "おる" in lem))}
    surf = p["text"] + "".join(k["text"] for k in ch)
    return conn or "連用", marks, surf


def kids_of(T):
    kids = defaultdict(list)
    for t in T:
        if t["head"] != t["i"]:
            kids[t["head"]].append(t)
    return kids


def isq(text):
    return text.lstrip("　 ").startswith(("「", "『", "（"))


def layer_rows(w):
    """行ごとの機能語の板と、行ごとの印（根の述語の形）。"""
    rows, marks = [], []
    for ln in w["lines"]:
        b = Board()
        T = ln["toks"]
        kids = kids_of(T)
        byi = {t["i"]: t for t in T}
        mk = None
        for t in T:
            if t["pos"] not in PRED:
                continue
            if t["dep"] == "advcl" and byi.get(t["head"], {}).get("pos") in PRED:
                conn, _, surf = layers(kids, t)
                q = byi[t["head"]]["lemma"]
                b[("繋", t["lemma"], q, conn)] += 1
                b[("繋計", t["lemma"], q)] += 1
                b[("繋p", t["lemma"], conn)] += 1
                b[("繋p計", t["lemma"])] += 1
                b[("繋g", conn)] += 1
                b[("繋g計",)] += 1
                b[("繋形", t["lemma"], conn, surf)] += 1
            elif t["dep"] == "ROOT":
                conn, m, surf = layers(kids, t)
                if conn == "連用":           # 言い切り（後ろに助詞の付かない根）
                    b[("終形", t["lemma"], surf)] += 1
                    for k in MARKS:
                        if m[k]:
                            b[("終印", k)] += 1
                    b[("終印計",)] += 1
                if mk is None:
                    mk = m
        rows.append(b)
        marks.append((isq(ln["text"]), mk))
    return rows, marks


class Layers:
    """1 か所ぶんの機能語の板の読み手。LG は全作品、lown は自分の作品（引く）。"""

    def __init__(self, LG, lown, idx, ctx_marks):
        self.LG, self.lown, self.idx, self.ctx = LG, lown, idx, ctx_marks

    def n(self, key):
        return light(self.LG, key, minus=self.lown)

    def conn_dist(self, p, q):
        labels = self.idx["conns"]
        g = self.n(("繋g計",))
        dist = {c: (self.n(("繋g", c)) + 0.5) / (g + 0.5 * len(labels)) for c in labels}
        for keys, tot in (([("繋p", p, c) for c in labels], ("繋p計", p)), ([("繋", p, q, c) for c in labels], ("繋計", p, q))):
            cnt = {c: self.n(k) for c, k in zip(labels, keys)}
            n = self.n(tot)
            t = sum(1 for v in cnt.values() if v > 0)
            if n > 0:
                lam = n / (n + t)
                dist = {c: lam * cnt[c] / n + (1 - lam) * dist[c] for c in labels}
        return dist

    def conn_surfs(self, p):
        """p のつなぎごとの表記（ほかの作品の表記も、自分の作品の表記も、語の書き方として使う）。"""
        out = {}
        for c, sc in self.idx["surf"].get(p, {}).items():
            out[c] = max(sc, key=lambda s: (sc[s], s))
        return out

    def end_score(self, p):
        """最後の表記の候補と点: log 確からしさ + Σ log 印の期待との一致。"""
        cands = self.idx["end"].get(p, {})
        if not cands:
            return []
        tot = sum(cands.values())
        out = []
        for s, v in cands.items():
            m = self.idx["end_marks"][s]
            sc = math.log(v / tot)
            for k in MARKS:
                r = self.ctx[k]
                sc += math.log(r if m[k] else 1 - r)
            out.append((sc, s))
        out.sort(reverse=True)
        return out


def build_index(LG):
    conns, surf, end, end_marks = set(), defaultdict(lambda: defaultdict(Counter)), defaultdict(Counter), {}
    for k, v in LG.items():
        if k[0] == "繋g":
            conns.add(k[1])
        elif k[0] == "繋形":
            surf[k[1]][k[2]][k[3]] += v
        elif k[0] == "終形":
            end[k[1]][k[2]] += v
    return dict(conns=sorted(conns), surf=surf, end=end, end_marks=end_marks)


def ctx_marks(marks, li, gl, win=10):
    """前後 win 行の地の文の印の割合（同じ作品・文脈として見える行）。全体の割合を 1 行分の重みで混ぜる。"""
    c, n = Counter(), 0
    for lj in range(max(0, li - win), min(len(marks), li + win + 1)):
        if lj == li:
            continue
        q, m = marks[lj]
        if q or m is None:
            continue
        n += 1
        for k in MARKS:
            c[k] += m[k]
    return {k: min(max((c[k] + gl[k]) / (n + 1), 1e-3), 1 - 1e-3) for k in MARKS}


# ---- 展開に入れる -------------------------------------------------------------

def choose_layers(ch, clauses, RB):
    """中の節のつなぎと最後の表記を選ぶ。点 = log 確からしさ − 反応の板。"""
    L = ch.L
    pen = 0.0
    for j, c in enumerate(clauses):
        p = c["述語"]
        if j < len(clauses) - 1:
            q = clauses[j + 1]["述語"]
            dist = L.conn_dist(p, q)
            if getattr(ch, "use_grain", False):
                dist = ch.grain_dist(ch, clauses, j)          # I33: 述部グレイン（回帰＋中身の語）
            surfs = L.conn_surfs(p)
            best = None
            for cn, s in surfs.items():
                v = math.log(dist.get(cn, 1e-6)) - light(RB, ("繋", p, cn))
                if best is None or v > best[0]:
                    best = (v, cn, s)
            if best is None:
                c["つなぎ"], c["表記"] = "て", ch.te(p)
            else:
                c["つなぎ"], c["表記"] = best[1], best[2]
                pen += light(RB, ("繋", p, best[1]))
        else:
            es = [(v - light(RB, ("終", p, s)), s) for v, s in L.end_score(p)]
            if es:
                v, s = max(es)
                c["つなぎ"], c["表記"] = "終", s
                pen += light(RB, ("終", p, s))
            else:
                c["つなぎ"], c["表記"] = "終", None
    return pen


def propose_layers(ch, first, RB, use_chain=True):
    sc, cl = X2.propose2(ch, first, RB, use_chain)
    cl = [dict(c) for c in cl]
    pen = choose_layers(ch, cl, RB)
    return sc - pen, cl


def render_layers(ch, clauses):
    st = ch.st
    line, surf = render2_particle(ch, clauses[0])
    p0 = clauses[0]["述語"]
    pf = max(st.form["する:" + p0], key=lambda x: (st.form["する:" + p0][x], x)) if st.form["する:" + p0] else p0
    head = line[:-1]
    if head.endswith(pf):
        head = head[: -len(pf)]
    else:
        pf = None
    if len(clauses) == 1:
        s0 = clauses[0].get("表記") or pf or p0
        return head + s0 + "。", [surf]
    parts = [head + (clauses[0].get("表記") or ch.te(p0))]
    surfs = [surf]
    for j, c in enumerate(clauses[1:], 1):
        sf, seg = {}, []
        if c["補語"]:
            sf["補語"] = st.BB_text.get(c["補語"], c["補語"])
            seg.append(sf["補語"] + pick_particle(ch, c["補語"], c["述語"]))
        if c["目的語"]:
            sf["目的語"] = st.BB_text.get(c["目的語"], c["目的語"])
            seg.append(sf["目的語"] + "を")
        last = j == len(clauses) - 1
        if last:
            fm = st.form["する:" + c["述語"]]
            seg.append(c.get("表記") or (max(fm, key=lambda x: (fm[x], x)) if fm else c["述語"]))
        else:
            seg.append(c.get("表記") or ch.te(c["述語"]))
        parts.append("".join(seg))
        surfs.append(sf)
    return "、".join(parts) + "。", surfs


def keys_layers(clauses):
    ks = _orig_keys(clauses)
    n = len(clauses)
    for j, c in enumerate(clauses):
        if c.get("つなぎ") and c["つなぎ"] != "終":
            ks[(j, "述語")] = ("繋", c["述語"], c["つなぎ"])      # 係りが戻らなければ、そのつなぎを暗くする
        elif c.get("表記") and j == n - 1:
            ks[(j, "述語")] = ("終", c["述語"], c["表記"])
    return ks


def particle_rows(w):
    rows = []
    for la in w["_attrs"]:
        b = Board()
        for n, a, f, p in la:
            if p and a.startswith("で・に:"):
                b[("助述", a, p)] += 1
                b[("助述計", a)] += 1
                b[("助g", p)] += 1
                b[("助g計",)] += 1
        rows.append(b)
    return rows


def pick_particle(ch, n, pred, dflt="に"):
    """補語 n の助詞を述語 pred と一緒に選ぶ。"""
    st = ch.st
    if not getattr(ch, "use_particle", False):
        part = st.part.get((n, "で・に"), Counter())
        return max(part, key=lambda x: (part[x], x)) if part else dflt
    PG, pown = ch.PG, ch.pown
    labels = ch.plabels
    g = light(PG, ("助g計",), minus=pown)
    pg = {x: (light(PG, ("助g", x), minus=pown) + 0.5) / (g + 0.5 * len(labels)) for x in labels}

    def wb(cnt, lower):
        tot = sum(cnt.values())
        t = sum(1 for v in cnt.values() if v > 0)
        if tot <= 0:
            return lower
        lam = tot / (tot + t)
        return {x: lam * cnt.get(x, 0) / tot + (1 - lam) * lower[x] for x in labels}
    pn = wb(st.part.get((n, "で・に"), Counter()), pg)
    pp = wb({x: light(PG, ("助述", "で・に:" + pred, x), minus=pown) for x in labels}, pg)
    seen = set(st.part.get((n, "で・に"), Counter())) | {x for x in labels if light(PG, ("助述", "で・に:" + pred, x), minus=pown) > 0}
    if not seen:
        return dflt
    return max(seen, key=lambda x: (math.log(pn[x]) + math.log(pp[x]) - math.log(pg[x]), x))


def render2_particle(ch, pick):
    """S.render2 と同じ並び。補語の助詞だけ pick_particle で選ぶ。"""
    st = ch.st

    def top(c):
        return max(c, key=lambda k: (c[k], k)) if c else None

    def pt(n, role, dflt):
        return top(st.part.get((n, role), Counter())) or dflt

    def mf(m):
        return top(st.form["修飾:" + m]) or m
    text = st.BB_text
    s, p, o, b = pick["主語"], pick["述語"], pick["目的語"], pick["補語"]
    head, surf = [], {}
    if pick["修飾"]:
        head.append(mf(pick["修飾"]))
    surf["主語"] = text.get(s, s)
    head.append(surf["主語"] + pt(s, "する", "が"))
    if b:
        surf["補語"] = text.get(b, b)
        head.append(surf["補語"] + pick_particle(ch, b, p))
    if o:
        if pick["目的語の修飾"]:
            head.append(mf(pick["目的語の修飾"]))
        surf["目的語"] = text.get(o, o)
        head.append(surf["目的語"] + "を")
    head.append((top(st.form["する:" + p]) or p) + "。")
    return "".join(head), surf


_orig = dict(propose=X.propose, render=X.render_chain, keys=X.keys_of)
_orig_keys = X.keys_of


def set_mode(mode):
    if mode in ("I32", "I32c", "I33"):
        X.propose, X.render_chain, X.keys_of = propose_layers, render_layers, keys_layers
    elif mode == "I27x2":
        X.propose, X.render_chain, X.keys_of = X2.propose2, _orig["render"], _orig["keys"]
    else:
        X.propose, X.render_chain, X.keys_of = _orig["propose"], _orig["render"], _orig["keys"]


def read_marks(doc):
    """作った行・元の行の最後の言い切りの印（GiNZA の解析し直しから）。"""
    roots = [t for t in doc if t.dep_ == "ROOT"]
    if not roots:
        return None
    r = roots[-1]
    lem = []
    for t in doc[r.i + 1:]:
        if t.head.i != r.i or t.dep_ not in FUNDEP:
            break
        lem.append(t.lemma_)
    return {"た": "た" in lem, "丁寧": "ます" in lem or "です" in lem, "否定": "ない" in lem or "ぬ" in lem, "相": "てる" in lem or ("て" in lem and ("いる" in lem or "おる" in lem))}


def gen_conns(doc):
    out = []
    for t in doc:
        if t.dep_ == "advcl" and t.pos_ in PRED:
            ch = sorted([k for k in t.children if k.i > t.i and k.dep_ in FUNDEP and k.pos_ != "PUNCT"], key=lambda k: k.i)
            j = len(ch)
            while j > 0 and ch[j - 1].pos_ in CONNPOS:
                j -= 1
            out.append("".join(k.text for k in ch[j:]) or "連用")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_stats", type=int, default=2000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--only", default="", help="この名前を含む組み方だけ回す（例: 回数 2）")
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
    jown = [bundle(list(enumerate(X2.joint_rows(w)))) for w in works]
    JG = Board()
    for b in jown:
        JG.update(b)
    lrows, lmarks, lown = [], [], []
    for w in works:
        r, m = layer_rows(w)
        lrows.append(r)
        lmarks.append(m)
        lown.append(bundle(list(enumerate(r))))
    LG = Board()
    for b in lown:
        LG.update(b)
    pown = [bundle(list(enumerate(particle_rows(w)))) for w in works]
    PG = Board()
    for b in pown:
        PG.update(b)
    plabels = sorted({k[1] for k in PG if k[0] == "助g"} | {p_ for w in works for la in w["_attrs"] for _, a_, _, p_ in la if p_ and a_.startswith("で・に:")})
    idx = build_index(LG)
    for s in {s for p in idx["end"] for s in idx["end"][p]}:
        idx["end_marks"][s] = None
    # 表記ごとの印（終形の表記は作品の中で見えた形なので、その時の印を引いておく）
    for w in works:
        for ln in w["lines"]:
            T = ln["toks"]
            kids = kids_of(T)
            for t in T:
                if t["dep"] == "ROOT" and t["pos"] in PRED:
                    conn, m, surf = layers(kids, t)
                    if conn == "連用":
                        idx["end_marks"][surf] = m
    gl = {k: light(LG, ("終印", k)) / max(1, light(LG, ("終印計",))) for k in MARKS}
    grains = None
    if not a.only or "I33" in a.only:
        import i33_predicate_grain as G33
        grains = G33.train_folds(works)
    propn = {t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] in ("PROPN", "PRON")}
    propn_only = {t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] == "PROPN"}
    rng = random.Random(a.seed)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(10, len(w["lines"]) - 10)
             if any(t["dep"] == "nsubj" for t in w["lines"][li]["toks"])]
    rng.shuffle(sites)
    sites = sites[:a.n_stats]
    names = [("I27x2・回数 2", "I27x2", 2), ("I32 つなぎと述部の形・回数 2", "I32", 2),
             ("I27x2・回数 6", "I27x2", 6), ("I32 つなぎと述部の形・回数 6", "I32", 6),
             ("I32c つなぎ・形・助詞・回数 2", "I32c", 2), ("I32c つなぎ・形・助詞・回数 6", "I32c", 6),
             ("I33 述部グレイン・回数 2", "I33", 2), ("I33 述部グレイン・回数 6", "I33", 6)]
    if a.only:
        names = [x for x in names if a.only in x[0]]
    res = {nm: Counter() for nm, _, _ in names}
    ctypes = {nm: Counter() for nm, _, _ in names}
    conns_used = {nm: Counter() for nm, _, _ in names}
    orig_types = Counter()
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
        cm = ctx_marks(lmarks[wi], li, gl)
        if grains is not None:
            import i33_predicate_grain as G33
            ch.grain_dist = G33.make_grain_dist(grains[wi % G33.FOLDS], lines, li, lmarks[wi])
        ch.L = Layers(LG, lown[wi], idx, cm)
        ch.PG, ch.pown, ch.plabels = PG, pown[wi], plabels
        pairs = S.pair_scores(board, avoid, prev_p, tr)
        om = lmarks[wi][li][1]
        T = lines[li]["toks"]
        kids, byi = kids_of(T), {t["i"]: t for t in T}
        for t in T:
            if t["pos"] in PRED and t["dep"] == "advcl" and byi.get(t["head"], {}).get("pos") in PRED:
                orig_types[ctype(layers(kids, t)[0])] += 1
        ex = {}
        for nm, mode, budget in names:
            if not pairs:
                continue
            set_mode(mode)
            ch.use_particle = mode in ("I32c", "I33")
            ch.use_grain = mode == "I33"
            (line, cl, chk, doc), tries = X.ring(nlp, ch, pairs, budget, True)
            mn = X2.meaning_all(st, cl, propn)
            mn2 = X2.meaning_all(st, cl, propn_only)
            g = Counter(x for c in cl for k, x in c.items() if x and k in ("主語", "述語", "目的語", "補語", "修飾", "目的語の修飾")
                        and not (k == "主語" and c is not cl[0]))
            r = res[nm]
            r["n"] += 1
            r["全部戻った"] += all(chk.values())
            r["意味も通る（代名詞も裏づけ）"] += all(chk.values()) and all(mn2.values())
            r["解析し直した回数"] += tries
            r["F1"] += M.f1(ref, g)
            r["内容語"] += S.n_content(doc)
            r["文字数"] += len(line)
            r["節の数"] += len(cl)
            gm = read_marks(doc)
            if om is not None and gm is not None:
                r["印を比べた"] += 1
                for k in MARKS:
                    r[f"印が元と合う:{k}"] += gm[k] == om[k]
                r["印が元と全部合う"] += all(gm[k] == om[k] for k in MARKS)
            for c in gen_conns(doc):
                ctypes[nm][ctype(c)] += 1
                conns_used[nm][c] += 1
            ex[nm] = (line, cl, [f"{j + 1}:{k}" for (j, k), v in chk.items() if not v] + [f"{j + 1}:{k}（意味）" for (j, k), v in mn2.items() if not v])
        set_mode("I27x2")
        for x, y in sub:
            tr[x][y] += 1
        if len(examples) < a.n and len(ex) == len(names):
            examples.append((wi, li, ex))
        if si % 200 == 0:
            print(si, flush=True)
    ot = sum(orig_types.values())
    rec = dict(exp="I32", n=len(sites), 元の行のつなぎの型={k: round(v / ot, 3) for k, v in orig_types.most_common()}, 結果={})
    for nm, r in res.items():
        tot = sum(ctypes[nm].values())
        dist = {k: round(v / tot, 3) for k, v in ctypes[nm].most_common()} if tot else {}
        tv = 0.5 * sum(abs(dist.get(k, 0) - orig_types[k] / ot) for k in set(dist) | set(orig_types)) if tot and ot else None
        rec["結果"][nm] = dict(n=r["n"], **{m: round(r[m] / max(1, r["n"]), 4) for m in r if m != "n" and not m.startswith("印が")},
                             **{m: round(r[m] / max(1, r["印を比べた"]), 4) for m in r if m.startswith("印が")},
                             つなぎの型=dist, 型の割合の差_元の行と=round(tv, 3) if tv is not None else None,
                             つなぎの種類=len(conns_used[nm]), よく使ったつなぎ=dict(conns_used[nm].most_common(12)))
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = ["# I32 つなぎと述部の形を選んだ行（非公開）", "",
           "（元）が抜いた元の行。[崩れ] は振り返りで戻らなかった節:枠と、意味の物差し（代名詞も裏づけ）で通らなかった節:枠。",
           "選んだ層: 節ごとのつなぎ（最後は 終）と表記。", ""]
    for j, (wi, li, ex) in enumerate(examples, 1):
        lines = works[wi]["lines"]
        out += [f"## {j}. 作品 {works[wi]['name']} {li} 行目", "",
                f"- 　 {lines[li - 1]['text']}", f"- （元）{lines[li]['text']}", f"- 　 {lines[li + 1]['text']}", ""]
        orig = "・".join(f"{n_}|{at}" for n_, at, f_, p_ in works[wi]["_attrs"][li]) or "—"
        out.append(f"- （元を縮めた形）{orig}")
        for nm, (line, cl, bad) in ex.items():
            comp = " ／ ".join("・".join(f"{k}={v}" for k, v in c.items() if v and k not in ("表記",) and not (k == "主語" and c is not cl[0])) for c in cl)
            out.append(f"- ▶ {nm}: **{line}**　[崩れ: {'、'.join(bad) or 'なし'}]")
            out.append(f"  - 縮めた形（選んだ枠）: {comp}")
        out.append("")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
