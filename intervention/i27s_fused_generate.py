"""介入の研究 実験I27s: 板の 4 つの操作（board_ops）と、融合した反応（差分）だけで行を作り、確かめる

持ち主（2026-11-08）: 「デジタル判断の部分を減らして、板の操作をモジュール化」「きずきや違和感、発見板で共通する部分は融合でまとめてみる」
I27r の手のきまり（切れ端の名詞の除外・意味の閾値 ≥3 と ≥1・崩れたら外す／次の組へ・いちばん多く戻った行）を、次の 1 本にまとめる:
  束ねる: 作品の行ごとの板を、直線の坂で 1 枚に（I27q と同じ坂）。ほかの作品の板（共有の板）も行ごとの板を束ねたもの。
  照らす: 枠の明るさ（坂の板）と、ほかの作品での裏づけ（共有の板から自分の作品を引いて読む）を、1 - exp(-v / sat) で 0〜1 に。
  差分:   枠ごとに「期待」と「実際」の差を取る。同じ 1 つの操作で、次の 3 つの反応を持つ。
          意味   期待 1（ほかの作品にもある）と、照らした裏づけ。
          枠     述語がその枠（目的語・補語）を取る割合・名詞が修飾を取る割合（共有の板）と、埋めたか（1 / 0）。
                 → 枠を「外す」のではなく、期待が高い枠は埋める方が差が小さい。
          逆回し 期待 1（意図した係り）と、解析し直して戻ったか（1 / 0）。
  勝者:   点 = Σ 明るさ − Σ |差| の勝者。解析し直しは点の上位 6 行だけ（I27r と同じ 6 回）。
I27r の 4 つ目（前さばき + 文法と意味のチェックで直す）と同じ 2,000 か所で比べる。
物差し: 全部の係りが戻った割合、意味も通る割合（I27r と同じ閾値の物差し）、F1、隣F1、行の内容語の数と文字数。
Web 小説は _wn。例の md と結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i27s_fused_generate.py --cache <解析済み JSON> --stats <jsonl> --out <例の md>
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
from board_ops import Board, bundle, light, react, winner  # noqa: E402

ROLES = (("目的語", "される:"), ("補語", "で・に:"))
SAT_FRAME, SAT_PAIR = 3.0, 1.0        # 照らす器の大きさ（I27r の閾値 ≥3・≥1 の代わり）


def line_rows(w):
    """行ごとの板。鍵: ("板", 名詞, 属性) / ("形", 属性, 表記) / ("助", 名詞, 役, 助詞)。"""
    rows = []
    for la in w["_attrs"]:
        b = Board()
        for n, a, f, p in la:
            b[("板", n, a)] += 1
            if f:
                b[("形", a, f)] += 1
            if p:
                b[("助", n, a.split(":")[0], p)] += 1
        rows.append(b)
    return rows


def frame_rows(w):
    """行ごとの「枠の有り無し」の板（共有の板の材料）。
    ("出", 述語) 行に述語が出た / ("有", 役, 述語) その行で述語が役を取った / ("名", 名詞) / ("飾", 名詞) 名詞が修飾を取った / ("組", 名詞, 属性) / ("枠", 属性)。"""
    rows = []
    for la in w["_attrs"]:
        b = Board()
        preds, has, nouns, modded = set(), set(), set(), set()
        for n, a, f, p in la:
            b[("組", n, a)] += 1
            b[("枠", a)] += 1
            role, x = a.split(":", 1)
            if role in ("する", "される", "で・に"):
                preds.add(x)
                has.add((role, x))
                nouns.add(n)
            elif role == "修飾":
                modded.add(n)
        for x in preds:
            b[("出", x)] += 1
        for r, x in has:
            b[("有", r, x)] += 1
        for n in nouns:
            b[("名", n)] += 1
        for n in modded & nouns:
            b[("飾", n)] += 1
        rows.append(b)
    return rows


def ratio(G, own, num, den):
    d = light(G, den, minus=own)
    return light(G, num, minus=own) / d if d > 0 else 0.0


class Site:
    """1 か所ぶんの板（坂の板・共有の板）と、4 つの操作で読む点。"""

    def __init__(self, BB, G, own, propn):
        self.BB, self.G, self.own, self.propn = BB, G, own, propn
        if role_move is not None:
            self.BB = BB = move_roles(BB)
        self.wb = defaultdict(Counter)
        self.form = defaultdict(Counter)
        self.part = defaultdict(Counter)
        for k, v in BB.items():
            if k[0] == "板":
                self.wb[k[1]][k[2]] = v
            elif k[0] == "形":
                self.form[k[1]][k[2]] = v
            else:
                self.part[(k[1], k[2])][k[3]] = v
        vals = [v for k, v in BB.items() if k[0] == "板"]
        self.sat = sum(vals) / len(vals) if vals else 1.0

    def bright(self, n, a):
        return light(self.BB, ("板", n, a), sat=self.sat)

    def attested(self, n, a):
        fr = light(self.G, ("枠", a), sat=SAT_FRAME, minus=self.own)
        pr = 1.0 if n in self.propn else light(self.G, ("組", n, a), sat=SAT_PAIR, minus=self.own)
        return fr * pr

    def slot(self, n, a, e_fill):
        """埋めた枠: 明るさ − |意味の差| − |枠の差|。e_fill は枠を取る割合（期待）。"""
        return self.bright(n, a) - react(1.0, self.attested(n, a))[0] - react(e_fill, 1.0)[0]


MOVE_ROLES = ("する", "される", "で・に")


def move_roles(BB):
    """I48c: (名詞, 述語) ごとに、する・される・で・に の明るさの合計はそのままに、役の間の分け方を
    「板の明るさ × 助詞の波から読んだ持ち上げ」に比べて置き直す。板に無かった役にも、合計の中から分ける。"""
    tot = defaultdict(dict)
    for k, v in BB.items():
        if k[0] == "板" and ":" in k[2]:
            r, p = k[2].split(":", 1)
            if r in MOVE_ROLES and v > 0:
                tot[(k[1], p)][r] = v
    out = Board()
    for k, v in BB.items():
        out[k] = v
    for (n, p), d in tot.items():
        lift = role_move(n, p)
        if not lift:
            continue
        T = sum(d.values())
        w = {r: (d.get(r, 0.0) + T / len(MOVE_ROLES)) * lift.get(r, 1.0) for r in MOVE_ROLES}   # 板に無い役にも、均しの分を持たせてから掛ける
        z = sum(w.values())
        if z <= 0:
            continue
        for r in MOVE_ROLES:
            out[("板", n, r + ":" + p)] = T * w[r] / z
    return out


def cands(c, k):
    return [x for _, x in sorted(((v, x) for x, v in c.items() if v > 0), reverse=True)[:k]]


role_move = None      # I48c: (名詞, 述語) → {役の頭: 持ち上げ}。場の板の (名詞, 役:述語) の明るさを、役の間で置き直す。None なら使わない
role_particle = None  # I48c: (名詞, 役の頭, 述語) → 助詞。場の板にその名詞・役の助詞が無い時に使う
role_light = None     # I48c: (名詞, 役, 述語) → 0〜1。助詞の波の板から読んだ「その役らしさ」。None なら使わない（前と同じ）


def pair_scores(board, avoid, prev_p, tr, kappa=0.05, avoid_w=0.2, k=6):
    """I27r の ranked_pairs と同じ点（I27o の joint）を、点ごと返す（照らすの材料）。"""
    w = board[0]
    subj_w = {n: sum(c.values()) for n, c in w.items()}
    tot = sum(tr.get(prev_p, {}).values()) if prev_p else 0
    out = []
    for n, c in w.items():
        if n in R.FORMAL2:
            continue
        for a, v in c.items():
            if not a.startswith("する:"):
                continue
            p = a.split(":", 1)[1]
            sc = subj_w[n] * v
            if avoid and (n, p) in avoid:
                sc *= avoid_w
            if tot:
                sc *= kappa + tr[prev_p].get(p, 0) / tot
            out.append((sc, n, p))
    out.sort(reverse=True)
    if role_light is not None:                              # 上の候補だけ、主語らしさ（は・が・も が明るいか）を掛けて並べ直す
        out = sorted(((sc * role_light(n, "主語", p), n, p) for sc, n, p in out[: 4 * k]), reverse=True)
    return out[:k]


def combos(st, s, p, k=3, pb=None):
    """主語 s・述語 p の組で、目的語・補語・主語の修飾・目的語の修飾の候補（無しも 1 つの候補）を全部並べて点をつける。"""
    G, own = st.G, st.own
    e = {r: ratio(G, own, ("有", ro[:-1], p), ("出", p)) for r, ro in ROLES}
    base = (st.bright(s, "する:" + p) if pb is None else pb) - react(1.0, st.attested(s, "する:" + p))[0]
    opts = {}
    for r, ro in ROLES:
        c = Counter({n: v.get(ro + p, 0) for n, v in st.wb.items() if n != s})
        opts[r] = [(None, -react(e[r], 0.0)[0])] + [(n, st.slot(n, ro + p, e[r]) - (react(1.0, role_light(n, r, p))[0] if role_light else 0.0))
                                                    for n in cands(c, k)]

    def mod_opts(n):
        em = ratio(G, own, ("飾", n), ("名", n))
        c = Counter({a.split(":", 1)[1]: v for a, v in st.wb[n].items() if a.startswith("修飾:")})
        return [(None, -react(em, 0.0)[0])] + [(m, st.slot(n, "修飾:" + m, em)) for m in cands(c, k)]
    ms = mod_opts(s)
    out = []
    for (o, so), (b, sb), (m1, sm1) in itertools.product(opts["目的語"], opts["補語"], ms):
        if o is not None and o == b:
            continue
        for m2, sm2 in (mod_opts(o) if o else [(None, 0.0)]):
            out.append((base + so + sb + sm1 + sm2, dict(主語=s, 述語=p, 目的語=o, 補語=b, 修飾=m1, 目的語の修飾=m2)))
    return out


def render2(st, pick):
    def top(c):
        return max(c, key=lambda k: (c[k], k)) if c else None

    def pt(n, role, dflt):
        return top(st.part.get((n, role), Counter())) or (role_particle(n, role, pick["述語"]) if role_particle else None) or dflt

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
        head.append(surf["補語"] + pt(b, "で・に", "に"))
    if o:
        if pick["目的語の修飾"]:
            head.append(mf(pick["目的語の修飾"]))
        surf["目的語"] = text.get(o, o)
        head.append(surf["目的語"] + "を")
    head.append((top(st.form["する:" + p]) or p) + "。")
    return "".join(head), surf


def check2(nlp, line, pick, surf):
    """逆回し（I27r の check に、目的語の修飾を足したもの）。"""
    m_of = pick["主語"]
    p0 = dict(pick)
    p0["修飾"] = pick["修飾"]
    res, doc = R.check(nlp, line, p0, surf, m_of)
    if pick["目的語の修飾"]:
        pred_ok = res["述語"]
        tg = {x.i for x in R.finds(doc, surf["目的語"])}
        res["目的語の修飾"] = pred_ok and any(t.lemma_ == pick["目的語の修飾"] and t.dep_ in ("acl", "amod") and t.head.i in tg for t in doc)
    return res, doc


def meaning2(pick, wi, Gr, propn):
    """物差し用: I27r と同じ閾値の意味のチェックを、目的語の修飾にも。"""
    res = R.meaning(pick, pick["主語"], wi, Gr, propn)
    if pick.get("目的語の修飾"):
        a = "修飾:" + pick["目的語の修飾"]
        n = pick["目的語"]
        res["目的語の修飾"] = (Gr["frame"][a] - Gr["own_frame"][wi][a]) >= 3 and (n in propn or Gr["pair"][(n, a)] - Gr["own_pair"][wi][(n, a)] >= 1)
    return res


def n_content(doc):
    return sum(1 for t in doc if t.pos_ in M.CONTENT)


def slot_keys(pick):
    """反応の板の鍵: 枠と、そこに入れた語。主語と述語は組で 1 つの鍵。"""
    ks = {"主語": ("組", pick["主語"], pick["述語"]), "述語": ("組", pick["主語"], pick["述語"])}
    for k in ("目的語", "補語", "修飾", "目的語の修飾"):
        if pick.get(k):
            ks[k] = (k, pick[k])
    return ks


def fused(nlp, st, pairs, n_parse, feedback=True):
    """勝者: 差分を引いた点の勝者を解析し直す。逆回しの差は「反応の板」に束ね、まだ解析していない行の点からも引く
    （同じ枠・同じ語を使う行が暗くなる。I27r の「外す・次の組へ」の代わり）。最後に解析した行の勝者。"""
    sc = []
    top = pairs[0][0] if pairs and pairs[0][0] > 0 else 1.0
    for v, s, p in pairs:
        sc += combos(st, s, p, pb=1.0 - pow(2.718281828, -v / top))
    seen, parsed = set(), []
    RB = Board()
    while len(parsed) < n_parse:
        live = [(pre - sum(light(RB, k) for k in set(slot_keys(pk).values())), i) for i, (pre, pk) in enumerate(sc) if i not in seen]
        i = winner(live)
        if i is None:
            break
        seen.add(i)
        pre, pick = sc[i]
        line, surf = render2(st, pick)
        if any(line == x[1][0] for x in parsed):
            continue
        chk, doc = check2(nlp, line, pick, surf)
        diffs = {k: react(1.0, float(v))[0] for k, v in chk.items()}
        parsed.append((pre - sum(diffs.values()), (line, pick, chk, doc)))
        if feedback:
            ks = slot_keys(pick)
            RB.update(bundle([(0, Board({ks[k]: d for k, d in diffs.items() if d}))]))
    return winner(parsed), len(parsed)


def baseline_r4(nlp, board, pairs, wi, Gr, propn, frag, max_parse):
    """I27r の 4 つ目（前さばき + 文法と意味のチェックで直す）。"""
    pi, drop, tries, best = 0, set(), 0, None
    while tries < max_parse and pi < len(pairs):
        s, p = pairs[pi]
        line, pick, surf, mform, m_of = R.render(board, s, p, drop, frag)
        chk, doc = R.check(nlp, line, pick, surf, m_of)
        mn = R.meaning(pick, m_of, wi, Gr, propn)
        chk = {k: v and mn.get(k, True) for k, v in chk.items()}
        tries += 1
        f = sum(chk.values()) / len(chk)
        if best is None or f > best[0]:
            best = (f, (line, pick, m_of, doc))
        if all(chk.values()):
            break
        if not chk["主語"] or not chk["述語"]:
            pi, drop = pi + 1, set()
        else:
            drop |= {k for k, v in chk.items() if not v}
    return best[1], tries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_stats", type=int, default=2000)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max_parse", type=int, default=6)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("ja_ginza")
    works = json.load(open(a.cache, encoding="utf-8"))
    Q.prep(works)
    tr, roots = O.flow_board(works)
    frags = [R.frag_nouns(w) for w in works]
    rows = [line_rows(w) for w in works]
    frows = [frame_rows(w) for w in works]
    own = [bundle(list(enumerate(fr))) for fr in frows]
    G = Board()
    for b in own:
        G.update(b)
    propn = {t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] in ("PROPN", "PRON")}
    # I27r の物差し用（同じ数え方）
    Gr = dict(pair=Counter(), frame=Counter(), own_pair=[], own_frame=[])
    for w in works:
        op, of = Counter(), Counter()
        for la in w["_attrs"]:
            for n, at, f, pp in la:
                op[(n, at)] += 1
                of[at] += 1
        Gr["pair"].update(op)
        Gr["frame"].update(of)
        Gr["own_pair"].append(op)
        Gr["own_frame"].append(of)
    rng = random.Random(a.seed)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(10, len(w["lines"]) - 10)
             if any(t["dep"] == "nsubj" for t in w["lines"][li]["toks"])]
    rng.shuffle(sites)
    sites = sites[:a.n_stats]
    names = ("I27r（前さばき + 文法と意味のチェックで直す）", "I27s 差分だけ（上位 6 行を解析）", "I27s 差分 + 反応の板（逆回しの差を戻す）")
    res = {k: Counter() for k in names}
    slot_ok = {k: Counter() for k in names}
    slot_n = {k: Counter() for k in names}
    orig = Counter()
    examples = []
    for si, (wi, li) in enumerate(sites):
        w = works[wi]
        lines = w["lines"]
        toks = lines[li]["toks"]
        ref = M.content_lemmas(toks)
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
        st = Site(BB, G, own[wi], propn)
        st.BB_text = w["_text"]
        board = (st.wb, st.form, st.part, w["_text"])
        ex = {}
        # I27r
        pairs_r = R.ranked_pairs(board, avoid, prev_p, tr, frags[wi])
        pairs_s = pair_scores(board, avoid, prev_p, tr)
        for name in names:
            if name.startswith("I27r"):
                if not pairs_r:
                    continue
                (line, pick, m_of, doc), tries = baseline_r4(nlp, board, pairs_r, wi, Gr, propn, frags[wi], a.max_parse)
                surf = {"主語": w["_text"].get(pick["主語"], pick["主語"])}
                for k in ("目的語", "補語"):
                    if pick.get(k):
                        surf[k] = w["_text"].get(pick[k], pick[k])
                chk, doc = R.check(nlp, line, pick, surf, m_of)
                mn = R.meaning(pick, m_of, wi, Gr, propn)
                pick = dict(pick)
            else:
                if not pairs_s:
                    continue
                (line, pick, chk, doc), tries = fused(nlp, st, pairs_s, a.max_parse, feedback="反応の板" in name)
                mn = meaning2(pick, wi, Gr, propn)
            g = Counter(x for k, x in pick.items() if x)
            r = res[name]
            r["n"] += 1
            r["全部戻った"] += all(chk.values())
            r["意味も通る"] += all(chk.values()) and all(mn.values())
            r["解析し直した回数"] += tries
            r["F1"] += M.f1(ref, g)
            r["隣F1"] += max(M.f1(nb[0], g), M.f1(nb[1], g))
            r["主語が切れ端"] += pick["主語"] in frags[wi]
            r["内容語"] += n_content(doc)
            r["文字数"] += len(line)
            r["埋めた枠"] += sum(1 for x in pick.values() if x)
            for k, v in chk.items():
                slot_n[name][k] += 1
                slot_ok[name][k] += v
            ex[name] = (line, [k for k, v in chk.items() if not v] + [k + "（意味）" for k, v in mn.items() if not v])
        for x, y in sub:
            tr[x][y] += 1
        if len(examples) < a.n and len(ex) == len(names):
            examples.append((wi, li, ex))
        if si % 200 == 0:
            print(si, flush=True)
    rec = dict(exp="I27s", n=len(sites), 元の行=dict(内容語=round(orig["内容語"] / len(sites), 3), 文字数=round(orig["文字数"] / len(sites), 2)), 結果={})
    for name, r in res.items():
        rec["結果"][name] = dict(n=r["n"], **{m: round(r[m] / max(1, r["n"]), 4) for m in r if m != "n"},
                               枠ごとに戻った割合={k: round(slot_ok[name][k] / slot_n[name][k], 4) for k in slot_n[name]},
                               枠を使った割合={k: round(slot_n[name][k] / max(1, r["n"]), 4) for k in slot_n[name]})
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = ["# I27s 4 つの操作と融合した差分で作った行（非公開）", "",
           "（元）が抜いた元の行。[崩れ] は逆回しで戻らなかった枠と、意味の物差しで通らなかった枠。", ""]
    for j, (wi, li, ex) in enumerate(examples, 1):
        lines = works[wi]["lines"]
        out += [f"## {j}. 作品 {works[wi]['name']} {li} 行目", "",
                f"- 　 {lines[li - 1]['text']}", f"- （元）{lines[li]['text']}", f"- 　 {lines[li + 1]['text']}", ""]
        for name, (line, bad) in ex.items():
            out.append(f"- ▶ {name}: **{line}**　[崩れ: {'、'.join(bad) or 'なし'}]")
        out.append("")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
