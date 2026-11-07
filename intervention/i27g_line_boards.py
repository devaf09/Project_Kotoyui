"""介入の研究 実験I27g: 行ごとに縮めた板で、中ほどの行を 2 通りに展開し、同じ名前の板を束ねて人物の統一感を見て、前後の行から展開を補う

持ち主（2026-11-07）: 「行単位で圧縮を行い、中ほどの行をどちらの種類でも展開を行う、あと、行ごとに同じ名前のバッファ板があればその属性を統合して、
全体として統一性があるか、特に人名、キャラクタで統一感があるかを調べてほしい。それから属性による展開で、正解率がある程度で頭打ちになるから、
そうしたら今度は前後の行の属性から選択を補完して正解率が上がるか試してほしい、とくに前後の行を展開して補完した場合と、
圧縮した（行のまとめられた板だけ）からの補完の差を調べてほしい」。
入力は作品ごとの行の並び（童話の 20 行版の文、または Web 小説のテキストの行）。Web 小説で回した結果・板は _wn を付け、非公開の置き場だけに置く。
  A. 行ごとの圧縮と、中ほどの行（各作品の行の真ん中 3 分の 1）の展開:
       可逆（ネストをたどる、I27e）と、関連性（ネストを捨て、兄弟の組の前後で並べ直す、I27f の「対」。順は試しの行以外から学ぶ）。
       関連性の展開は GiNZA で読み直し、値の帯の組の F1 で比べる。
  B. 同じ名前の板を束ねる: 行ごとに、名詞のまとまり（I27c）の板に属性を集める
       （修飾:M、する:V（が・は で係った述語）、される:V（を）、で・に:V、持つ:X（「N の X」の N））。
       人物 = 2 行以上で述語の主語になったまとまり。人名 = その中で中心が固有名詞のもの。
       統一感: (1) 行の板から名前を隠して、ほかの行で束ねた板のどれにいちばん近いか（余弦）で名前を当てる。
              (2) 同じ人物の行の板どうしの近さと、違う人物の行の板との近さを比べる。
  C. 属性による展開と、前後の行からの補い（中ほどの行が試し）:
       問い: 行の中の名詞 N に係っていた語（修飾）と、N が主語になった述語（する）を、縮めた行の板（N と、そこに係り口があることだけ）から当てる。
       属性による展開: N の束ねた板（ほかの行から）の、同じ種類の属性の数えで選ぶ。束ねに使う行を増やすと正解率がどこで頭打ちになるかを見る。
       前後の行からの補い: 前後 w 行の語を候補と加点に足す。2 通り:
         展開した前後: 前後の行の語すべて（値の帯をすべて開いた形）。
         縮めた前後: 前後の行のまとめた板だけ（述語の中心と、そこに係る名詞の中心だけ）。
       前後の語の重み = N との関連（ほかの行で N と同じ行に出た数、0 なら 0.1）。属性の数えに足して選ぶ（足す、β = 0.3 / 1 / 3）か、
       束ねた板に候補が無い問いだけ前後で補う（無いときだけ）。正解率は 1 位で当てた割合と、上位 5 に入った割合。
物差し: 展開の一致と F1、名前当ての正解率（偶然・いちばん多い人物と比べる）、近さの差、属性の当て方の正解率と、答えが候補に入っていた割合。
結果: --out（童話は ../Project_Kotoyui_private/intervention/results/i27g.jsonl、Web 小説は i27g_wn.jsonl）。
実行: python intervention/i27g_line_boards.py --tales <tales_v1_parsed.json> | --texts <作品1.txt> <作品2.txt> ... --out <jsonl>
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
from i27c_compress_units import units  # noqa: E402
from i27e_restore import compress, restore  # noqa: E402
from i27f_generate import edges, f1, learn_pairs, realize, to_toks  # noqa: E402

NOUN = ("NOUN", "PROPN", "PRON")


def load_works(a, nlp):
    works = []
    if a.tales:
        for t in json.load(open(a.tales, encoding="utf-8")):
            works.append(dict(name=t["name"], lines=[dict(text=s["text"], toks=s["toks"]) for s in t["parsed"]["20"]]))
    for p in a.texts or []:
        lines = [x.strip() for x in open(p, encoding="utf-8") if x.strip()]
        if a.max_lines:
            lines = lines[: a.max_lines]
        works.append(dict(name=os.path.basename(p), lines=[dict(text=x, toks=to_toks(nlp(x))) for x in lines]))
    return works


def middle(n):
    return set(range(n // 3, n - n // 3))


def line_boards(toks):
    """行の中の名詞のまとまり → 属性の数え。あわせて、そのまとまりが主語か、中心が固有名詞か、属性の答え（修飾・する）。"""
    tok2u, us = units(toks)
    out = {}
    for ui, u in enumerate(us):
        if u["kind"] != "名詞":
            continue
        h = u["head"]
        b = Counter()
        mods = []
        for i, w in enumerate(toks):
            if w["head"] == h and tok2u[i] != ui and w["pos"] in ("ADJ", "VERB", "NOUN", "PROPN") and w["dep"] in ("acl", "amod", "nmod"):
                b["修飾:" + w["lemma"]] += 1
                mods.append((w["lemma"], w["dep"]))
        hw = toks[h]
        p = toks[hw["head"]]
        subj = False
        preds = []
        if hw["head"] != h:
            if hw["dep"] == "nsubj" and p["pos"] in ("VERB", "ADJ"):
                b["する:" + p["lemma"]] += 1
                subj = True
                preds.append(p["lemma"])
            elif hw["dep"] == "obj" and p["pos"] == "VERB":
                b["される:" + p["lemma"]] += 1
            elif hw["dep"] == "obl" and p["pos"] in ("VERB", "ADJ"):
                b["で・に:" + p["lemma"]] += 1
            elif hw["dep"] == "nmod" and p["pos"] in NOUN:
                b["持つ:" + p["lemma"]] += 1
        name = u["text"]
        if name in out:
            out[name]["board"].update(b)
            out[name]["mods"] += mods
            out[name]["preds"] += preds
            out[name]["subj"] |= subj
        else:
            out[name] = dict(board=b, subj=subj, propn=hw["pos"] == "PROPN", mods=mods, preds=preds)
    return out


def skeleton(toks):
    """行のまとめた板: 述語の中心と、そこに係る名詞の中心の見出しだけ（修飾は捨てる）。"""
    keep = set()
    for i, w in enumerate(toks):
        if w["pos"] in ("VERB", "ADJ") and w["dep"] in ("ROOT", "advcl", "conj", "ccomp", "csubj"):
            keep.add(w["lemma"])
            for j, x in enumerate(toks):
                if x["head"] == i and x["pos"] in NOUN and x["dep"] in ("nsubj", "obj", "obl", "iobj"):
                    keep.add(x["lemma"])
    return Counter(keep)


def cos(a, b, db=None):
    num = sum(v * b.get(k, 0) for k, v in a.items())
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values())) if db is None else db
    return num / (da * db) if da and db else 0.0


def expand_part(works, nlp):
    train = []
    for w in works:
        mid = middle(len(w["lines"]))
        train.append(dict(name=w["name"], parsed={"20": [ln for i, ln in enumerate(w["lines"]) if i not in mid]}))
    pairs = learn_pairs(train, -1)
    rng = random.Random(0)
    res = dict(n=0, reversible_exact=0, reversible_noisy_exact=0, assoc_same=0, assoc_f1=0.0, nest_f1_ceiling=0.0)
    for w in works:
        for i in sorted(middle(len(w["lines"]))):
            toks = w["lines"][i]["toks"]
            tok2u, us = units(toks)
            boards, roots = compress(toks, tok2u, us)
            res["n"] += 1
            res["reversible_exact"] += restore(boards, roots, us, 0.0, rng) == list(range(len(toks)))
            res["reversible_noisy_exact"] += restore(boards, roots, us, 0.2, rng) == list(range(len(toks)))
            oc, _ = edges(toks)
            ref = "".join(t["text"] for t in toks)
            for how, key in (("関連（対）", "assoc"), ("元", "nest")):
                seq = realize(toks, how, pairs, rng)
                txt = "".join(toks[k]["text"] for k in seq)
                nc, _ = edges(to_toks(nlp(txt)))
                if key == "assoc":
                    res["assoc_same"] += txt == ref
                    res["assoc_f1"] += f1(oc, nc)
                else:
                    res["nest_f1_ceiling"] += f1(oc, nc)
    n = max(1, res["n"])
    return {k: (v / n if k != "n" else v) for k, v in res.items()}


def unity_part(works):
    res = []
    for w in works:
        lb = [line_boards(ln["toks"]) for ln in w["lines"]]
        subj_lines = Counter(nm for b in lb for nm, x in b.items() if x["subj"])
        chars = {nm for nm, c in subj_lines.items() if c >= 2}
        propn = {nm for nm in chars if any(b.get(nm, {}).get("propn") for b in lb)}
        merged = {c: Counter() for c in chars}
        for b in lb:
            for nm, x in b.items():
                if nm in chars:
                    merged[nm].update(x["board"])
        norms = {c: math.sqrt(sum(v * v for v in merged[c].values())) for c in chars}
        sizes = {c: sum(merged[c].values()) for c in chars}
        rng = random.Random(0)
        hits, n, base_hits = 0, 0, 0
        most = max(chars, key=lambda c: sum(merged[c].values())) if chars else None
        within, between = [], []
        confuse = Counter()
        per = defaultdict(lambda: [0, 0])
        for li, b in enumerate(lb):
            for nm, x in b.items():
                if nm not in chars or not x["board"]:
                    continue
                sims = {c: cos(x["board"], merged[c], norms[c]) for c in chars if c != nm}
                sims[nm] = cos(x["board"], merged[nm] - x["board"])           # 自分の行は束ねから除く
                pred = max(sims, key=lambda c: (sims[c], sizes[c]))
                n += 1
                hits += pred == nm
                base_hits += most == nm
                per[nm][0] += pred == nm
                per[nm][1] += 1
                if pred != nm:
                    confuse[(nm, pred)] += 1
                for lj in rng.sample(range(len(lb)), min(len(lb), 200)):      # 近さは 200 行までの見本で測る
                    b2 = lb[lj]
                    if lj == li:
                        continue
                    for nm2, x2 in b2.items():
                        if nm2 in chars and x2["board"]:
                            (within if nm2 == nm else between).append(cos(x["board"], x2["board"]))
        top = {c: [k for k, _ in merged[c].most_common(8)] for c in sorted(chars, key=lambda c: -sum(merged[c].values()))[:8]}
        res.append(dict(work=w["name"], n_lines=len(lb), n_chars=len(chars), n_names=len(propn), n_mentions=n,
                        acc=hits / max(1, n), chance=1 / max(1, len(chars)), most_frequent=base_hits / max(1, n),
                        acc_names=(sum(per[c][0] for c in propn) / max(1, sum(per[c][1] for c in propn))),
                        within=sum(within) / max(1, len(within)), between=sum(between) / max(1, len(between)),
                        confusions=[[a, b, c] for (a, b), c in confuse.most_common(5)], top_attrs=top,
                        per_char={c: per[c] for c in chars}))
    return res


def completion_part(works, fracs=(0.1, 0.25, 0.5, 0.75, 1.0), win=(1, 2, 3), beta=1.0, seed=0, all_lines=False, policies=False):
    rng = random.Random(seed)
    targets = []                               # (作品, 行, 名詞, 種類, 答え, 品詞の口)
    lbs, sks, allw = [], [], []
    for wi, w in enumerate(works):
        lb = [line_boards(ln["toks"]) for ln in w["lines"]]
        lbs.append(lb)
        sks.append([skeleton(ln["toks"]) for ln in w["lines"]])
        allw.append([Counter(t["lemma"] for t in ln["toks"] if t["pos"] in ("NOUN", "PROPN", "VERB", "ADJ")) for ln in w["lines"]])
        pos_of = {}
        for ln in w["lines"]:
            for t in ln["toks"]:
                pos_of.setdefault(t["lemma"], t["pos"])
        w["_pos"] = pos_of
        for li in (range(len(lb)) if all_lines else sorted(middle(len(lb)))):
            for nm, x in lb[li].items():
                for m, dep in x["mods"]:
                    targets.append((wi, li, nm, "修飾", m, dep))
                for p in x["preds"]:
                    targets.append((wi, li, nm, "する", p, "pred"))

    def merged_board(wi, li, nm, frac, order):
        b = Counter()
        k = max(1, int(round(frac * (len(lbs[wi]) - 1))))
        for lj in names_in[wi].get(nm, ()):
            if lj != li and rank[wi][lj] - (rank[wi][lj] > rank[wi][li]) < k:     # 並べ替えた順で先頭 k 行（自分の行を除く）に入る行だけ
                b.update(lbs[wi][lj][nm]["board"])
        return b

    def slot_ok(wi, lem, kind, dep):
        p = works[wi]["_pos"].get(lem, "")
        if kind == "する":
            return p in ("VERB", "ADJ")
        return p in (("NOUN", "PROPN") if dep == "nmod" else ("ADJ", "VERB"))
    names_in = []
    for lb in lbs:
        d = defaultdict(set)
        for lj, b in enumerate(lb):
            for nm in b:
                d[nm].add(lj)
        names_in.append(d)
    orders = [list(range(len(lb))) for lb in lbs]
    for o in orders:
        rng.shuffle(o)
    rank = [{lj: r for r, lj in enumerate(o)} for o in orders]
    out = dict(n_targets=len(targets), plateau={}, completion={})
    for frac in fracs:
        hit = cov = 0
        for wi, li, nm, kind, ans, dep in targets:
            b = merged_board(wi, li, nm, frac, orders)
            cands = {k.split(":", 1)[1]: v for k, v in b.items() if k.startswith(kind + ":")}
            cov += ans in cands
            if cands:
                hit += max(cands, key=lambda c: (cands[c], c)) == ans
        out["plateau"][str(frac)] = dict(acc=hit / max(1, len(targets)), coverage=cov / max(1, len(targets)))
    def ctx_counts(wi, li, w_, mode):
        ctx = Counter()
        for lj in range(li - w_, li + w_ + 1):
            if lj != li and 0 <= lj < len(lbs[wi]):
                ctx.update(allw[wi][lj] if mode == "展開した前後" else sks[wi][lj])
        return ctx

    def run(w_, mode, fill, beta_, rec=None, scored=None):
        hit = top5 = cov = 0
        for wi, li, nm, kind, ans, dep in targets:
            b = merged_board(wi, li, nm, 1.0, orders)
            cands = Counter({k.split(":", 1)[1]: v for k, v in b.items() if k.startswith(kind + ":")})
            if mode != "属性だけ" and (fill == "足す" or not cands):          # 無いときだけ: 束ねた板に候補が無い問いだけ前後で補う
                for lem, c in ctx_counts(wi, li, w_, mode).items():
                    if lem != nm and slot_ok(wi, lem, kind, dep):
                        # 前後の行に出た語を、N との関連（ほかの行で N と同じ行に出た数）で重みを付けて足す
                        co = sum(1 for lj in names_in[wi].get(nm, ()) if lj != li and lem in allw[wi][lj])
                        cands[lem] += beta_ * min(c, 1) * (co if co else 0.1)
            cov += ans in cands
            rk = sorted(cands, key=lambda c: (-cands[c], c)) if cands else []
            if rk:
                hit += rk[0] == ans
                top5 += ans in rk[:5]
            if rec is not None:
                rec.append(rk[:5])
            if scored is not None:
                scored.append([[c, round(cands[c], 3)] for c in rk[:10]])
        n = max(1, len(targets))
        return dict(acc=hit / n, top5=top5 / n, coverage=cov / n)
    out["sweep"] = {}
    for w_ in win:
        for mode in ("属性だけ", "展開した前後", "縮めた前後"):
            out["completion"][f"{mode}|{w_}"] = run(w_, mode, "足す", beta)
            if mode == "属性だけ":
                continue
            out["sweep"][f"{mode}|{w_}|無いときだけ"] = run(w_, mode, "無いときだけ", beta)
            for b_ in (0.3, 3.0):
                out["sweep"][f"{mode}|{w_}|足す β{b_}"] = run(w_, mode, "足す", b_)
    ex = {}
    for mode in ("属性だけ", "展開した前後", "縮めた前後"):
        ex[mode] = []
        run(3, mode, "足す", 0.3, ex[mode])
    out["_examples"] = [dict(t=t, picks={m: ex[m][i] for m in ex}) for i, t in enumerate(targets)]
    if policies:                                   # 判断の板（I27h の Web 小説版）用: 補い方ごとの上位 10 候補と点
        pol = {"属性だけ": (3, "属性だけ", "足す", 0.3), "展開 β0.3": (3, "展開した前後", "足す", 0.3),
               "縮め β0.3": (3, "縮めた前後", "足す", 0.3), "展開 β1": (3, "展開した前後", "足す", 1.0),
               "縮め β1": (3, "縮めた前後", "足す", 1.0), "展開 無いときだけ": (3, "展開した前後", "無いときだけ", 0.3),
               "縮め 無いときだけ": (3, "縮めた前後", "無いときだけ", 0.3)}
        lists = {}
        for k, (w_, mode, fill, b_) in pol.items():
            lists[k] = []
            run(w_, mode, fill, b_, scored=lists[k])
        subj = [Counter(nm for b in lb for nm, x in b.items() if x["subj"]) for lb in lbs]
        out["_policies"] = dict(targets=[list(t) + [subj[t[0]][t[2]] >= 2] for t in targets], lists=lists)
    return out


def write_examples(works, examples, path, n, ctx=5, seed=0):
    """補いの例: 前後の行で直った / 壊れた / どれも当たり / どれも外れを混ぜて n 個。中ほどの行は関連性で作り直した文も付ける。"""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for e in examples:
        wi, li, nm, kind, ans, dep = e["t"]
        ok = {m: bool(p) and p[0] == ans for m, p in e["picks"].items()}
        if not ok["属性だけ"] and ok["展開した前後"] and ok["縮めた前後"]:
            g = "前後で直った（両方）"
        elif not ok["属性だけ"] and ok["展開した前後"]:
            g = "展開した前後だけで直った"
        elif not ok["属性だけ"] and ok["縮めた前後"]:
            g = "縮めた前後だけで直った"
        elif ok["属性だけ"] and not (ok["展開した前後"] and ok["縮めた前後"]):
            g = "前後で壊れた"
        elif all(ok.values()):
            g = "どれも当たり"
        else:
            g = "どれも外れ"
        groups[g].append(e)
    quota = {"前後で直った（両方）": 3, "展開した前後だけで直った": 4, "縮めた前後だけで直った": 4, "前後で壊れた": 4, "どれも当たり": 2, "どれも外れ": 3}
    train = [dict(name=w["name"], parsed={"20": [ln for i, ln in enumerate(w["lines"]) if i not in middle(len(w["lines"]))]}) for w in works]
    pairs = learn_pairs(train, -1)
    out = ["# I27g 補いの例（非公開: Web 小説の本文を含む。_wn）", "",
           f"前後 {ctx} 行つき。問い = 中ほどの行の名詞 N に係っていた語（修飾）か、N が主語の述語（する）。補いは前後 3 行・β 0.3。上位 5 を左から。", ""]
    k = 0
    for g, q in quota.items():
        pick = rng.sample(groups[g], min(q, len(groups[g])))
        out += [f"## {g}（全 {len(groups[g])} 問中 {len(pick)}）", ""]
        for e in sorted(pick, key=lambda e: (e["t"][0], e["t"][1])):
            k += 1
            wi, li, nm, kind, ans, dep = e["t"]
            w = works[wi]
            toks = w["lines"][li]["toks"]
            gen = "".join(toks[j]["text"] for j in realize(toks, "関連（対）", pairs, rng))
            out += [f"### {k}. 作品 {chr(65 + wi)}・{li + 1} 行目　N = {nm}（{kind}）　答え = {ans}", ""]
            for lj in range(max(0, li - ctx), min(len(w["lines"]), li + ctx + 1)):
                mark = "▶ " if lj == li else "　 "
                out.append(f"    {mark}{lj + 1:>4}: {w['lines'][lj]['text']}")
            out += ["", f"- 関連性で作り直した行: {gen}"]
            for m, p in e["picks"].items():
                out.append(f"- {m}: {' / '.join(p) if p else '（候補なし）'}{'　○' if p and p[0] == ans else ''}")
            out.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", default=None)
    ap.add_argument("--texts", nargs="*", default=None)
    ap.add_argument("--max_lines", type=int, default=0)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i27g.jsonl")
    ap.add_argument("--tag", default="tales")
    ap.add_argument("--examples", default=None, help="補いの例を前後 5 行つきで書く md（非公開、本文を含む）")
    ap.add_argument("--n_examples", type=int, default=20)
    ap.add_argument("--all_lines", action="store_true", help="補いの問いを中ほどの行だけでなく全部の行から作る")
    ap.add_argument("--dump_policies", default=None, help="補い方ごとの候補と点を JSON で書く（非公開、I27h の Web 小説版の入力）")
    ap.add_argument("--cache", default=None, help="解析した行を JSON で置く / 読む（非公開の置き場）")
    a = ap.parse_args()
    import spacy                                                     # CPU で動かす
    nlp = spacy.load("ja_ginza")
    if a.cache and os.path.exists(a.cache):
        works = json.load(open(a.cache, encoding="utf-8"))
    else:
        works = load_works(a, nlp)
        if a.cache:
            json.dump(works, open(a.cache, "w", encoding="utf-8"), ensure_ascii=False)
    rec = dict(exp="I27g", tag=a.tag, works=len(works), lines=sum(len(w["lines"]) for w in works),
               expand=expand_part(works, nlp), unity=unity_part(works),
               completion=completion_part(works, all_lines=a.all_lines, policies=bool(a.dump_policies)))
    examples = rec["completion"].pop("_examples")
    pols = rec["completion"].pop("_policies", None)
    if a.dump_policies and pols:
        json.dump(pols, open(a.dump_policies, "w", encoding="utf-8"), ensure_ascii=False)
    if a.examples:
        write_examples(works, examples, a.examples, a.n_examples)
    print(json.dumps(dict(rec, unity=[{k: v for k, v in u.items() if k not in ("top_attrs", "per_char")} for u in rec["unity"]]),
                     ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
