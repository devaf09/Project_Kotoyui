"""介入の研究 実験I27q: 本（作品）全体を読み、中ほどに行を足す（試験の一覧 C）

持ち主（2026-11-08）: 「１つは本や章全体を読んで、中ほどに１行を追加してほしい。前後の行に属性の重みの坂をつけるといいかも。
2つ目は、１の行作成のさい、関係あるけど少し遠い属性の、その本に入ってる名詞をランダムに１ついれて、行数関係なく前後の属性重みの違和感が消えるまで文章を作成してほしい。」

方法 1（--mode 1）: 作品の全部の行から想定属性板を作る。足す位置からの行の距離 d に重みの坂をつける:
  平ら（前後 10 行、I27o と同じ）/ 指数 exp(-d/τ)（τ = 5・20・100）/ 直線（1 - d/作品の行数）/ 平ら（作品全部）。
  行の作り方は I27o の (a)+(b)+(c)+(d)（形式名詞を主語にしない・隣の行の組を避ける・話の流れの板・修飾の中の語）。
  物差しは I27o と同じ（行を 1 つ抜いて、そこに作った行と元の行の内容語 F1、隣の行との F1、述語・主語の一致）。
方法 2（--mode 2）: 方法 1 の板（坂 τ = 20）で、足す位置の「近く」の重みが小さく、作品全体の属性は近くの話題と少し似ている名詞 N を
  でたらめに 1 つ選ぶ（関係あるが少し遠い名詞）。N を主語にした行から始め、次の行は前の行とつながる組（同じ名詞・前の行の述語からの流れ）の中から、
  近くの板で重い組を選ぶ（前の行の名詞を 1 つは持つ行だけ、近くの元の行の組は写さない）。行の違和感 = 1 - （主語と述語が近くの板でどれだけ重いかの平均）。違和感が、その位置の元の前後 3 行の違和感の平均以下になったら止める（最大 12 行）。
  比べ: N を「近い名詞」「関係あるが少し遠い名詞」「関係ない名詞」から選んだとき、何行で違和感が消えるか。
Web 小説は _wn。例の md と結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i27q_book_insert.py --cache <解析済み JSON> --mode 1 --stats <jsonl> --out <例の md>
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
from i27l_wave_scale import FORMAL2  # noqa: E402

NOUN = M.NOUN


def prep(works):
    """行ごとの属性（修飾の中の語を含めた形）と名詞の表記を 1 回だけ数える。"""
    for w in works:
        w["_attrs"] = []
        w["_text"] = {}
        for ln in w["lines"]:
            toks = ln["toks"]
            for t in toks:
                if t["pos"] in NOUN:
                    w["_text"].setdefault(t["lemma"], t["text"])
            seen, out = set(), []
            for n, a, f, p in O.line_attrs2(toks, True):
                if (n, a) not in seen:
                    seen.add((n, a))
                    out.append((n, a, f, p))
            w["_attrs"].append(out)


def slope(kind, d, n):
    if kind == "平ら10":
        return 1.0 + 0.1 / d if d <= 10 else 0.0
    if kind == "平ら全部":
        return 1.0
    if kind == "直線":
        return max(0.0, 1.0 - d / n)
    tau = float(kind.split("τ")[1])
    return math.exp(-d / tau)


def book_board(w, li, kind, skip=True):
    """行 li を除いた作品の全部の行から、坂の重みで板を作る（skip=False なら li も入れる: 行の「間」に足すとき用）。"""
    lines = w["lines"]
    n = len(lines)
    wb = defaultdict(Counter)
    form = defaultdict(Counter)
    part = defaultdict(Counter)
    for lj in range(n):
        if skip and lj == li:
            continue
        d = abs(lj - li) if skip else abs(lj - (li - 0.5)) + 0.5
        wt = slope(kind, d, n)
        if wt <= 1e-4:
            continue
        for nm, a, f, p in w["_attrs"][lj]:
            wb[nm][a] += wt
            if f:
                form[a][f] += wt
            if p:
                part[(nm, a.split(":")[0])][p] += wt
    return wb, form, part, w["_text"]


KINDS = ("平ら10", "τ5", "τ20", "τ100", "直線", "平ら全部")


def mode1(works, tr_all, roots, a):
    rng = random.Random(a.seed)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(10, len(w["lines"]) - 10)
             if any(t["dep"] == "nsubj" for t in w["lines"][li]["toks"])]
    rng.shuffle(sites)
    sites = sites[:a.n_stats]
    res = {k: Counter() for k in KINDS}
    examples = []
    for wi, li in sites:
        lines = works[wi]["lines"]
        toks = lines[li]["toks"]
        ref = M.content_lemmas(toks)
        nb = [M.content_lemmas(lines[lj]["toks"]) for lj in (li - 1, li + 1)]
        ref_p, ref_s = O.root_pred(toks), O.subj_of(toks)
        prev_p = roots[wi][li - 1]
        sub = []
        if prev_p and roots[wi][li]:
            sub.append((prev_p, roots[wi][li]))
        if roots[wi][li] and roots[wi][li + 1]:
            sub.append((roots[wi][li], roots[wi][li + 1]))
        for x, y in sub:
            tr_all[x][y] -= 1
        avoid = O.pairs_in(lines[li - 1]["toks"]) | O.pairs_in(lines[li + 1]["toks"])
        ex = {}
        for kind in KINDS:
            board = book_board(works[wi], li, kind)
            s, p = O.choose(board, skip_formal=True, avoid=avoid, flow=True, prev_p=prev_p, tr=tr_all, joint=True)
            if not s:
                continue
            line, pick, mform = O.expand2(board, s, p)
            g = Counter(x for x in pick.values() if x)
            r = res[kind]
            r["n"] += 1
            r["F1"] += M.f1(ref, g)
            r["隣F1"] += max(M.f1(nb[0], g), M.f1(nb[1], g))
            r["述語一致"] += p == ref_p
            r["主語一致"] += s == ref_s
            ex[kind] = line
        for x, y in sub:
            tr_all[x][y] += 1
        if len(examples) < a.n and len(ex) == len(KINDS):
            examples.append((wi, li, ex))
    rec = dict(exp="I27q", mode=1, n=len(sites),
               結果={k: dict(n=r["n"], **{m: round(r[m] / max(1, r["n"]), 4) for m in r if m != "n"}) for k, r in res.items()})
    out = ["# I27q 方法 1: 作品全体を読み、坂の重みの板から中ほどに 1 行（非公開）", "",
           "行を 1 つ抜き（（元））、その位置に坂ごとの板から作った行（▶）を置いた。前後は 2 行ずつ。", ""]
    for j, (wi, li, ex) in enumerate(examples, 1):
        lines = works[wi]["lines"]
        out += [f"## {j}. 作品 {works[wi]['name']} {li} 行目（全 {len(lines)} 行）", ""]
        for lj in range(li - 2, li):
            out.append(f"- 　 {lines[lj]['text']}")
        out.append(f"- （元）{lines[li]['text']}")
        for lj in range(li + 1, li + 3):
            out.append(f"- 　 {lines[lj]['text']}")
        out.append("")
        for k, line in ex.items():
            out.append(f"- ▶ {k}: **{line}**")
        out.append("")
    return rec, out


# ---- 方法 2 ----

def local_support(local, s, p):
    """近くの板（坂 τ5）で、主語 s と述語 p がどれだけ重いか（それぞれ板の 1 位で割る）。"""
    nw = {n: sum(c.values()) for n, c in local.items()}
    pw = Counter()
    for n, c in local.items():
        for at, v in c.items():
            if at.startswith("する:"):
                pw[at.split(":", 1)[1]] += v
    mn = max(nw.values()) if nw else 1.0
    mp = max(pw.values()) if pw else 1.0
    return 0.5 * min(1.0, nw.get(s, 0) / mn) + 0.5 * min(1.0, pw.get(p, 0) / mp), nw, pw, mn, mp


def discomfort(local, s, p):
    return 1.0 - local_support(local, s, p)[0]


def cos(a, b):
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if dot <= 0:
        return 0.0
    return dot / ((sum(v * v for v in a.values()) ** 0.5) * (sum(v * v for v in b.values()) ** 0.5))


def pick_noun(flat, local, kind, rng, top_s):
    """足す名詞 N を選ぶ。kind: 近い / 関係あるが少し遠い / 関係ない。"""
    nw = {n: sum(c.values()) for n, c in local.items()}
    mx = max(nw.values()) if nw else 1.0
    ctx = Counter()                                            # 近くの話題の属性（近くの板の重い名詞 5 つの属性を足したもの）
    for n, _ in Counter(nw).most_common(5):
        ctx.update(flat.get(n, {}))
    cands = []
    for n, c in flat.items():
        if n in FORMAL2 or sum(c.values()) < 3 or not any(at.startswith("する:") for at in c) or n == top_s:
            continue
        near = nw.get(n, 0) / mx
        rel = cos(c, ctx)
        if kind == "近い" and near >= 0.3:
            cands.append(n)
        elif kind == "関係あるが少し遠い" and near < 0.1 and 0.15 <= rel <= 0.5:
            cands.append(n)
        elif kind == "関係ない" and near < 0.1 and rel < 0.05:
            cands.append(n)
    return rng.choice(sorted(cands)) if cands else None


def gen_passage(w, li, flat_t, local, sloped, N, tr, max_lines=12, thr=None, near_pairs=frozenset()):
    """N を主語にした行から始め、違和感が thr 以下になるまで行をつなぐ。返り値 [(行, 主語, 述語, 違和感, 選んだ語)]。"""
    flat = flat_t[0]
    def top_pred(n):
        c = Counter({at.split(":", 1)[1]: v for at, v in flat.get(n, {}).items() if at.startswith("する:")})
        return max(c, key=lambda k: (c[k], k)) if c else None
    out = []
    s, p = N, top_pred(N)
    if not p:
        return out
    used = set()
    for _ in range(max_lines):
        line, pick, _ = O.expand2(sloped if s in sloped[0] else flat_t, s, p)
        dc = discomfort(local[0], s, p)
        out.append((line, s, p, round(dc, 3), dict(pick)))
        used.add((s, p))
        if thr is not None and dc <= thr:
            break
        # 次の組: 前の行の名詞（主語・目的語・補語）か、近くの板の話題の名詞を主語に、前の述語からの流れで述語を選ぶ
        prev_n = {x for x in (pick.get("主語"), pick.get("目的語"), pick.get("補語")) if x}
        nxt = tr.get(p, {})
        tot = sum(nxt.values()) or 1
        sup, nw, pw, mn, mp = local_support(local[0], s, p)
        cands = []
        for n, c in sloped[0].items():
            if n in FORMAL2:
                continue
            for at, v in c.items():
                if not at.startswith("する:"):
                    continue
                q = at.split(":", 1)[1]
                if (n, q) in used or (n, q) in near_pairs:          # 同じ組を繰り返さない・近くの元の行を写さない
                    continue
                sc = (0.05 + nxt.get(q, 0) / tot) * (0.05 + min(1.0, nw.get(n, 0) / mn)) * (0.05 + min(1.0, pw.get(q, 0) / mp)) * (1 + v)
                cands.append((sc, n, q))
        cands.sort(reverse=True)
        nxt_sp = None
        for sc, n, q in cands[:80]:                                  # 前の行の名詞を 1 つは持つ行だけ（つながりを切らない）
            _, pk, _ = O.expand2(sloped, n, q)
            if {x for x in (pk.get("主語"), pk.get("目的語"), pk.get("補語")) if x} & prev_n:
                nxt_sp = (n, q)
                break
        if not nxt_sp:
            break
        s, p = nxt_sp
    return out


def mode2(works, tr_all, roots, a):
    rng = random.Random(a.seed)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(len(w["lines"]) // 3, 2 * len(w["lines"]) // 3)
             if O.root_pred(w["lines"][li - 1]["toks"]) and O.root_pred(w["lines"][li]["toks"])]
    rng.shuffle(sites)
    sites = sites[:a.n_stats2]
    kinds = ("近い", "関係あるが少し遠い", "関係ない")
    res = {k: Counter() for k in kinds}
    lens = {k: Counter() for k in kinds}
    examples = []
    for si, (wi, li) in enumerate(sites):
        w = works[wi]
        lines = w["lines"]
        sloped = book_board(w, li, "τ20", skip=False)          # 行 li-1 と li の間に足す
        local = book_board(w, li, "τ5", skip=False)
        flat_t = book_board(w, li, "平ら全部", skip=False)
        flat = flat_t[0]
        # その位置の元の行の違和感（前後 3 行の主語・述語）
        base = []
        near_pairs = set()
        for lj in range(max(0, li - 3), min(len(lines), li + 3)):
            near_pairs |= O.pairs_in(lines[lj]["toks"])
        for lj in range(max(0, li - 3), min(len(lines), li + 3)):
            sj, pj = O.subj_of(lines[lj]["toks"]), O.root_pred(lines[lj]["toks"])
            if pj:
                base.append(discomfort(local[0], sj or "", pj))
        thr = sum(base) / len(base)
        s0, _ = O.choose(sloped, skip_formal=True, avoid=None, flow=False, prev_p=None, tr=tr_all)
        ex = {}
        for k in kinds:
            N = pick_noun(flat, local[0], k, random.Random(a.seed * 1000 + si), s0)
            if not N:
                continue
            ps = gen_passage(w, li, flat_t, local, sloped, N, tr_all, thr=thr, near_pairs=near_pairs)
            if not ps:
                continue
            r = res[k]
            r["n"] += 1
            done = ps[-1][3] <= thr
            r["消えた"] += done
            r["つながりが切れた"] += (not done) and len(ps) < 12
            if done:
                r["行数（消えたとき）"] += len(ps)
                lens[k][len(ps)] += 1
            r["始めの違和感"] += ps[0][3]
            ex[k] = (N, ps, thr)
        if len(examples) < a.n and len(ex) == len(kinds):
            examples.append((wi, li, ex))
    rec = dict(exp="I27q", mode=2, n=len(sites),
               結果={k: dict(n=r["n"], 消えた割合=round(r["消えた"] / max(1, r["n"]), 4),
                            つながりが切れた割合=round(r["つながりが切れた"] / max(1, r["n"]), 4),
                            消えるまでの行数=round(r["行数（消えたとき）"] / max(1, r["消えた"]), 3),
                            始めの違和感=round(r["始めの違和感"] / max(1, r["n"]), 4),
                            行数の分かれ=dict(sorted(lens[k].items()))) for k, r in res.items()})
    out = ["# I27q 方法 2: 関係あるが少し遠い名詞を 1 つ入れ、違和感が消えるまで行を作る（非公開）", "",
           "行 li-1 と li の間に足す。N = 入れた名詞。各行の [ ] は違和感（1 - 近くの板での主語と述語の重さ）。"
           "止める目安 = その位置の元の前後 3 行の違和感の平均。比べに「近い名詞」「関係ない名詞」で始めた場合も付けた。", ""]
    for j, (wi, li, ex) in enumerate(examples, 1):
        lines = works[wi]["lines"]
        out += [f"## {j}. 作品 {works[wi]['name']} {li - 1} 行目と {li} 行目の間（全 {len(lines)} 行）", ""]
        for lj in range(li - 2, li):
            out.append(f"- 　 {lines[lj]['text']}")
        k = "関係あるが少し遠い"
        N, ps, thr = ex[k]
        out.append(f"- 　 ――ここに足す（N = {N}、止める目安 {thr:.3f}）――")
        for line, s, p, dc, _ in ps:
            out.append(f"- ▶ **{line}** [{dc:.3f}]")
        for lj in range(li, li + 2):
            out.append(f"- 　 {lines[lj]['text']}")
        out.append("")
        for k2 in ("近い", "関係ない"):
            N2, ps2, _ = ex[k2]
            out.append(f"- 比べ（{k2}、N = {N2}、{len(ps2)} 行）: " + " / ".join(f"{x[0]} [{x[3]:.2f}]" for x in ps2))
        out.append("")
    return rec, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_stats", type=int, default=2000)
    ap.add_argument("--n_stats2", type=int, default=600)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    prep(works)
    tr_all, roots = O.flow_board(works)
    rec, out = (mode1 if a.mode == 1 else mode2)(works, tr_all, roots, a)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
