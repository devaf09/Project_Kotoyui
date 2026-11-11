"""介入の研究 実験I27l: 波の仮説（I27k）を、作った曖昧さで数千問に広げる

I27k は手の印 73 か所だけで、差が 2 問ぶんしかなかった。答えの分かる曖昧な所を、行の並びから作って数を増やす。
作り方（作品ごと。Web 小説は _wn を付け、非公開の置き場だけに置く）:
  - 行の中で、修飾 M（形容詞・動詞の連体、acl / amod）が名詞 H に直に係っている所を拾う（曖昧でない所）。
  - 「遠い方が正解」の問い: M のあとに名詞 D の「の」を挟んだ形（M D の H）とみなす。候補 = [D, H]、答え = H。
  - 「近い方が正解」の問い: H のあとに「の 名詞 D」を足した形（M H の D）とみなす。候補 = [H, D]、答え = H。
  - D（紛らわしい語）の選び方 2 通り: でたらめ（作品の中で 3 回以上出た名詞）、
    紛らわしい（作品のほかの行で M が係ったことのある、H 以外の名詞。D には必ず根拠が来るので H に不利）、
    両方に根拠（紛らわしいのうち、正解の組 M + H もほかの行にある問いだけ。公平な比べ）。
  - 試しの行そのものは根拠に使わない。根拠 = ほかの行の「名詞 + 修飾:M」（I27a の話の板を 1 行ずつ）。
読み方: 近い方 / 規則（根拠が 1 つの候補だけなら選ぶ、そうでなければ近い方）/ 波（I27k、位相 0 で足し、いちばん明るい仮説）。
  読む行は試しの行から近い順（後ろ → 前を交互）、前後 w 行まで（w = 10・50・200・全部）。
物差し: 正解率、静か（根拠が来ない）の割合と誤り、根拠の来た問いで収束までに読んだ行の割合。
実行: python intervention/i27l_wave_scale.py --cache <I27g の --cache の JSON> --tag wn --out <..._wn.jsonl>
      python intervention/i27l_wave_scale.py --tales <tales_v1_parsed.json> --tag tales --out <jsonl>（童話は 20 行版をつないで 1 作品とする）
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
from i27a_story_check import story_boards  # noqa: E402
from i27c_compress_units import FORMAL  # noqa: E402

FORMAL2 = FORMAL | {"わけ", "つもり", "方", "時", "気", "の", "ん", "事", "物", "所", "為", "訳", "筈", "儘"}   # 形式名詞（機能語に近い名詞）

NOUN = ("NOUN", "PROPN")


def line_mods(toks):
    """行の中の (M の見出し, H の見出し)。"""
    out = []
    for w in toks:
        h = toks[w["head"]] if 0 <= w["head"] < len(toks) else None
        if h is not None and w is not h and w["pos"] in ("ADJ", "VERB") and w["dep"] in ("acl", "amod") and h["pos"] in NOUN:
            out.append((w["lemma"], h["lemma"]))
    return out


def order(li, n, win):
    out = []
    for d in range(1, n):
        if win and d > win:
            break
        for lj in (li + d, li - d):
            if 0 <= lj < n:
                out.append(lj)
    return out


def make_items(works, seed, per_work=1500):
    rng = random.Random(seed)
    items = []
    for wi, w in enumerate(works):
        lines = w["lines"]
        boards = [story_boards({"parsed": {"x": [ln]}}, ("x",)) for ln in lines]
        nouns = Counter(t["lemma"] for ln in lines for t in ln["toks"] if t["pos"] in NOUN)
        common = [k for k, v in nouns.items() if v >= 3]
        m_on = defaultdict(lambda: defaultdict(set))            # M → 名詞 → その組が出た行
        for li, ln in enumerate(lines):
            for m, h in line_mods(ln["toks"]):
                m_on[m][h].add(li)
        cand_sites = [(li, m, h) for li, ln in enumerate(lines) for m, h in line_mods(ln["toks"])]
        rng.shuffle(cand_sites)
        for li, m, h in cand_sites[:per_work]:
            h_else = bool(m_on[m][h] - {li})                     # 正解の組がほかの行にもあるか
            for dk in ("でたらめ", "紛らわしい", "両方に根拠"):
                if dk == "でたらめ":
                    pool = [d for d in common if d != h and d not in FORMAL2]   # 紛らわしい語に形式名詞は使わない
                elif dk == "紛らわしい" or h_else:
                    pool = [d for d, ls in m_on[m].items() if d != h and ls - {li} and d not in FORMAL2]
                else:
                    pool = []
                if not pool:
                    continue
                d = rng.choice(pool)
                for side in ("遠い方が正解", "近い方が正解"):
                    cand = [d, h] if side == "遠い方が正解" else [h, d]
                    items.append(dict(work=wi, line=li, m=m, cand=cand, gold=h, distract=dk, side=side))
        w["_boards"] = boards
        vec = defaultdict(Counter)                            # 作品の中の名詞の属性（似た語を探す用）
        for b in boards:
            for k, c in b.items():
                vec[k].update(c)
        w["_vec"] = vec
        # 関連の多い語板を、前後の品詞で別の板に分ける（持ち主 2026-11-07「同じ読みでも別の機能をもつ語板として、関連が多い語板は分解OK」）
        breadth = sorted((len(c) for c in vec.values()), reverse=True)
        cut = max(20, breadth[max(0, len(breadth) // 50 - 1)] if breadth else 20)       # 属性の種類が上位 2%（20 種以上）
        split = {k for k, c in vec.items() if len(c) >= cut} | FORMAL2
        bss, vecs = [], defaultdict(Counter)
        for ln, b in zip(lines, boards):
            nb = {k: c for k, c in b.items()}
            for t in ln["toks"]:
                k = t["lemma"]
                if k in split and k in nb and "|" not in k:
                    i = t["i"]
                    pv = ln["toks"][i - 1]["pos"] if i > 0 else "頭"
                    nx = ln["toks"][i + 1] if i + 1 < len(ln["toks"]) else None
                    nv = "尾" if nx is None else (nx["text"] if nx["pos"] in ("ADP", "AUX", "SCONJ", "PART") else nx["pos"])
                    nb[f"{k}|{pv}>{nv}"] = nb.pop(k)
            bss.append(nb)
            for k, c in nb.items():
                vecs[k].update(c)
        w["_boards_s"], w["_vecs"], w["_split"] = bss, vecs, split
        # 関わりの種類ごとの似方: 名詞 → 受けた連体の修飾（acl / amod）の数え。行ごとにも持つ（試しの行の分を引くため）
        w["_mods_line"] = [Counter() for _ in lines]
        vm = defaultdict(Counter)
        for li, ln in enumerate(lines):
            for m, h in line_mods(ln["toks"]):
                vm[h][m] += 1
                w["_mods_line"][li][(h, m)] += 1
        w["_vmod"] = vm
    return items


def sim_in_work(w, x, a, b, key, split=False):
    """作品の中の属性の余弦。問いの属性（修飾:M）と、試しの行の分は除く。split なら b は分けた板。"""
    own_a = w["_boards"][x["line"]]
    own_b = w["_boards_s" if split else "_boards"][x["line"]]
    va, vb = w["_vec"].get(a, {}), w["_vecs" if split else "_vec"].get(b, {})

    def get(v, own, word, k):
        return v.get(k, 0) - own.get(word, {}).get(k, 0)
    ks = [k for k in va if k != key and k in vb]
    dot = sum(get(va, own_a, a, k) * get(vb, own_b, b, k) for k in ks)
    if dot <= 0:
        return 0.0
    na = sum(get(va, own_a, a, k) ** 2 for k in va if k != key) ** 0.5
    nb = sum(get(vb, own_b, b, k) ** 2 for k in vb if k != key) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


VEC = {}


def sim_vec(a, b):
    import numpy as np
    for wd in (a, b):
        if wd not in VEC:
            lx = NLP.vocab[wd]
            VEC[wd] = lx.vector / (np.linalg.norm(lx.vector) + 1e-9) if lx.has_vector else None
    if VEC[a] is None or VEC[b] is None:
        return 0.0
    c = float(VEC[a] @ VEC[b])
    return c if 0 < c < 0.999 else 0.0                       # 辞書に無い語が同じ行に落ちた組（1.00）は除く


NLP = None
QUICK = ("似た語へ広げる 作品の属性 λ1", "広げる 同じ修飾 λ3", "広げる 同じ修飾 + 語ベクトル λ1",
         "同じ修飾 λ3 + ほかの作品 μ0.3", "同じ修飾 λ3 + ほかの作品 μ0.3 似た語も", "同じ組だけ + ほかの作品 μ1")


def sim_mod(w, x, a, b, m):
    """「同じ連体の修飾を受けたか」だけで測る似方。問いの修飾 M と、試しの行の分は除く。"""
    own = w["_mods_line"][x["line"]]
    va, vb = w["_vmod"].get(a, {}), w["_vmod"].get(b, {})

    def get(v, word, k):
        return v.get(k, 0) - own.get((word, k), 0)
    dot = sum(get(va, a, k) * get(vb, b, k) for k in va if k != m and k in vb)
    if dot <= 0:
        return 0.0
    na = sum(get(va, a, k) ** 2 for k in va if k != m) ** 0.5
    nb = sum(get(vb, b, k) ** 2 for k in vb if k != m) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def shared(works):
    """作品をまたいで共有する板: 全作品の「名詞 → 受けた連体の修飾」と、修飾 → 名詞の引き。"""
    G, byM = defaultdict(Counter), defaultdict(Counter)
    for w in works:
        for h, c in w["_vmod"].items():
            G[h].update(c)
            for m, v in c.items():
                byM[m][h] += v
    return G, byM


def cos_c(a, b, skip):
    dot = sum(v * b.get(k, 0) for k, v in a.items() if k != skip)
    if dot <= 0:
        return 0.0
    na = sum(v * v for k, v in a.items() if k != skip) ** 0.5
    nb = sum(v * v for k, v in b.items() if k != skip) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def cross_row(works, x, mu, lam, top=50):
    """ほかの作品の板からの根拠（1 行ぶんとして最後に足す）。同じ組の数 + 似た語（同じ修飾の似方）に M が係った数 × λ。"""
    G, byM = SHARED
    w = works[x["work"]]
    own = w["_vmod"]
    m = x["m"]

    def other(h):                                   # ほかの作品だけの数え（作品と語ごとに 1 回だけ作る）
        k = (x["work"], h)
        if k not in _OTHER:
            o = Counter(G.get(h, {}))
            o.subtract(own.get(h, {}))
            _OTHER[k] = +o
        return _OTHER[k]
    row = []
    for c in x["cand"]:
        oc = other(c)
        v = oc.get(m, 0)
        if lam:
            for n, _ in byM.get(m, Counter()).most_common(top):
                if n in x["cand"]:
                    continue
                on = other(n)
                if on.get(m, 0) > 0:
                    v += lam * cos_c(oc, on, m) * on[m]
        row.append(mu * v)
    return row


SHARED = None
_OTHER = {}


def evidence(works, x, win, spread=0.0, how="作品の属性", th=0.0, skip_formal=False, freq_discount=False, split=False,
             cross=0.0, cross_lam=0.0):
    """行ごとの候補の根拠。spread > 0 なら、同じ組だけでなく、候補に似た名詞に M が係った数も（似ている度合い × spread）足す（B1）。"""
    w = works[x["work"]]
    bs = w["_boards"]
    key = "修飾:" + x["m"]
    out = []
    for lj in order(x["line"], len(bs), win):
        b = bs[lj]
        row = []
        for c in x["cand"]:
            v = b.get(c, {}).get(key, 0)
            if spread:
                for n, attrs in (w["_boards_s"][lj] if split else b).items():
                    base = n.split("|")[0]
                    if base in x["cand"] or not attrs.get(key) or (skip_formal and base in FORMAL2):
                        continue
                    if how == "作品の属性":
                        sm = sim_in_work(w, x, c, n, key, split)
                    elif how == "同じ修飾":
                        sm = sim_mod(w, x, c, base, x["m"])
                    elif how == "同じ修飾 + 語ベクトル":
                        sm = max(sim_mod(w, x, c, base, x["m"]), sim_vec(c, base) if sim_vec(c, base) >= 0.5 else 0.0)
                    else:
                        sm = sim_vec(c, base)
                    if freq_discount:                   # よく出る語ほど何にでも似るので、出た数の対数で割り引く
                        sm /= math.log(2 + sum(w["_vecs" if split else "_vec"].get(n, {}).values()))
                    if sm > th:
                        v += spread * sm * attrs[key]
            row.append(v)
        out.append(row)
    if cross:
        out.append(cross_row(works, x, cross, cross_lam))
    return out


def read_rule(ev):
    tot = [sum(r[k] for r in ev) for k in range(2)]
    pos = [k for k in range(2) if tot[k] > 0]
    return pos[0] if len(pos) == 1 else 0


def read_wave(ev, eps=0.3, q=0.6):
    amp = [eps, 0.0]
    hist, got = [], False
    for r in ev:
        got = got or any(r)
        for k in range(2):
            amp[k] += r[k]
        b = [a * a for a in amp]
        top = 0 if b[0] >= b[1] else 1
        hist.append((top, b[top] / sum(b) if got else 0.0))
    b = [a * a for a in amp]
    top = 0 if b[0] >= b[1] else 1
    conv = len(hist)
    for i in range(len(hist) - 1, -1, -1):
        if hist[i][0] == top and hist[i][1] >= q:
            conv = i
        else:
            break
    return top, (conv + 1) / max(1, len(hist)), not got


_JOB = {}


def _one(k):
    """1 問を読む（並列の働き手が呼ぶ。作品・問い・広げ方は fork のときに _JOB で受け継ぐ）。"""
    works, items, win, ekw = _JOB["works"], _JOB["items"], _JOB["win"], _JOB["ekw"]
    x = items[k]
    ev = evidence(works, x, win, **ekw)
    gi = x["cand"].index(x["gold"])
    t, f, s = read_wave(ev)
    return gi == 0, read_rule(ev) == gi, t == gi, f, s, t


JOBS = 1
ANSWERS = {}                                            # (広げ方の名前, 読む幅) → 問いごとの波の答え（例を書き出す用）


def run(works, items, wins, name="同じ組だけ", **ekw):
    rec = {}
    for win in wins:
        _JOB.update(works=works, items=items, win=win, ekw=ekw)
        if JOBS > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(JOBS) as pool:
                res = pool.map(_one, range(len(items)), chunksize=max(1, len(items) // (JOBS * 20)))
        else:
            res = [_one(k) for k in range(len(items))]
        ANSWERS[(name, win)] = [r[5] for r in res]
        r = {}
        groups = defaultdict(list)
        for k, x in enumerate(items):
            groups[(x["distract"], x["side"])].append(k)
            groups[(x["distract"], "合わせて")].append(k)
        for g, ks in sorted(groups.items()):
            near = sum(res[k][0] for k in ks)
            rule_ok = sum(res[k][1] for k in ks)
            wave_ok = sum(res[k][2] for k in ks)
            silent = sum(res[k][4] for k in ks)
            silent_err = sum(res[k][4] and not res[k][2] for k in ks)
            rf = [res[k][3] for k in ks if not res[k][4]]
            n = len(ks)
            r["/".join(g)] = dict(n=n, 近い方=near / n, 規則=rule_ok / n, 波=wave_ok / n, 静か=silent / n,
                                  静かの誤り=silent_err / max(1, silent), 波_静か以外=(wave_ok - (silent - silent_err)) / max(1, n - silent),
                                  収束まで読んだ割合=sum(rf) / max(1, len(rf)))
        rec[str(win or "全部")] = r
    return rec


def dump_examples(works, items, path, n=20, seed=0):
    """問いを n 個でたらめに選び、前後 1 行と、広げ方ごとの波の答えを書く（非公開の置き場だけに置く）。"""
    rng = random.Random(seed)
    ks = rng.sample(range(len(items)), min(n, len(items)))
    out = ["# I27l 問いの例（非公開）", "",
           "問いの作り方: 行の中の「M → H」（修飾が名詞に直に係る所）に、紛らわしい名詞 D を足して曖昧にする。"
           "遠い方が正解 = 「M D の H」、近い方が正解 = 「M H の D」。答えはいつも H。", ""]
    for j, k in enumerate(ks, 1):
        x = items[k]
        lines = works[x["work"]]["lines"]
        li = x["line"]
        d = [c for c in x["cand"] if c != x["gold"]][0]
        shape = f"{x['m']} {d} の {x['gold']}" if x["side"] == "遠い方が正解" else f"{x['m']} {x['gold']} の {d}"
        out += [f"## {j}. 作品 {works[x['work']]['name']} {li} 行目　{x['distract']}・{x['side']}", "",
                f"- 曖昧にした形: 「{shape}」　候補 = {x['cand']}　答え = {x['gold']}"]
        for lj in (li - 1, li, li + 1):
            if 0 <= lj < len(lines):
                out.append(f"- {'▶' if lj == li else '　'} {lines[lj]['text']}")
        for (nm, win), ans in ANSWERS.items():
            got = x["cand"][ans[k]]
            out.append(f"- [{'○' if got == x['gold'] else '×'}] {nm}（読む幅 {win or '全部'}）→ {got}")
        out.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--tales", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="並列の数（CPU の働き手）")
    ap.add_argument("--dump", default=None, help="問いの例を書く md（非公開の置き場）")
    ap.add_argument("--dump_n", type=int, default=20)
    ap.add_argument("--only", default=None, help="この名前の広げ方だけ回す（| で区切る）")
    ap.add_argument("--quick", action="store_true", help="大きい量で回すとき: 読む幅は 50 行と全部、広げ方は 3 通りだけ")
    a = ap.parse_args()
    global JOBS
    JOBS = a.jobs
    if a.cache:
        works = json.load(open(a.cache, encoding="utf-8"))
    else:
        tales = json.load(open(a.tales, encoding="utf-8"))
        works = [dict(name=str(i), lines=t["parsed"]["20"]) for i, t in enumerate(tales)]
    items = make_items(works, a.seed)
    rec = dict(exp="I27l", tag=a.tag, works=len(works), lines=sum(len(w["lines"]) for w in works), n=len(items),
               result=run(works, items, (50, 0) if a.quick else (10, 50, 200, 0)))
    global NLP
    import spacy
    NLP = spacy.load("ja_ginza")
    global SHARED
    SHARED = shared(works)
    rec["spread"] = {}
    for name, kw in (("似た語へ広げる 作品の属性 λ1", dict(spread=1.0)), ("似た語へ広げる 作品の属性 λ1 似0.3以上", dict(spread=1.0, th=0.3)),
                     ("似た語へ広げる 語ベクトル λ1 似0.5以上", dict(spread=1.0, how="語ベクトル", th=0.5)),
                     ("広げる 作品の属性 λ1 形式名詞を除く", dict(spread=1.0, skip_formal=True)),
                     ("広げる 作品の属性 λ3 形式名詞を除く + 頻度で割り引く", dict(spread=3.0, skip_formal=True, freq_discount=True)),
                     ("広げる 語ベクトル λ1 似0.5以上 形式名詞を除く", dict(spread=1.0, how="語ベクトル", th=0.5, skip_formal=True)),
                     ("広げる 作品の属性 λ1 関連の多い語板を前後の品詞で分ける", dict(spread=1.0, split=True)),
                     ("広げる 同じ修飾 λ1", dict(spread=1.0, how="同じ修飾")),
                     ("広げる 同じ修飾 λ3", dict(spread=3.0, how="同じ修飾")),
                     ("広げる 同じ修飾 + 語ベクトル λ1", dict(spread=1.0, how="同じ修飾 + 語ベクトル")),
                     ("同じ組だけ + ほかの作品 μ1", dict(cross=1.0)),
                     ("同じ修飾 λ3 + ほかの作品 μ0.3", dict(spread=3.0, how="同じ修飾", cross=0.3)),
                     ("同じ修飾 λ3 + ほかの作品 μ0.3 似た語も", dict(spread=3.0, how="同じ修飾", cross=0.3, cross_lam=1.0)),
                     ("広げる 作品の属性 λ3 分ける + 頻度で割り引く", dict(spread=3.0, split=True, freq_discount=True))):
        if a.quick and name not in QUICK:
            continue
        if a.only and name not in a.only.split("|"):
            continue
        rec["spread"][name] = run(works, items, (0,), name=name, **kw)
        print(name, "done", flush=True)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.dump:
        dump_examples(works, items, a.dump, a.dump_n, a.seed)


if __name__ == "__main__":
    main()
