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


def evidence(works, x, win, spread=0.0, how="作品の属性", th=0.0, skip_formal=False, freq_discount=False, split=False):
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


def run(works, items, wins, **ekw):
    rec = {}
    for win in wins:
        r = {}
        groups = defaultdict(list)
        for x in items:
            groups[(x["distract"], x["side"])].append(x)
            groups[(x["distract"], "合わせて")].append(x)
        for g, xs in sorted(groups.items()):
            near = rule_ok = wave_ok = silent = silent_err = 0
            rf = []
            for x in xs:
                ev = evidence(works, x, win, **ekw)
                gi = x["cand"].index(x["gold"])
                near += gi == 0
                rule_ok += read_rule(ev) == gi
                t, f, s = read_wave(ev)
                wave_ok += t == gi
                silent += s
                silent_err += s and t != gi
                if not s:
                    rf.append(f)
            n = len(xs)
            r["/".join(g)] = dict(n=n, 近い方=near / n, 規則=rule_ok / n, 波=wave_ok / n, 静か=silent / n,
                                  静かの誤り=silent_err / max(1, silent), 波_静か以外=(wave_ok - (silent - silent_err)) / max(1, n - silent),
                                  収束まで読んだ割合=sum(rf) / max(1, len(rf)))
        rec[str(win or "全部")] = r
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--tales", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cache:
        works = json.load(open(a.cache, encoding="utf-8"))
    else:
        tales = json.load(open(a.tales, encoding="utf-8"))
        works = [dict(name=str(i), lines=t["parsed"]["20"]) for i, t in enumerate(tales)]
    items = make_items(works, a.seed)
    rec = dict(exp="I27l", tag=a.tag, works=len(works), lines=sum(len(w["lines"]) for w in works), n=len(items),
               result=run(works, items, (10, 50, 200, 0)))
    global NLP
    import spacy
    NLP = spacy.load("ja_ginza")
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
                     ("広げる 作品の属性 λ3 分ける + 頻度で割り引く", dict(spread=3.0, split=True, freq_discount=True))):
        rec["spread"][name] = run(works, items, (0,), **kw)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
