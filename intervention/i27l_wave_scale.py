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
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i27a_story_check import story_boards  # noqa: E402

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
                    pool = [d for d in common if d != h]
                elif dk == "紛らわしい" or h_else:
                    pool = [d for d, ls in m_on[m].items() if d != h and ls - {li}]
                else:
                    pool = []
                if not pool:
                    continue
                d = rng.choice(pool)
                for side in ("遠い方が正解", "近い方が正解"):
                    cand = [d, h] if side == "遠い方が正解" else [h, d]
                    items.append(dict(work=wi, line=li, m=m, cand=cand, gold=h, distract=dk, side=side))
        w["_boards"] = boards
    return items


def evidence(works, x, win):
    bs = works[x["work"]]["_boards"]
    key = "修飾:" + x["m"]
    return [[bs[lj].get(c, {}).get(key, 0) for c in x["cand"]] for lj in order(x["line"], len(bs), win)]


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


def run(works, items, wins):
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
                ev = evidence(works, x, win)
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
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
