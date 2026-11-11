"""介入の研究 実験I27m: 前後の行の属性を 1 枚の想定属性板に集め、そこから 1 行を作って差し込む

持ち主（2026-11-08）: 「前後の１０行くらい属性を集めて１つの想定属性板を作って、それを展開して１行追加したものを１０個ほど例にだしてくれないかな？」
作り方（Web 小説は _wn を付け、非公開の置き場だけに置く。本文はコードに書かない）:
  - 想定属性板: ある行 L を抜き、L の前後 w 行（既定 10）の名詞ごとに、属性（修飾: / する: / される: / で・に:）を数える。
    行ごとに 1 票（同じ行の中の重なりは数えない）、近い行は少しだけ重い（+0.1 / 距離）。属性ごとに、文の中で現れた形（活用した形と助詞）も覚える。
  - 展開（関連性で 1 行作る）:
      主語 S = する: を持つ名詞のうち、板の重みの合計がいちばん大きいもの（前後でいちばん話題の名詞）。述語 P = S の する: でいちばん重いもの。
      目的語 O = P を される: に持つ、S 以外でいちばん重い名詞（あれば）。無ければ P の で・に: を持つ名詞（あれば）。
      修飾 = S（無ければ O）の 修飾: でいちばん重いもの。助詞は板で S・O・で に が付いていた形でいちばん多いもの。
      並べ方は日本語の基本の順（修飾 主語+助詞 [補語+助詞] [目的語+を] 述語。）にする。
  - 比べ: 抜いた元の行と、作った行の内容語（見出し）の重なり（F1）。目安に、すぐ前の行をそのまま写した場合の F1 も出す。
実行: python intervention/i27m_board_line.py --cache <解析済み JSON> --n 10 --out <md> [--win 10] [--seed 0] [--stats <jsonl>]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

NOUN = ("NOUN", "PROPN", "PRON")
PRED = ("VERB", "ADJ")
CONTENT = ("NOUN", "PROPN", "PRON", "VERB", "ADJ", "ADV", "NUM")
SKIP = {"「", "」", "『", "』", "（", "）"}


def kids(toks):
    k = defaultdict(list)
    for i, t in enumerate(toks):
        if t["head"] != i:
            k[t["head"]].append(i)
    return k


def pred_surface(toks, i, ks):
    """述語の形: 述語の語と、それに付く助動詞・て などをつないだもの（読点・句点は落とす）。"""
    aux = {j for j in ks.get(i, []) if toks[j]["dep"] in ("aux", "fixed") and toks[j]["pos"] != "PUNCT"}
    seq = [i]
    while seq[-1] + 1 in aux:                      # 述語のすぐ後ろに続く助動詞だけ（間の空いたものは取らない）
        seq.append(seq[-1] + 1)
    return "".join(toks[j]["text"] for j in seq)


def is_final(toks, i):
    """述語が文の終わり（行の根で、後ろが句点・行末・かぎかっこ）か。終わりの形だけを 1 行の述語に使う。"""
    if toks[i]["dep"] != "ROOT":
        return False
    j = i + 1
    while j < len(toks) and toks[j]["dep"] in ("aux", "fixed") and toks[j]["pos"] != "PUNCT":
        j += 1
    return j >= len(toks) or toks[j]["text"] in ("。", "」", "！", "？", "…")


def mod_surface(toks, i, ks):
    seq = [i] + [j for j in ks.get(i, []) if toks[j]["dep"] in ("aux", "fixed") and toks[j]["pos"] != "PUNCT"]
    seq.sort()
    return "".join(toks[j]["text"] for j in seq)


def particle(toks, i, ks, obl=False):
    ps = [toks[j]["text"] for j in ks.get(i, []) if toks[j]["dep"] == "case" and toks[j]["pos"] == "ADP"
          and not (obl and toks[j]["text"] in ("は", "も"))]
    return ps[-1] if ps else ""


def noun_surface(toks, i, ks):
    """名詞の形: 前に付く複合語（compound）をつないだもの（例: 名字 + さん）。"""
    seq = [j for j in ks.get(i, []) if j < i and toks[j]["dep"] == "compound" and toks[j]["pos"] in NOUN] + [i]
    seq.sort()
    return "".join(toks[j]["text"] for j in seq)


def line_attrs(toks):
    """行 → [(名詞, 属性, 形, 助詞)]。"""
    ks = kids(toks)
    out = []
    for i, t in enumerate(toks):
        if t["pos"] not in NOUN or t["text"] in SKIP or t["dep"] == "compound":
            continue
        nm = noun_surface(toks, i, ks)
        for j in ks.get(i, []):
            c = toks[j]
            if c["pos"] in PRED and c["dep"] in ("acl", "amod") and j < i:
                out.append((nm, "修飾:" + c["lemma"], mod_surface(toks, j, ks), ""))
        h = t["head"]
        if h == i:
            continue
        p = toks[h]
        if p["pos"] not in PRED:
            continue
        role = {"nsubj": "する:", "obj": "される:", "obl": "で・に:"}.get(t["dep"])
        if role:
            f = pred_surface(toks, h, ks) if is_final(toks, h) else ""
            out.append((nm, role + p["lemma"], f, particle(toks, i, ks, role == "で・に:")))
    return out


def build_board(lines, li, win):
    """抜いた行 li の前後 win 行から想定属性板を作る。"""
    w = defaultdict(Counter)                       # 名詞 → 属性 → 重み
    form = defaultdict(Counter)                    # 属性 → 形
    part = defaultdict(Counter)                    # (名詞, 役) → 助詞
    text = {}                                      # 名詞の見出し → 表記
    for lj in range(max(0, li - win), min(len(lines), li + win + 1)):
        if lj == li:
            continue
        toks = lines[lj]["toks"]
        for t in toks:
            if t["pos"] in NOUN:
                text.setdefault(t["lemma"], t["text"])
        seen = set()
        for n, a, f, p in line_attrs(toks):
            if (n, a) in seen:
                continue
            seen.add((n, a))
            wt = 1.0 + 0.1 / abs(lj - li)               # 行ごとに 1 票（10 行を束ねる）、近い行は少しだけ重い
            w[n][a] += wt
            if f:
                form[a][f] += wt
            if p:
                part[(n, a.split(":")[0])][p] += wt
    return w, form, part, text


def expand(board):
    w, form, part, text = board

    def top(c):
        return max(c, key=lambda k: (c[k], k)) if c else None
    subj = Counter({n: sum(c.values()) for n, c in w.items() if any(a.startswith("する:") for a in c)})   # 話によく出て、何かをする名詞
    subj = +subj
    if not subj:
        return None, {}
    s = top(subj)
    preds = Counter({a.split(":", 1)[1]: v for a, v in w[s].items() if a.startswith("する:")})
    p = top(preds)
    pick = dict(主語=s, 述語=p)
    objs = Counter({n: c.get("される:" + p, 0) for n, c in w.items() if n != s})
    objs = +objs
    obls = Counter({n: c.get("で・に:" + p, 0) for n, c in w.items() if n != s})
    obls = +obls
    o = top(objs)
    b = top(obls)
    mods = Counter({a.split(":", 1)[1]: v for a, v in w[s].items() if a.startswith("修飾:")})
    m_of = s
    if not mods and o:
        mods = Counter({a.split(":", 1)[1]: v for a, v in w[o].items() if a.startswith("修飾:")})
        m_of = o
    m = top(mods)

    def pt(n, role, dflt):
        return top(part.get((n, role), Counter())) or dflt
    head = []
    if m and m_of == s:
        head.append(top(form["修飾:" + m]))
    head.append(text.get(s, s) + pt(s, "する", "が"))
    if b:
        head.append(text.get(b, b) + pt(b, "で・に", "に"))
    if o:
        if m and m_of == o:
            head.append(top(form["修飾:" + m]))
        head.append(text.get(o, o) + "を")
    head.append((top(form["する:" + p]) or p) + "。")                      # 終わりの形を見ていなければ辞書の形
    pick.update(目的語=o, 補語=b, 修飾=m)
    return "".join(head), pick


def content_lemmas(toks):
    return Counter(t["lemma"] for t in toks if t["pos"] in CONTENT)


def f1(a, b):
    m = sum((a & b).values())
    if not m:
        return 0.0
    p, r = m / sum(b.values()), m / sum(a.values())
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--win", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None, help="多くの行で F1 を数えて jsonl に足す（例とは別）")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    rng = random.Random(a.seed)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(a.win, len(w["lines"]) - a.win)
             if any(t["dep"] == "nsubj" for t in w["lines"][li]["toks"])]
    rng.shuffle(sites)
    out = ["# I27m 想定属性板から作った 1 行（非公開）", "",
           f"前後 {a.win} 行（抜いた行を除く）の属性を、近い行ほど重く 1 枚の板に集め、そこから 1 行を作って、抜いた行の位置に差し込んだ。"
           "▶ が作った行、（元）が抜いた元の行。前後は 2 行ずつ見せる。", ""]
    shown = 0
    for wi, li in sites:
        if shown >= a.n:
            break
        lines = works[wi]["lines"]
        board = build_board(lines, li, a.win)
        line, pick = expand(board)
        if not line:
            continue
        shown += 1
        w = board[0]
        tops = sorted(((v, n, at) for n, c in w.items() for at, v in c.items()), reverse=True)[:8]
        out += [f"## {shown}. 作品 {works[wi]['name']} {li} 行目", ""]
        for lj in range(li - 2, li):
            out.append(f"- 　 {lines[lj]['text']}")
        out.append(f"- ▶ **{line}**")
        for lj in range(li + 1, li + 3):
            out.append(f"- 　 {lines[lj]['text']}")
        out += ["", f"- （元）{lines[li]['text']}",
                "- 選んだ語: " + "、".join(f"{k} = {v}" for k, v in pick.items() if v),
                "- 想定属性板の上位: " + "、".join(f"{n}（{at} {v:.2f}）" for v, n, at in tops), ""]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    if a.stats:
        gen = prev = got = 0
        k = 0
        for wi, li in sites[:3000]:
            lines = works[wi]["lines"]
            line, pick = expand(build_board(lines, li, a.win))
            if not line:
                continue
            got += 1
            ref = content_lemmas(lines[li]["toks"])
            k += 1
            gen += f1(ref, Counter(v for v in pick.values() if v))          # 作った行の内容語 = 板から選んだ見出し
            prev += f1(ref, content_lemmas(lines[li - 1]["toks"]))
        rec = dict(exp="I27m", win=a.win, n=k, 作った行のF1=gen / max(1, k), すぐ前の行のF1=prev / max(1, k))
        print(json.dumps(rec, ensure_ascii=False))
        with open(a.stats, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
