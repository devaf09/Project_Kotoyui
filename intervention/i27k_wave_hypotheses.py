"""介入の研究 実験I27k（試験の一覧 B11）: 係り先の仮説を波として重ねて持ち、後の文脈で干渉させて収束させる

持ち主（2026-11-07）: 「複数の構文解釈の仮説を『波』として重ね合わせて同時に保持して、後続の文脈が入力された瞬間に、
全体として最も矛盾のない（干渉し合って波幅が最大化する）仮説に大きく反応しつつ、介入の違和感検出が動作できると判断を早くできるかも」。
問い: 童話の曖昧な修飾 M の係り先（I27c〜I27j と同じ手の印 73 か所）。
仕組み:
  - 候補ごとに振幅（複素数）を持つ。初めは近い方に小さな振幅 ε（I27c の既定「近い方」）、ほかは 0。
  - 同じ話の文を 1 つずつ読む（後ろの文 → 前の文 → ほかの版。どれも試しの文そのものは除く）。
    読んだ文に「候補 u + 修飾:M」があれば、u の振幅に波を足す。
      きれいな根拠（曖昧でない句から来た）: 位相 0 で足す。同じ向きの根拠が重なると、明るさ（振幅の 2 乗）は数の 2 乗で育つ。
      写し（曖昧な句の GiNZA の係り、I27d）: 文ごとに位相がそろわない（でたらめな位相）として足す。重ねても明るさは数に比例するだけ。
    I27d では写しを規則で除いたが、ここでは規則を書かず、位相のそろい方の違いだけで効き方を変える。
  - 明るさの割合 = 仮説の確からしさ。いちばん明るい仮説の割合が q 以上になり、最後までそのままなら、そこで「収束した」とする。
  - 違和感（介入のきっかけ）の候補 2 つを別々に数える: (1) 読み終えても ε より明るい仮説が無い（根拠が来ない = 静か）、
    (2) 一度 q を超えた仮説が途中で入れ替わった（考え直し）。どちらが誤りを多く含むかを見て、違和感のある問いだけ聞き返す（I27b の形）。
比べ: 話の板で 1 つに決まれば選ぶ・決まらなければ近い方（I27c の規則、写しを含む / 除く）。
物差し: 正解率（全部・短い版）、何文目で収束したか（全部読む場合との比）、違和感のある問いと無い問いの誤りの割合。
結果: --out（非公開の置き場）。
実行: python intervention/i27k_wave_hypotheses.py --tales <tales_v1_parsed.json> --labels <i27c_labels.json> --out <jsonl>
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i27a_story_check import story_boards  # noqa: E402
from i27c_compress_units import LEVELS, candidates, modifiers, story_has, units  # noqa: E402
from i27d_attention_boards import clean_board, copy_board  # noqa: E402


def stream(tale, lv, si):
    """試しの文を除いた、読む順の文: 同じ版の後ろ → 同じ版の前（近い順）→ ほかの版（長い版から）。"""
    same = tale["parsed"][lv]
    out = [(lv, j) for j in range(si + 1, len(same))] + [(lv, j) for j in range(si - 1, -1, -1)]
    for l2 in LEVELS:
        if l2 != lv:
            out += [(l2, j) for j in range(len(tale["parsed"].get(l2, [])))]
    return out


def sent_boards(tale, lv, j, cache):
    k = (lv, j)
    if k not in cache:
        one = {"parsed": {lv: [tale["parsed"][lv][j]]}}
        full = story_boards(one, (lv,))
        cb = copy_board(one, (lv,))
        cache[k] = (clean_board(full, cb), cb)
    return cache[k]


def build(tales, labels):
    items = []
    for ti, t in enumerate(tales):
        cache = {}
        for lv in LEVELS:
            for si, s in enumerate(t["parsed"].get(lv, [])):
                toks = s["toks"]
                tok2u, us = units(toks)
                for ui, gu in modifiers(toks, tok2u, us):
                    cand, _ = candidates(toks, tok2u, us, ui)
                    if len(cand) < 2:
                        continue
                    pid = f"{ti}.{lv}.{si}.{us[ui]['head']}"
                    g = labels.get(pid)
                    if g is None or g == "対象外":
                        continue
                    mlem = toks[us[ui]["head"]]["lemma"]
                    ev = []                                      # 読む順の文ごとの (きれいな根拠, 写し) を候補ごとに
                    for l2, j in stream(t, lv, si):
                        cl, cb = sent_boards(t, l2, j, cache)
                        ev.append([(story_has(cl, toks, us, u, mlem), story_has(cb, toks, us, u, mlem)) for u in cand])
                    items.append(dict(id=pid, level=lv, gold=g, cand=[us[u]["text"] for u in cand], ev=ev))
    return items


def right(x, r):
    return x["gold"] == "両方" or r == x["gold"]


def wave(x, rng, eps=0.3, q=0.6, copy_phase="random", w_copy=1.0):
    n = len(x["cand"])
    amp = [complex(eps if k == 0 else 0.0) for k in range(n)]
    hist = []
    for row in x["ev"]:
        for k, (c, p) in enumerate(row):
            if c:
                amp[k] += c                                       # きれいな根拠は位相 0
            if p and w_copy:
                for _ in range(p):
                    ph = rng.uniform(0, 2 * math.pi) if copy_phase == "random" else 0.0
                    amp[k] += w_copy * cmath.exp(1j * ph)
        b = [abs(a) ** 2 for a in amp]
        tot = sum(b)
        top = max(range(n), key=lambda k: (b[k], -k))
        lit = b[top] > eps ** 2 + 1e-9                          # ε だけの明るさは根拠ではない（収束に数えない）
        hist.append((top, b[top] / tot if lit else 0.0))
    b = [abs(a) ** 2 for a in amp]
    top = max(range(n), key=lambda k: (b[k], -k))
    # 収束: いちばん明るい仮説が q 以上で、そのあと最後まで同じ
    conv = len(hist)
    for i in range(len(hist) - 1, -1, -1):
        if hist[i][0] == top and hist[i][1] >= q:
            conv = i
        else:
            break
    flipped = any(h[1] >= q and h[0] != top for h in hist)
    silent = b[top] <= eps ** 2 + 1e-9
    return x["cand"][top], conv + 1, len(hist), flipped, silent


def rule(x, use_copy):
    tot = [0] * len(x["cand"])
    for row in x["ev"]:
        for k, (c, p) in enumerate(row):
            tot[k] += c + (p if use_copy else 0)
    pos = [k for k, v in enumerate(tot) if v > 0]
    return x["cand"][pos[0] if len(pos) == 1 else 0]


def evaluate(items, seeds=6, **kw):
    acc, acc_s, conv, asked = [], [], [], []
    grp = {g: [0, 0] for g in ("静か", "考え直し", "そのほか")}
    short = [x for x in items if x["level"] != "20"]
    for sd in range(seeds):
        rng = random.Random(sd)
        res = {x["id"]: wave(x, rng, **kw) for x in items}
        acc.append(sum(right(x, res[x["id"]][0]) for x in items) / len(items))
        acc_s.append(sum(right(x, res[x["id"]][0]) for x in short) / len(short))
        lit = [x for x in items if not res[x["id"]][4]]
        conv.append(sum(res[x["id"]][1] / max(1, res[x["id"]][2]) for x in lit) / max(1, len(lit)))   # 根拠の来た問いだけ
        for x in items:
            r = res[x["id"]]
            g = "静か" if r[4] else ("考え直し" if r[3] else "そのほか")
            grp[g][0] += 1
            grp[g][1] += not right(x, r[0])
        asked.append(sum(1 if res[x["id"]][4] else right(x, res[x["id"]][0]) for x in items) / len(items))
    m = lambda v: sum(v) / len(v)                               # noqa: E731
    return dict(acc=m(acc), acc_min=min(acc), acc_short=m(acc_s), read_frac_lit=m(conv),
                groups={g: dict(n=v[0] / seeds, err=v[1] / max(1, v[0])) for g, v in grp.items()},
                acc_if_ask_silent=m(asked))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tales = json.load(open(a.tales, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    items = build(tales, labels)
    short = [x for x in items if x["level"] != "20"]
    rec = dict(exp="I27k", n=len(items), n_short=len(short))
    rec["rule"] = {name: dict(acc=sum(right(x, rule(x, uc)) for x in items) / len(items),
                              acc_short=sum(right(x, rule(x, uc)) for x in short) / len(short))
                   for name, uc in (("規則 写しも数える", True), ("規則 写しを除く（I27d の層 1）", False))}
    rec["near"] = dict(acc=sum(right(x, x["cand"][0]) for x in items) / len(items),
                       acc_short=sum(right(x, x["cand"][0]) for x in short) / len(short))
    rec["wave"] = {}
    for name, kw in (("波 写しはでたらめな位相", dict()), ("波 写しも位相 0（そろう）", dict(copy_phase="zero")),
                     ("波 写しを除く", dict(w_copy=0.0)), ("波 q0.8", dict(q=0.8)), ("波 ε0.1", dict(eps=0.1))):
        rec["wave"][name] = evaluate(items, **kw)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
