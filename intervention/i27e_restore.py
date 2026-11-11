"""介入の研究 実験I27e: 縮めた文を戻す — ネストで戻す可逆復元と、値の帯（属性）と周りの語のつながりから似た内容を戻す復元

持ち主（2026-11-07）: 「算術のときネストとバッファ板で元の式を戻す方法を覚えてる？ そのバッファ板から元の値を戻す可逆復元と、
バッファが持っている属性と周辺単語の関連から似た内容を復元する方法を考えたんだけど」。
  A. 可逆復元: 20 行版の文を自分で縮める。まとまり（I27c）を節にし、係り受け（GiNZA）の木の下から順に、係る節を係り先の節へ畳んでバッファ板を作る。
     バッファ板 = 値の帯（係った語の属性: 修飾:M・の:A・する:V など）+ 参照の帯（ネスト: 畳んだ子の板の番号を、左から順に）。
     最後の 1 枚から参照の帯をたどって、元の文（トークンの並び）に戻るか。参照の帯は板の番号の明るさ（1 つだけ 1）で持ち、雑音 σ を足していちばん明るい番号を読む（I19 と同じ）。
  B. 似た内容の復元: 名詞句の中のネストを捨て、値の帯だけにする（名詞句の中の修飾 M は、句の中心のバッファ板の属性として平らに残る）。
     M が句の中のどの名詞に係っていたかを、周りの語とのつながりで戻す:
       - 平ら（中心に係ったとする）、近い方、
       - 話の板（同じ話の、この文を除いたすべての版の文から作る。曖昧な句の写しは除く、I27d の層 1）、
       - 一般常識の語板（I26 の合い方: 修飾先のクラス・性質など）。
     物差し: 元の木（GiNZA）と同じ係りに戻った割合、手の正解の印（I27c・I27d、意味の上で同じ内容か）と合った割合。
  C. 持ち主の短い版は、ネストの無い縮め。20 行版の内容（名詞と属性の組）のうち、短い版にも残っている割合を数える（伸ばして戻せる上限の目安）。
結果: ../Project_Kotoyui_private/intervention/results/i27e.jsonl
実行: python intervention/i27e_restore.py --tales <tales_v1_parsed.json> --labels <i27c_labels.json> --boards <content_boards.json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i26_ambig_judge import Boards, compat  # noqa: E402
from i27a_story_check import story_boards  # noqa: E402
from i27c_compress_units import LEVELS, candidates, modifiers, story_has, units  # noqa: E402
from i27d_attention_boards import clean_board, copy_board  # noqa: E402


def unit_tree(toks, tok2u, us):
    """節 = まとまり。親 = まとまりの中心の係り先のまとまり（根は -1）。"""
    par = []
    for ui, u in enumerate(us):
        h = toks[u["head"]]["head"]
        p = tok2u[h]
        par.append(-1 if h == u["head"] or p == ui else p)
    for ui in range(len(us)):                                    # 輪があれば切る
        seen, v = set(), ui
        while v != -1 and v not in seen:
            seen.add(v)
            v = par[v]
        if v != -1:
            par[ui] = -1
    return par


def compress(toks, tok2u, us):
    """バッファ板の列。板 = dict(unit, nest=[子の板の番号か、自分の字（負の番号）を左から], value=[属性])。最後の板たちが根。"""
    par = unit_tree(toks, tok2u, us)
    kids = {ui: [] for ui in range(len(us))}
    for ui, p in enumerate(par):
        if p != -1:
            kids[p].append(ui)
    boards, bid = [], {}

    def build(ui):
        for c in kids[ui]:
            build(c)
        order = sorted([("板", v, us[v]["ids"][0]) for v in kids[ui]] + [("字", k, k) for k in us[ui]["ids"]], key=lambda e: e[2])
        nest = [-(k + 1) if e == "字" else bid[k] for e, k, _ in order]                 # 自分の字は負の番号、子は板の番号
        value = []
        for c in kids[ui]:
            w = toks[us[c]["head"]]
            value.append(f"{w['dep']}:{w['lemma']}")
        bid[ui] = len(boards)
        boards.append(dict(unit=ui, nest=nest, value=value))
    roots = [ui for ui in range(len(us)) if par[ui] == -1]
    for r in roots:
        build(r)
    return boards, [bid[r] for r in roots]


def restore(boards, roots, us, sigma, rng):
    n = len(boards)

    def read(ref):
        if sigma <= 0:
            return ref
        v = [rng.gauss(0, sigma) for _ in range(n)]
        v[ref] += 1.0
        return max(range(n), key=lambda k: v[k])
    out, guard = [], [0]

    def walk(b):
        guard[0] += 1
        if guard[0] > 4 * n:
            return
        for r in boards[b]["nest"]:
            if r < 0:
                out.append(-r - 1)
            else:
                walk(read(r))
    for r in sorted(roots, key=lambda b: us[boards[b]["unit"]]["ids"][0]):
        walk(read(r))
    return out


def content(tale, lv):
    sb = story_boards(tale, (lv,))
    return {(k, a) for k, c in sb.items() for a in c if not a.startswith("修飾:")}


def without(tale, lv, si):
    t2 = dict(tale)
    t2["parsed"] = {k: [s for j, s in enumerate(v) if not (k == lv and j == si)] for k, v in tale["parsed"].items()}
    return t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--boards", required=True)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i27e.jsonl")
    a = ap.parse_args()
    tales = json.load(open(a.tales, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    B = Boards(a.boards, tales)
    rng = random.Random(0)
    # A. 可逆復元
    sig = (0.0, 0.2, 0.3, 0.5)
    exact = {s: 0 for s in sig}
    n_sent = n_tok = n_board = n_root = 0
    for t in tales:
        for s in t["parsed"]["20"]:
            toks = s["toks"]
            tok2u, us = units(toks)
            boards, roots = compress(toks, tok2u, us)
            n_sent += 1
            n_tok += len(toks)
            n_board += len(boards)
            n_root += len(roots)
            for sg in sig:
                exact[sg] += restore(boards, roots, us, sg, rng) == list(range(len(toks)))
    recA = dict(sentences=n_sent, tokens=n_tok, boards=n_board, roots_per_sentence=n_root / max(1, n_sent),
                exact={str(k): v / n_sent for k, v in exact.items()})
    # B. ネストを捨てて、値の帯と周りの語から名詞句の中の係りを戻す（20 行版の曖昧な所）
    items = []
    for ti, t in enumerate(tales):
        for si, s in enumerate(t["parsed"]["20"]):
            toks = s["toks"]
            tok2u, us = units(toks)
            mods = list(modifiers(toks, tok2u, us))
            if not mods:
                continue
            t2 = without(t, "20", si)
            full = story_boards(t2, LEVELS)
            sb = clean_board(full, copy_board(t2, LEVELS))
            for ui, gu in mods:
                cand, _ = candidates(toks, tok2u, us, ui)
                if len(cand) < 2 or gu not in cand:
                    continue
                pid = f"{ti}.20.{si}.{us[ui]['head']}"
                mt = toks[us[ui]["head"]]
                mlem = mt["lemma"]
                ev = {u: story_has(sb, toks, us, u, mlem) for u in cand}
                pos = [u for u in cand if ev[u] > 0]
                story = pos[0] if len(pos) == 1 else None
                bm = B.get(mt["text"], mlem)
                cs = {u: compat(bm, mt, B.get(us[u]["text"], toks[us[u]["head"]]["lemma"]), mlem) for u in cand}
                best = max(cand, key=lambda u: (cs[u], -cand.index(u)))
                general = best if cs[best] > 0 and sorted(cs.values())[-1] != sorted(cs.values())[-2] else None
                rd = dict(flat=cand[-1], near=cand[0], story=story if story is not None else cand[-1],
                          story_near=story if story is not None else cand[0],
                          general=general if general is not None else cand[-1],
                          story_general=story if story is not None else (general if general is not None else cand[-1]))
                g = labels.get(pid)
                items.append(dict(id=pid, orig=us[gu]["text"], gold=g, story_decided=story is not None,
                                  general_decided=general is not None,
                                  read={k: us[v]["text"] for k, v in rd.items()}))
    recB = dict(n=len(items), story_decided=sum(x["story_decided"] for x in items),
                general_decided=sum(x["general_decided"] for x in items))
    gl = [x for x in items if x["gold"] not in (None, "対象外")]
    for k in items[0]["read"] if items else []:
        recB[f"orig_{k}"] = sum(x["read"][k] == x["orig"] for x in items) / len(items)
        recB[f"gold_{k}"] = sum(x["gold"] == "両方" or x["read"][k] == x["gold"] for x in gl) / max(1, len(gl))
    recB["n_gold"] = len(gl)
    recB["gold_orig"] = sum(x["gold"] == "両方" or x["orig"] == x["gold"] for x in gl) / max(1, len(gl))
    # C. 持ち主の短い版に残る内容（名詞と属性の組、修飾 以外: する・される・で・に・の・連用）
    recC = {}
    for lv in ("12", "5", "1"):
        r = []
        for t in tales:
            c20, cl = content(t, "20"), content(t, lv)
            r.append((len(c20 & cl), len(c20), len(cl)))
        recC[lv] = dict(recall=sum(x[0] for x in r) / sum(x[1] for x in r), precision=sum(x[0] for x in r) / max(1, sum(x[2] for x in r)))
    rec = dict(exp="I27e", A=recA, B=recB, C=recC, items=items)
    print(json.dumps({k: v for k, v in rec.items() if k != "items"}, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
