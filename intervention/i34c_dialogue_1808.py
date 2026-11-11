"""介入の研究 実験I34c: 1,808 作品の台詞で、会話の板と人物ごとの丁寧を数え直す（PC で回す）

持ち主（2026-11-09）: 「Web 小説使ってくれる？」（I34a・I34b は 50 作品で、会話の板が薄かった）
I27v の下ごしらえと同じ取り出し方（台詞の行と、そのすぐあとの行だけを GiNZA で解析）で、
I27t の quotes_of から話し手の正解（gold）も取る。本文は出さない（数と、語・表記だけ）。
作品ごとに数えたもの（--out の JSONL、1 作品 1 行。途中で止めても続きから回せる）:
  1. 隣り合う台詞の問いの型 → 答えの型（I34a と同じ）。話し手が替わると分かる組だけの数も別に。
  2. chat_bot2 の会話の板: 始めの感動詞（問いの型, 肯定/否定/相槌）、聞き返しの形、問いの語の問いへの否定の表記、
     問いの語の並び（述語, 助詞 → 問いの語）。
  3. 人物ごとの丁寧: 話し手の分かる台詞の言い切りの丁寧。次の台詞の丁寧を、その人物のこれまでの割合・作品の台詞の割合・
     全体の割合で当てたときの損失（--merge で）。相手の丁寧に合わせるか（問いの台詞の丁寧 → 答えの台詞の丁寧、話し手が替わる組）。
--merge で、作品ごとの数を足し、chat_bot2.py の --dialogue に渡す板（dialogue_board_1808_wn.json）と結果を出す。
CPU だけで回す（CUDA_VISIBLE_DEVICES=）。Web 小説は _wn。結果は非公開の置き場だけ。
実行: python intervention/i34c_dialogue_1808.py --corpus <webNW_txt_v1> --out <i34c_counts_wn.jsonl> --jobs 10
      python intervention/i34c_dialogue_1808.py --merge <i34c_counts_wn.jsonl> --board <dialogue_board_1808_wn.json> --stats <i34c_wn.jsonl>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_nlp = None


def _echo_form(rest):
    """聞き返しの形: 名詞のあとの 4 文字まで、最初の ？ で切る。"""
    r = rest[:4]
    for q in ("？", "?"):
        if q in r:
            r = r[: r.index(q) + 1]
    return r.rstrip("。…　 ")


def _init():
    global _nlp
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import spacy
    _nlp = spacy.load("ja_ginza")


def qtext(T, text):
    e = T.quote_end(text)
    return text[: e + 1] if e >= 0 else text


def do_work(args):
    import i27t_first_person as T
    import i27v_extract_quotes as V
    import i32_function_layers as L
    import i34_dialogue_board as D
    corpus, wid, eps = args
    lines = V.work_lines(corpus, wid, eps)
    need = set()
    for li, x in enumerate(lines):
        if T.quote_end(x) >= 0:
            need.add(li)
            if li + 1 < len(lines):
                need.add(li + 1)
    idx = sorted(need)
    docs = _nlp.pipe([lines[i] for i in idx], batch_size=64)
    parsed = {i: V.to_toks(d) for i, d in zip(idx, docs)}
    w = dict(lines=[dict(text=x, toks=parsed.get(i, [])) for i, x in enumerate(lines)])
    gold = {q["li"]: q["gold"] for q in T.quotes_of(w)}
    shape, shape_sw = Counter(), Counter()
    start, echo, dunno, askpat = Counter(), Counter(), Counter(), Counter()
    pol_seq = []                     # (話し手 or None, 丁寧) 台詞の順
    acc = Counter()                  # 相手に合わせるか: (問いの丁寧, 答えの丁寧) 話し手が替わる組
    Ls = w["lines"]
    isq = [T.quote_end(x["text"]) >= 0 for x in Ls]

    def marks(toks):
        kids = L.kids_of(toks)
        root = next((t for t in toks if t["dep"] == "ROOT"), None)
        return L.layers(kids, root)[1] if root is not None else None

    for i in range(len(Ls)):
        if not isq[i]:
            continue
        m = marks(Ls[i]["toks"])
        if m is not None:
            pol_seq.append((gold.get(i), int(m["丁寧"])))
        if i + 1 >= len(Ls) or not isq[i + 1]:
            continue
        a, b = Ls[i], Ls[i + 1]
        ta, tb = qtext(T, a["text"]), qtext(T, b["text"])
        qt = D.q_type(a["toks"], ta).split(":")[0]
        qp = {t["lemma"] for t in a["toks"] if t["pos"] in ("VERB", "ADJ")}
        qn = {t["text"] for t in a["toks"] if t["pos"] in D.NOUN and len(t["text"]) >= 2}
        sh = D.a_shape(qt, b["toks"], tb, qp, qn)
        sw = gold.get(i) is not None and gold.get(i + 1) is not None and gold[i] != gold[i + 1]
        shape[qt] += 1
        for s in sh:
            shape[qt + "|" + s.split(":")[0]] += 1
            if sw:
                shape_sw[qt + "|" + s.split(":")[0]] += 1
        if sw:
            shape_sw[qt] += 1
            mb = marks(b["toks"])
            if m is not None and mb is not None:
                acc[f"{int(m['丁寧'])}>{int(mb['丁寧'])}"] += 1
        # chat_bot2 の会話の板（chat_bot2.dialogue_board と同じ数え方）
        at = b["toks"]
        ab = D.body(tb)
        first = next((t for t in at if t["pos"] not in ("PUNCT", "SYM")), None)
        neg = any(t["lemma"] in ("ない", "ぬ") and t["pos"] == "AUX" for t in at)
        ap_ = {t["lemma"] for t in at if t["pos"] in ("VERB", "ADJ")}
        if first is not None and first["pos"] == "INTJ":
            if qt == "問いでない":
                start[f"{qt}|相槌|{first['text']}"] += 1
            elif qp & ap_:
                start[f"{qt}|{'否定' if neg else '肯定'}|{first['text']}"] += 1
            elif qt == "問いの語" and neg:
                start[f"{qt}|否定|{first['text']}"] += 1
        for t in a["toks"]:
            if t["pos"] in D.NOUN and len(t["text"]) >= 2 and t["text"] in ab:
                rest = ab.split(t["text"], 1)[1]
                if rest.startswith(("って", "？", "?")):
                    echo[_echo_form(rest)] += 1
        if qt != "問いでない":
            for t in a["toks"]:
                if t["lemma"] in D.ASK and t["pos"] in ("PRON", "NOUN", "ADV"):
                    cs = [k["text"] for k in a["toks"] if k["head"] == t["i"] and k["dep"] == "case"]
                    hd = a["toks"][t["head"]]
                    if hd["pos"] in ("VERB", "ADJ"):
                        askpat[f"{hd['lemma']}|{cs[0] if cs else ''}|{t['text']}"] += 1
        if qt == "問いの語" and neg:
            kids = L.kids_of(at)
            for t in at:
                if t["pos"] in ("VERB", "ADJ") and t["dep"] == "ROOT":
                    conn, mm, surf = L.layers(kids, t)
                    if mm["否定"]:
                        dunno[surf + "|" + str(int(mm["丁寧"]))] += 1
    rec = dict(work_id=wid, shape=shape, shape_sw=shape_sw, start=start, echo=echo, dunno=dunno, askpat=askpat,
               pol_seq=pol_seq, accommodate=acc)
    return wid, rec


def merge(a):
    tot = defaultdict(Counter)
    seqs = []
    n = 0
    with open(a.merge, encoding="utf-8") as f:
        for x in f:
            r = json.loads(x)
            n += 1
            for k in ("shape", "shape_sw", "start", "echo", "dunno", "askpat", "accommodate"):
                tot[k].update(r[k])
            seqs.append(r["pol_seq"])
    # 人物ごとの丁寧: 次の台詞の丁寧を当てる損失（作品を通して順に）
    g = sum(p for s in seqs for _, p in s) / max(1, sum(len(s) for s in seqs))
    ll = Counter()

    def lg(p, y):
        p = min(max(p, 1e-3), 1 - 1e-3)
        return -math.log(p if y else 1 - p)
    for s in seqs:
        per = defaultdict(lambda: [0, 0])
        work = [0, 0]
        for who, y in s:
            if who is not None:
                pw = (work[1] + g) / (work[0] + 1)                  # 作品の台詞の割合（全体を 1 台詞ぶんで混ぜる）
                ps = (per[who][1] + pw) / (per[who][0] + 1)         # その人物の割合（作品の割合を 1 台詞ぶんで混ぜる）
                ll["n"] += 1
                ll["全体の割合"] += lg(g, y)
                ll["作品の台詞の割合"] += lg(pw, y)
                ll["人物の割合"] += lg(ps, y)
                ll["人物の割合・当たり"] += (ps >= 0.5) == bool(y)
                ll["丁寧の台詞"] += y
                per[who][0] += 1
                per[who][1] += y
            work[0] += 1
            work[1] += y
    nn = ll.pop("n")
    acc = tot["accommodate"]
    p11 = acc["1>1"] / max(1, acc["1>1"] + acc["1>0"])
    p01 = acc["0>1"] / max(1, acc["0>1"] + acc["0>0"])

    def shape_tab(c):
        qs = [q for q in c if "|" not in q]
        return {q: dict(n=c[q], **{k.split("|", 1)[1]: round(v / c[q], 4) for k, v in c.items() if k.startswith(q + "|")}) for q in qs}
    rec = dict(exp="I34c", 作品=n, 台詞の組=sum(v for k, v in tot["shape"].items() if "|" not in k),
               答えの型=shape_tab(tot["shape"]), 答えの型_話し手が替わる組=shape_tab(tot["shape_sw"]),
               人物ごとの丁寧=dict(台詞=nn, 丁寧の割合=round(ll.pop("丁寧の台詞") / nn, 4),
                             **{k: round(v / nn, 4) for k, v in ll.items()}),
               相手に合わせる=dict(問いが丁寧なら答えも丁寧=round(p11, 4), 問いが普通なら答えが丁寧=round(p01, 4), 組=sum(acc.values())))
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    board = dict(start=dict(tot["start"].most_common(3000)), echo=dict(tot["echo"].most_common(200)),
                 dunno=dict(tot["dunno"].most_common(500)), askpat=dict(tot["askpat"].most_common(20000)))
    with open(a.board, "w", encoding="utf-8") as f:
        json.dump(board, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge")
    ap.add_argument("--board")
    ap.add_argument("--stats")
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    eps = OrderedDict()
    with open(os.path.join(a.corpus, "index.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            eps.setdefault(r["work_id"], []).append(r["episode_id"])
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            for x in f:
                try:
                    done.add(json.loads(x)["work_id"])
                except Exception:
                    pass
    todo = [(a.corpus, w, e) for w, e in eps.items() if w not in done]
    if a.limit:
        todo = todo[:a.limit]
    print("作品", len(eps), "済み", len(done), "これから", len(todo), flush=True)
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.jobs, initializer=_init) as pool, open(a.out, "a", encoding="utf-8") as out:
        for k, (wid, rec) in enumerate(pool.imap_unordered(do_work, todo, chunksize=1), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if k % 20 == 0:
                print(k, len(todo), f"{time.time() - t0:.0f}s", flush=True)
    print("おわり", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
