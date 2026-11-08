"""介入の研究 I27v の下ごしらえ: 1,808 作品の台詞を GiNZA で解析し、話し手の正解と語尾・一人称だけを取り出す（PC で回す）

I27t の quotes_of と同じ取り出し方。解析するのは、台詞の行と、そのすぐあとの行だけ（話し手の目印はそこにしかない）。
語りの一人称（＜語り手＞ の一人称の板）は、解析せずに語りの行の文字で数える。
出力は作品ごとに 1 行の JSONL（本文は入れない: 語尾の語・一人称の語・話し手の名前・行の番号だけ）。途中で止めても、続きから回せる。
CPU だけで回す（CUDA_VISIBLE_DEVICES=）。
実行: python intervention/i27v_extract_quotes.py --corpus <webNW_txt_v1> --out <quotes_1808_wn.jsonl> --jobs 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NARR_WORDS = ("俺", "おれ", "オレ", "私", "わたし", "僕", "ぼく", "ボク", "あたし", "わたくし")
_nlp = None


def _init():
    global _nlp
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import spacy
    _nlp = spacy.load("ja_ginza")


def to_toks(doc):
    return [dict(i=t.i, text=t.text, lemma=t.lemma_, pos=t.pos_, dep=t.dep_, head=t.head.i) for t in doc]


def work_lines(corpus, wid, eps):
    lines = []
    for e in eps:
        p = os.path.join(corpus, wid, e + ".txt")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            lines += [x.strip() for x in f if x.strip()]
    return lines


def do_work(args):
    import i27t_first_person as T
    corpus, wid, eps = args
    lines = work_lines(corpus, wid, eps)
    need = set()
    for li, x in enumerate(lines):
        if T.quote_end(x) >= 0:
            need.add(li)
            if li + 1 < len(lines):
                need.add(li + 1)
    idx = sorted(need)
    docs = _nlp.pipe([lines[i] for i in idx], batch_size=64)
    parsed = {i: to_toks(d) for i, d in zip(idx, docs)}
    w = dict(lines=[dict(text=x, toks=parsed.get(i, [])) for i, x in enumerate(lines)])
    qs = T.quotes_of(w)
    narr = {}
    for x in lines:
        if x.startswith("「"):
            continue
        for word in NARR_WORDS:
            c = x.count(word)
            if c and word in T.CLS:
                narr[T.CLS[word]] = narr.get(T.CLS[word], 0) + c
    rec = dict(work_id=wid, n_lines=len(lines), narr=narr,
               quotes=[dict(li=q["li"], f=[[k[0], k[1], v] for k, v in q["f"].items()],
                            fp=[[k[0], k[1], v] for k, v in q["fp"].items()], gold=q["gold"], kind=q["kind"]) for q in qs])
    return wid, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    eps = OrderedDict()
    with open(os.path.join(a.corpus, "index.tsv"), encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
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
