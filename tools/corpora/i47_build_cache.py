"""I47 用: 持ち主が集めた作品（資料は公開しない）を、i47_wave_resonance.py の --cache の形にする。
[{name, lines:[{text, toks:[{i,text,lemma,pos,dep,head}]}]}]。i47 が使うのは台詞（「で始まる行）だけなので、台詞の行だけ GiNZA で読み、
地の文は text だけ（toks は空）にする。作品ごとに 1 行の JSON を jsonl に足していき（途中から続けられる）、最後に 1 つの JSON にまとめる。

  python tools/corpora/i47_build_cache.py --corpus <作品の txt の置き場> --out <非公開の置き場>/i47/quotes.jsonl --jobs 16
  python tools/corpora/i47_build_cache.py --merge <非公開の置き場>/i47/quotes.jsonl --json <非公開の置き場>/i47/quotes.json [--limit N]
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict

_nlp = None


def _init():
    global _nlp
    import spacy
    _nlp = spacy.load("ja_ginza")


def to_toks(doc):
    return [dict(i=t.i, text=t.text, lemma=t.lemma_, pos=t.pos_, dep=t.dep_, head=t.head.i) for t in doc]


def do_work(args):
    corpus, wid, eps = args
    lines = []
    for e in eps:
        p = os.path.join(corpus, wid, e + ".txt")
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                lines += [x.strip() for x in f if x.strip()]
    idx = [i for i, x in enumerate(lines) if x.startswith("「")]
    parsed = {}
    for i, d in zip(idx, _nlp.pipe([lines[i][:400] for i in idx], batch_size=64)):
        parsed[i] = to_toks(d)
    return json.dumps(dict(name=wid, lines=[dict(text=x, toks=parsed.get(i, [])) for i, x in enumerate(lines)]), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--merge")
    ap.add_argument("--json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.merge:
        n = 0
        with open(a.merge, encoding="utf-8") as f, open(a.json, "w", encoding="utf-8") as g:
            g.write("[")
            for x in f:
                if a.limit and n >= a.limit:
                    break
                g.write(("," if n else "") + x.strip())
                n += 1
            g.write("]")
        print("作品", n, "→", a.json)
        return
    eps = OrderedDict()
    with open(os.path.join(a.corpus, "index.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            eps.setdefault(r["work_id"], []).append(r["episode_id"])
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            for x in f:
                try:
                    done.add(json.loads(x)["name"])
                except Exception:
                    pass
    todo = [(a.corpus, w, e) for w, e in eps.items() if w not in done]
    print("作品", len(eps), "済み", len(done), "これから", len(todo), flush=True)
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.jobs, initializer=_init) as pool, open(a.out, "a", encoding="utf-8") as out:
        for k, s in enumerate(pool.imap_unordered(do_work, todo, chunksize=1), 1):
            out.write(s + "\n")
            if k % 20 == 0:
                out.flush()
                print(k, len(todo), f"{time.time() - t0:.0f}s", flush=True)
    print("おわり", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
