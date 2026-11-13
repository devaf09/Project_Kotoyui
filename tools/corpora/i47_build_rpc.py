"""I47 用: RealPersonaChat（data/realpersonachat、CC BY-SA 4.0）を i47 の --cache の形にする。
作品 = 1 対話、行 = 1 発話（text は「」で囲む）、toks は GiNZA で「」ごと読む。

  python tools/corpora/i47_build_rpc.py --src data/realpersonachat/real_persona_chat/dialogues --out <非公開の置き場>/i47/rpc.json --jobs 16
"""
import argparse
import glob
import json
import os
import sys
import time

_nlp = None


def _init():
    global _nlp
    import spacy
    _nlp = spacy.load("ja_ginza")


def do(path):
    d = json.load(open(path, encoding="utf-8"))
    texts = ["「" + u["text"].strip().replace("\n", " ") + "」" for u in d["utterances"] if u["text"].strip()]
    lines = [dict(text=t, toks=[dict(i=x.i, text=x.text, lemma=x.lemma_, pos=x.pos_, dep=x.dep_, head=x.head.i) for x in doc])
             for t, doc in zip(texts, _nlp.pipe([t[:400] for t in texts], batch_size=64))]
    return json.dumps(dict(name=f"rpc_{d['dialogue_id']:05d}", lines=lines), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "*.json")))
    print("対話", len(files), flush=True)
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.jobs, initializer=_init) as pool, open(a.out, "w", encoding="utf-8") as g:
        g.write("[")
        for k, s in enumerate(pool.imap(do, files, chunksize=8)):
            g.write(("," if k else "") + s)
            if (k + 1) % 1000 == 0:
                print(k + 1, len(files), f"{time.time() - t0:.0f}s", flush=True)
        g.write("]")
    print("おわり", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
