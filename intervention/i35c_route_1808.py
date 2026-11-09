"""介入の研究 実験I35c: 指摘の板（I35a）を 1,808 作品の台詞で数え直す（PC で回す）

台詞の行（i34_dialogue_board.isq、「『 で始まる行。閉じない行も）を GiNZA で解析し、続く台詞 A → B → C の
手がかりの型 → 回し先 を数える（i35_route_grain.count_raw）。否定の始めの語は I34c の会話の板（--dialogue）から。
語の珍しさ（作品をまたぐ）は 1,808 作品では数えない。本文は出さない（数だけ）。
CPU だけで回す（CUDA_VISIBLE_DEVICES=）。Web 小説は _wn。結果は非公開の置き場だけ。
実行: python intervention/i35c_route_1808.py --corpus <webNW_txt_v1> --dialogue <dialogue_board_1808_wn.json> --out <i35c_counts_wn.jsonl> --jobs 10
      python intervention/i35c_route_1808.py --merge <i35c_counts_wn.jsonl> --dialogue <...> --board <route_board_1808_wn.json> --stats <i35c_wn.jsonl>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_nlp = None
_negx = None
MIN_N = 5          # 否定の始めの語は、肯定・否定の答えの始めに 5 回以上出た語だけ


def neg_from_board(path):
    b = json.load(open(path, encoding="utf-8"))
    c = defaultdict(Counter)
    for k, v in b["start"].items():
        qt, kind, x = k.split("|", 2)
        if qt != "問いでない" and kind in ("肯定", "否定"):
            c[x][kind] += v
    return {x for x, v in c.items() if v["否定"] + v["肯定"] >= MIN_N and (v["否定"] + 0.5) / (v["否定"] + v["肯定"] + 1) > 0.5}


def _init(dialogue):
    global _nlp, _negx
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import spacy
    _nlp = spacy.load("ja_ginza")
    _negx = neg_from_board(dialogue)


def do_work(args):
    import i27t_first_person as T
    import i27v_extract_quotes as V
    import i35_route_grain as R
    corpus, wid, eps = args
    lines = V.work_lines(corpus, wid, eps)
    import i34_dialogue_board as D
    need = {li for li, x in enumerate(lines) if D.isq(x)}             # 数え上げ（isq）と同じ行を解析する（『 で始まる行・閉じない行も）
    idx = sorted(need)
    docs = _nlp.pipe([lines[i] for i in idx], batch_size=64)
    parsed = {i: V.to_toks(d) for i, d in zip(idx, docs)}
    w = dict(lines=[dict(text=x, toks=parsed.get(i, [])) for i, x in enumerate(lines)])
    tab, _, n = R.count_raw([w], _negx, None, 0)
    return wid, dict(work_id=wid, tab={k: dict(v) for k, v in tab.items()}, n=dict(n))


def merge(a):
    import i35_route_grain as R
    tab, n, k = defaultdict(Counter), Counter(), 0
    with open(a.merge, encoding="utf-8") as f:
        for x in f:
            r = json.loads(x)
            k += 1
            for c, v in r["tab"].items():
                tab[c].update(v)
            n.update(r["n"])
    negx = neg_from_board(a.dialogue)
    rec = dict(exp="I35c", 作品=k, **R.summarize(tab, {}, n, negx))
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(a.board, "w", encoding="utf-8") as f:
        json.dump(R.route_board(tab), f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--dialogue", required=True)
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
    with Pool(a.jobs, initializer=_init, initargs=(a.dialogue,)) as pool, open(a.out, "a", encoding="utf-8") as out:
        for k, (wid, rec) in enumerate(pool.imap_unordered(do_work, todo, chunksize=1), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if k % 20 == 0:
                print(k, len(todo), f"{time.time() - t0:.0f}s", flush=True)
    print("おわり", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
