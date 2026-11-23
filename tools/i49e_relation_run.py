"""I49e: 関係の表（intervention/i49e_relation.Relation、変えない）を、数えの試し（tools/i49e_triple_count.py の七歩目）と同じ物で回す。数えだけ。

先生もループも使わない。CPU だけ。
1. 突き合わせ: 七歩目と同じ覚える側（青空の 3 つ組、会話の試しには ＋作った会話。全部 と 1/10）を learn で入れ（tick しない = 薄れなし）、
   同じ 3 つの試し（童話・会話 Gemma・会話 DeepSeek）を ask で当てる。候補は七歩目と同じ（ページの内容語のうちベクトルを持つ物、見せた 2 語を除く、＋答え）。
   1 位は 2 通りで数える: 答え（同じ点なら候補の並びで先の物）が隠した語か ／ 隠した語の点が、ほかのどの候補より大きいか（七歩目の数え方。同じ点は前に置く）。
2. 速さ: ask 1 回あたりの秒を、答えた段ごとに。
3. 聞きながら育つ形: 作った会話（覚える用 500 ＋ Gemma の試し用 100、この順）を 1 本ずつ読む。会話の中の 3 つ組ごとに、
   先に 3 席それぞれを ask（候補 = その会話の内容語、見せた 2 語を除く）→ 答え合わせ → learn。会話 1 本ごとに tick 1 回。初めは白紙。
   100 本ごとの 1 位の割合・答えた段の内訳・白紙の割合。比べ: 借りない形（2 語 → 述語側 → もう片方。Relation を継いで段だけ替える）、
   薄れあり（half 1,024）と薄れなし（half 1e12）。
4. 同じ流れで、青空の 3 つ組を先に learn してから読む場合。
記録はリポジトリの外。手で置く物: 100 本ごとの区切り、薄れなしの half 1e12、1/10 の取り方（七歩目と同じ）。
python tools/i49e_relation_run.py --src ... --meta ... --wave <波の板> --conllu <青空の conllu> --learn_talks <会話.jsonl>,...
    --tests Gemma=<試し用.jsonl>,DeepSeek=<試し用.jsonl> --stream_test <Gemma の試し用.jsonl> --out <stats.json> [--parts 1,2,3,4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

NO_BORROW = ("2語", "述語側", "もう片方")


def no_borrow_class():
    import i49e_relation as RL

    class NoBorrow(RL.Relation):
        """比べ: 借りない形（今の表と同じ下がり方）。段の並びだけを替える（骨組みの ask と同じ決め方）。"""

        def ask(self, si, tri, cands):
            cands = list(cands)
            near = RL.NEAR[si]
            other = next(j for j in range(3) if j not in (si, near))
            out = dict(段="答えられない", 点={}, 答え=None)
            for lv in NO_BORROW:
                s = self._level(lv, si, tri, cands)
                if any(v > 0 for v in s.values()):
                    best = max(cands, key=lambda c: (s[c], -cands.index(c)))
                    out = dict(段=lv, 点=s, 答え=best)
                    break
            self.rec.append((self.now, si, (tri[near], tri[other]), out["段"], out["答え"]))
            return out

    return NoBorrow


def strict_first(score, hid):
    """隠した語の点が、ほかのどの候補より大きいか（同じ点は前に置く数え方の 1 位）。"""
    if not score or score.get(hid, 0) <= 0:
        return False
    h = score[hid]
    return all(v < h for c, v in score.items() if c != hid)


def ask_items(rel, items, vec, timing=None):
    """試しの 3 つ組を、3 席それぞれで ask。返す: 席 → dict(数, 一位_答え, 一位_同点は前, 段, 段ごとの一位)。"""
    out = {}
    import i49e_relation as RL
    for si, slot in enumerate(RL.SLOTS):
        shown = [j for j in range(3) if j != si]
        n, h_ans, h_strict, lv, lv_hit = 0, 0, 0, Counter(), Counter()
        for it in items:
            tri, hid = it["組"], it["組"][si]
            show = {tri[j] for j in shown}
            cands = sorted({w for w in it["ページ"] if w in vec and w not in show} | {hid}, key=str)
            if len(cands) < 2:
                continue
            q = tuple(None if j == si else tri[j] for j in range(3))
            t1 = time.time()
            o = rel.ask(si, q, cands)
            if timing is not None:
                timing.setdefault(o["段"], []).append(time.time() - t1)
            n += 1
            ok = o["答え"] == hid
            h_ans += ok
            h_strict += strict_first(o["点"], hid)
            lv[o["段"]] += 1
            lv_hit[o["段"]] += ok
        out[slot] = dict(数=n, 一位_答え=round(h_ans / max(1, n), 4), 一位_同点は前=round(h_strict / max(1, n), 4),
                         段=dict(lv), 段ごとの一位={k: round(lv_hit[k] / v, 4) for k, v in lv.items()})
    return out


def talk_words(nlp, rows):
    """会話ごとの 3 つ組の並びと内容語の集まり。"""
    import i48_wave_board as W48
    import i49e_triple_count as TC
    out = []
    for r in rows:
        docs = list(nlp.pipe(r["発言"], batch_size=64))
        words = {(t.lemma_, t.pos_) for d in docs for t in d if t.pos_ in W48.CONTENT and t.lemma_ not in W48.STOP}
        tris = [ws for d in docs for kind, _, _, ws in TC.triples2(TC.toks_of_doc(d)) if kind == "三つ組"]
        out.append((tris, words))
    return out


def stream(rel, talks, every=100):
    """会話を 1 本ずつ読む: 3 つ組ごとに 3 席を ask → 答え合わせ → learn。1 本ごとに tick。every 本ごとに数える。"""
    import i49e_relation as RL
    rows, cur = [], dict(問い=0, 当たり=0, 段=Counter())
    for ti, (tris, words) in enumerate(talks):
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                o = rel.ask(si, q, cands)
                cur["問い"] += 1
                cur["当たり"] += o["答え"] == tri[si]
                cur["段"][o["段"]] += 1
            rel.learn(tri)
        rel.tick()
        if (ti + 1) % every == 0 or ti + 1 == len(talks):
            n = max(1, cur["問い"])
            rows.append({"会話": ti + 1, "問い": cur["問い"], "一位": round(cur["当たり"] / n, 4),
                         "白紙": round(cur["段"]["答えられない"] / n, 4),
                         "段": {k: round(v / n, 4) for k, v in cur["段"].items()}})
            cur = dict(問い=0, 当たり=0, 段=Counter())
    return rows


def main():
    import spacy
    import i49c_read_words as Rw
    import i49e_relation as RL
    import i49e_triple_count as TC
    ap = argparse.ArgumentParser()
    for k in ("src", "meta", "wave", "conllu", "learn_talks", "tests", "stream_test", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--books", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--parts", default="1,2,3,4")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    vec_of = lambda w: wv.of(*w)
    dou, learn, _ = TC.prepare(a, nlp, wv, t0)
    rows = [json.loads(l) for f in a.learn_talks.split(",") for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["語"]["選び方"] in ("い", "覚える側")]
    talks = talk_words(nlp, rows)
    tests = {"童": dou}
    for spec in a.tests.split(","):
        name, path = spec.split("=", 1)
        tests[name], _ = TC.talk_items(nlp, [json.loads(l) for l in open(path, encoding="utf-8")], wv)
    vec = {}
    for its in tests.values():
        vec.update(TC.vec_of_items(its, wv))
    ids = sorted(learn)
    res = dict(試し={k: len(v) for k, v in tests.items()})
    print("用意", round(time.time() - t0), flush=True)
    if parts & {"1", "2"}:
        for size, sel, n_talk in (("1/10", ids[::10], 50), ("全部", ids, len(talks))):
            aoz = [tri for w in sel for tri in learn[w]]
            talk = [tri for tris, _ in talks[:n_talk] for tri in tris]
            for lname, tris, tnames in (("青空", aoz, ["童"]), ("青空＋会話", aoz + talk, [k for k in tests if k != "童"])):
                rel = RL.Relation(vec_of)
                for tri in tris:
                    rel.learn(tri)
                timing = {}
                for tn in tnames:
                    res[f"1・{size}・覚える側 {lname}・試し {tn}"] = ask_items(rel, tests[tn], vec, timing if size == "全部" else None)
                if size == "全部":
                    res[f"2・速さ・覚える側 {lname}"] = {k: dict(回=len(v), 秒の中央値=round(float(np.median(v)), 4), 秒の平均=round(float(np.mean(v)), 4))
                                                    for k, v in timing.items()}
                print("1・2", size, lname, round(time.time() - t0), flush=True)
                json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if parts & {"3", "4"}:
        stream_talks = talks + talk_words(nlp, [json.loads(l) for l in open(a.stream_test, encoding="utf-8")])
        NoBorrow = no_borrow_class()
        runs = []
        if "3" in parts:
            runs += [("借りる・薄れあり", RL.Relation, 1024.0, False), ("借りる・薄れなし", RL.Relation, 1e12, False),
                     ("借りない・薄れあり", NoBorrow, 1024.0, False), ("借りない・薄れなし", NoBorrow, 1e12, False)]
        if "4" in parts:
            runs += [("青空を先に・借りる・薄れあり", RL.Relation, 1024.0, True)]
        all_aoz = [tri for w in ids for tri in learn[w]]
        for name, cls, half, pre in runs:
            rel = cls(vec_of, half=half)
            if pre:
                for tri in all_aoz:
                    rel.learn(tri)
            t1 = time.time()
            res[f"3・聞きながら育つ・{name}"] = dict(流れ=stream(rel, stream_talks), 秒=round(time.time() - t1))
            print("3・4", name, round(time.time() - t0), flush=True)
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
