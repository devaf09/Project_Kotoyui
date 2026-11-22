"""I49e: 先生 2 人（子ども役）が、静かなお話を 1 ページずつ、次の展開を予想しながら読む記録を作り、数える（持ち主の案「3 人で読む」の一歩目、未確定）。

ループ（I49d）はまだ座らせない。ここは先生 2 人だけ。本文はリポジトリにも報告にも入れない（記録 jsonl もリポジトリの外）。
本の選び方（pick_book）: 青空文庫の目録で、分類 NDC K913（日本の子ども向けの物語）・新字新仮名・役割が著者・原題も副題も無い作品のうち、
  広く知られた作品の多い作者と、事件や冒険の筋の作者（EXCLUDE）を除き、文の数（tools/i49b_read_long.sentences_with_marks）が 200〜600 の物を、
  作品 ID の crc32 の小さい順に並べた --book 番目（1 から、既定 1）。
1 ページ = PAGE_LEN 文。1 ページの流れ: 2 人ともそのページの本文と、それまでのページの 2 人の発言を見る（先のページは見せない）。
  A1「次に何が起きそうか」を一言 → B1（A1 を見て、自分の予想を一言。同じでも違ってもよい）→ A2（B1 を受けて一言）。
先生: A = ollama（--host、think off）、B = OpenAI 互換の口（さくらの AI Engine。鍵は --key_env の環境変数からだけ読む。enable_thinking false）。
  温度・seed は手で置く数（両方に同じ物を渡す）。答えの最初の 1 行だけを使う。答えでない物（空・日本語でない・呼べない）は数に残し、代わりを作らない。
数える（count）: 内容語（I48 の CONTENT から STOP を除いた原形、GiNZA）で、
  1 発言の長さ（字・内容語の数）と決まり（1 文・30 字以内・日本語）を守った割合
  2 拾う割合（B1 の内容語のうち A1 に出た語、A2 の内容語のうち B1 に出た語）と、1 ページずらした組（前のページの相手の発言）での同じ割合
  3 予想の確かめ（A1・B1 の内容語のうち次のページに出た語）、比べ: (ア) 2〜5 ページ先の 1 ページに出た割合の平均、
    (イ) 今のページに出ていた語を除いた時の、次のページと 2〜5 ページ先の割合
  4 2 人の割れ（A1 と B1 の内容語の重なり: Jaccard、重なり 0 のページの割合）
  5 話を知っていそうか: 予想の発言のうち、まだ読んでいない 3 ページ以上先にだけ出る語（今までのページと次の 2 ページに無い語）を含む割合。
    その語が固有名詞（PROPN）の物と、本の中で 3 回以下しか出ない物に分けても数える
  6 かかった時間、呼んだ回数、答えでない物の数
  例: 指定したページの先生の発言だけ（本文と 8 字以上同じ並びの所は「（本文と同じ所）」に置き換える）
手で置く物: 1 ページ 10 文、20 ページ、温度 0.7、seed 7、答えの長さの上限（トークン）200、7 歳くらいの子どもの役の文（ROLE）、決まりの 30 字、
  2〜5 ページ先、3 ページ以上先、3 回以下、写しの境 8 字、本の選び方の EXCLUDE と文の数 200〜600。
python tools/i49e_read_together.py run --src <青空の本文> --meta <目録の csv> --host http://127.0.0.1:<port> --model_a <A のモデル>
    --url_b <口の URL .../v1> --model_b <B のモデル> --key_env <鍵の環境変数名> --log <記録.jsonl> [--pages 20] [--budget_b 40]
python tools/i49e_read_together.py count --src ... --meta ... --log <記録.jsonl> --out <stats.json> [--examples 1,8,15]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

PAGE_LEN = 10
EXCLUDE = ("宮沢賢治", "新美南吉", "芥川竜之介", "有島武郎", "楠山正雄", "鈴木三重吉",          # 広く知られた作品・昔話の書き直しが多い
           "江戸川乱歩", "海野十三", "小酒井不木", "野村胡堂", "押川春浪", "大阪圭吉")          # 事件・冒険の筋
ROLE = ("あなたは 7 歳くらいの子どもです。友だちと 2 人で、お話を 1 ページずつ読んでいます。"
        "思ったことを、そのまま短く言います。1 文だけ、30 字くらいまで。点数をつけたり、まちがいを言ったりはしません。")
LIMIT = 30
FAR = (2, 3, 4, 5)
AHEAD = 3
RARE = 3
COPY = 8


# ---- 本 ----
def pick_book(src_dir, meta_csv, lo=200, hi=600, rank=1):
    import i49b_read_long as RL
    files = {}
    for f in os.listdir(src_dir):
        files.setdefault(f.split("_")[0], []).append(f)
    seen, cands = set(), []
    for r in csv.DictReader(open(meta_csv, encoding="utf-8-sig")):
        wid = str(int(r["作品ID"]))
        if (r["分類番号"].strip() != "NDC K913" or r["役割フラグ"] != "著者" or r["文字遣い種別"] != "新字新仮名"
                or wid in seen or wid not in files):
            continue
        seen.add(wid)
        if r["姓"] + r["名"] in EXCLUDE or r["原題"] or r["副題"]:
            continue
        B = RL.body_of(os.path.join(src_dir, sorted(files[wid])[0]))
        if not B:
            continue
        S = [s for s, _ in RL.sentences_with_marks(B)]
        if lo <= len(S) <= hi:
            cands.append((zlib.crc32(wid.encode()), wid, S))
    cands.sort()
    _, wid, S = cands[rank - 1]
    return dict(作品ID=wid, 並びの番号=rank, 候補の数=len(cands), 文の数=len(S)), S


def pages_of(S, n=PAGE_LEN):
    return ["".join(S[i:i + n]) for i in range(0, len(S), n)]


# ---- 先生 ----
def ja(text):
    n = sum(1 for ch in text if 0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF)
    return n > sum(1 for ch in text if ch.isascii() and ch.isalpha())


def first_line(text):
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text.splitlines()[0].strip().strip("「」") if text else ""


def ask_ollama(host, model, msgs, temperature, seed, max_tokens):
    body = dict(model=model, messages=msgs, stream=False, think=False,
                options=dict(temperature=temperature, seed=seed, num_predict=max_tokens))
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode("utf-8"))
        return dict(text=out["message"]["content"], finish=out.get("done_reason"))
    except Exception as e:                                # 呼べない時は数に残す
        return dict(text="", finish=f"error {type(e).__name__}")


def ask_api(url, model, msgs, temperature, seed, max_tokens, key_env, budget):
    import openai_chat as OC
    r = OC.chat(url, model, msgs, key_env, budget, temperature=temperature, max_tokens=max_tokens, json_mode=False,
                chat_template_kwargs={"enable_thinking": False}, extra=dict(seed=seed))
    return dict(text=r["text"], finish=r["finish"])


def prompt(page_no, page, said, who, ask):
    hist = []
    for p, d in said:
        hist.append(f"{p} ページ目: A「{d['A1']}」 B「{d['B1']}」 A「{d['A2']}」")
    h = ("これまでに 2 人が言ったこと:\n" + "\n".join(hist) + "\n\n") if hist else ""
    return [dict(role="system", content=ROLE),
            dict(role="user", content=f"{h}{page_no} ページ目:\n{page}\n\nあなたは {who} です。{ask}")]


def run(a):
    import openai_chat as OC
    info, S = pick_book(a.src, a.meta, rank=a.book)
    P = pages_of(S)
    budget = OC.Budget(a.budget_b)
    said, t0 = [], time.time()
    out = open(a.log, "w", encoding="utf-8")
    out.write(json.dumps(dict(本=info, 手で置く物=dict(ページの文=PAGE_LEN, ページ=a.pages, 温度=a.temperature, seed=a.seed,
                                                   答えの上限=a.max_tokens, A=a.model_a, B=a.model_b)), ensure_ascii=False) + "\n")
    for i in range(min(a.pages, len(P) - 1)):
        no, page, d, tm = i + 1, P[i], {}, {}
        steps = (("A1", "A", "次に何が起きそうか、一言で言ってね。"),
                 ("B1", "B", None),
                 ("A2", "A", None))
        for key, who, ask in steps:
            if key == "B1":
                ask = f"友だち（A）は「{d['A1']}」と言ったよ。きみは、次に何が起きると思う？ 同じでもちがってもいいよ。一言で。"
            elif key == "A2":
                ask = f"友だち（B）は「{d['B1']}」と言ったよ。それを聞いて、一言。"
            msgs = prompt(no, page, said, who, ask)
            t1 = time.time()
            if who == "A":
                r = ask_ollama(a.host, a.model_a, msgs, a.temperature, a.seed, a.max_tokens)
            else:
                try:
                    r = ask_api(a.url_b, a.model_b, msgs, a.temperature, a.seed, a.max_tokens, a.key_env, budget)
                except OC.BudgetOver:
                    r = dict(text="", finish="budget")
            d[key] = first_line(r["text"])
            tm[key] = dict(秒=round(time.time() - t1, 2), finish=r["finish"], 行の数=len((r["text"] or "").strip().splitlines()))
        said.append((no, d))
        out.write(json.dumps(dict(ページ=no, 発言=d, 呼び=tm), ensure_ascii=False) + "\n")
        out.flush()
        print(no, round(time.time() - t0), flush=True)
    out.write(json.dumps({"秒": round(time.time() - t0), "B の呼んだ回数": budget.used}, ensure_ascii=False) + "\n")
    out.close()


# ---- 数える ----
def content_of(nlp, texts):
    import i48_wave_board as W48
    return [[t.lemma_ for t in d if t.pos_ in W48.CONTENT and t.lemma_ not in W48.STOP] for d in nlp.pipe(texts, batch_size=64)], \
        [{t.lemma_ for t in d if t.pos_ == "PROPN"} for d in nlp.pipe(texts, batch_size=64)]


def one_sent(text):
    body = text.rstrip("。！？!?」』）) 　")
    return not re.search(r"[。！？!?]", body)


def mask_copy(text, book, n=COPY):
    out, i = "", 0
    while i < len(text):
        j = i + n
        if j <= len(text) and text[i:j] in book:
            while j < len(text) and text[i:j + 1] in book:
                j += 1
            out += "（本文と同じ所）"
            i = j
        else:
            out += text[i]
            i += 1
    return out


def rate(words, target):
    w = set(words)
    return (len(w & target) / len(w)) if w else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def count(a):
    import spacy
    rows = [json.loads(l) for l in open(a.log, encoding="utf-8")]
    head, pages, tail = rows[0], [r for r in rows if "ページ" in r], rows[-1]
    info, S = pick_book(a.src, a.meta, rank=a.book)
    P = pages_of(S)
    nlp = spacy.load("ja_ginza")
    pc, pp = content_of(nlp, P)
    pw = [set(x) for x in pc]
    freq = {}
    for x in pc:
        for w in x:
            freq[w] = freq.get(w, 0) + 1
    K = ("A1", "B1", "A2")
    texts = [r["発言"][k] for r in pages for k in K]
    cw, _ = content_of(nlp, texts)
    W = {(r["ページ"], k): set(cw[i * 3 + j]) for i, r in enumerate(pages) for j, k in enumerate(K)}
    T = {(r["ページ"], k): r["発言"][k] for r in pages for k in K}
    nos = [r["ページ"] for r in pages]
    book = "".join(S)
    res = dict(本=info, 手で置く物=head["手で置く物"], ページ=len(pages))
    # 1 長さと決まり
    ln = {}
    for k in K:
        xs = [T[(n, k)] for n in nos if T[(n, k)]]
        ln[k] = dict(答えの数=len(xs), 字の中央値=float(np.median([len(x) for x in xs])) if xs else None,
                     内容語の数の中央値=float(np.median([len(W[(n, k)]) for n in nos if T[(n, k)]])) if xs else None,
                     一文=mean([one_sent(x) for x in xs]), 三十字以内=mean([len(x) <= LIMIT for x in xs]), 日本語=mean([ja(x) for x in xs]),
                     三つとも=mean([one_sent(x) and len(x) <= LIMIT and ja(x) for x in xs]))
    res["1_長さと決まり"] = ln
    # 2 拾う割合（同じページ ／ 1 ページずらした組）
    res["2_拾う割合"] = {
        "B1がA1から": dict(同じページ=mean([rate(W[(n, "B1")], W[(n, "A1")]) for n in nos]),
                         ずらした組=mean([rate(W[(n, "B1")], W[(n - 1, "A1")]) for n in nos if (n - 1, "A1") in W])),
        "A2がB1から": dict(同じページ=mean([rate(W[(n, "A2")], W[(n, "B1")]) for n in nos]),
                         ずらした組=mean([rate(W[(n, "A2")], W[(n - 1, "B1")]) for n in nos if (n - 1, "B1") in W]))}
    # 3 予想の確かめ
    pr = {}
    for k in ("A1", "B1"):
        nx, far, nx_new, far_new = [], [], [], []
        for n in nos:
            w, cur = W[(n, k)], pw[n - 1]
            if n < len(pw):
                nx.append(rate(w, pw[n]))
                nx_new.append(rate(w - cur, pw[n]))
            for d in FAR:
                if n - 1 + d < len(pw):
                    far.append(rate(w, pw[n - 1 + d]))
                    far_new.append(rate(w - cur, pw[n - 1 + d]))
        pr[k] = dict(次のページ=mean(nx), 離れたページ=mean(far), 今のページの語を除いて_次のページ=mean(nx_new),
                     今のページの語を除いて_離れたページ=mean(far_new),
                     今のページの語の割合=mean([rate(W[(n, k)], pw[n - 1]) for n in nos]))
    res["3_予想の確かめ"] = pr
    # 4 2 人の割れ
    jac = [len(W[(n, "A1")] & W[(n, "B1")]) / len(W[(n, "A1")] | W[(n, "B1")]) for n in nos if W[(n, "A1")] | W[(n, "B1")]]
    res["4_二人の割れ"] = dict(Jaccard=mean(jac), 重なり0のページ=mean([j == 0 for j in jac]))
    # 5 話を知っていそうか
    kn = {}
    for k in ("A1", "B1"):
        hit, hit_p, hit_r = [], [], []
        for n in nos:
            seen = set().union(*pw[: min(len(pw), n + 2)])          # 今までのページと次の 2 ページ（n は 1 から）
            future = set().union(*pw[n + AHEAD - 1:]) if n + AHEAD - 1 < len(pw) else set()
            only = (W[(n, k)] - seen) & future
            hit.append(bool(only))
            props = set().union(*pp[n + AHEAD - 1:]) if n + AHEAD - 1 < len(pp) else set()
            hit_p.append(bool(only & props))
            hit_r.append(bool({w for w in only if freq.get(w, 0) <= RARE}))
        kn[k] = dict(先にだけ出る語を含む=mean(hit), そのうち固有名詞=mean(hit_p), そのうち本で3回以下=mean(hit_r))
    res["5_話を知っていそうか"] = kn
    # 6 時間と呼び
    ex = {}
    for who, ks in (("A", ("A1", "A2")), ("B", ("B1",))):
        ts = [r["呼び"][k]["秒"] for r in pages for k in ks]
        ex[who] = dict(呼んだ回数=len(ts), 秒の合計=round(sum(ts), 1), 秒の中央値=float(np.median(ts)),
                       答えでない物=sum(1 for r in pages for k in ks if not r["発言"][k] or not ja(r["発言"][k])),
                       終わり方={f: sum(1 for r in pages for k in ks if r["呼び"][k]["finish"] == f)
                               for f in sorted({str(r["呼び"][k]["finish"]) for r in pages for k in ks})},
                       二行以上の答え=sum(1 for r in pages for k in ks if r["呼び"][k]["行の数"] > 1))
    res["6_時間と呼び"] = dict(先生=ex, 全体の秒=tail.get("秒"), B_の口を呼んだ回数=tail.get("B の呼んだ回数"))
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.examples:
        exs = {n: {k: mask_copy(T[(n, k)], book) for k in K} for n in map(int, a.examples.split(",")) if (n, "A1") in T}
        json.dump(exs, open(a.out.replace(".json", "_examples.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    c = sp.add_parser("count")
    for p in (r, c):
        p.add_argument("--src", required=True)
        p.add_argument("--meta", required=True)
        p.add_argument("--log", required=True)
        p.add_argument("--book", type=int, default=1, help="本の選び方の並びの何番目か（1 から）")
    r.add_argument("--host", required=True)
    r.add_argument("--model_a", required=True)
    r.add_argument("--url_b", required=True)
    r.add_argument("--model_b", required=True)
    r.add_argument("--key_env", required=True)
    r.add_argument("--pages", type=int, default=20)
    r.add_argument("--budget_b", type=int, default=40)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--max_tokens", type=int, default=200)
    c.add_argument("--out", required=True)
    c.add_argument("--examples", default=None, help="例に出すページの番号（, 区切り）")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else count(a)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
