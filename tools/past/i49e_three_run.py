"""I49e 三歩目: ことゆい（I49d の思考のループ）を、先生 2 人（子ども役）と読む場の 3 人目として座らせる試し（持ち主の案、未確定）。

席は intervention/i49e_three.Seat（変えない）。1 ページの流れ: 本 → A1 → B1 → こ（speak）→ A2。
  A2 を言う先生 A には、B1 と、ことゆいの行（語を空白なしでつないだ文字）の両方を見せる。ことゆいが話さなかったページは、今までと同じ A2。
  先生 2 人の役の文は tools/i49e_read_together.ROLE に KOTO を足した物。
  促し: ことゆいが声に出した 10 回目・20 回目…の時だけ（決め打ち、乱数なし）、A2 の問いに NUDGE を足し、記録に印を付ける。
初めに、今ある読む記録（--warm、5 作品 100 ページ）を、拾われやすさ（LivePickup、prior = 会話の表）に番として聞かせる（ループには読ませない）。
その後、同じ選び方の --books の作品で、次の 3 通りを、どれも聞かせた後の同じ拾われやすさから始める（ループは毎回新しく作る）:
  ア  重さ = 今の拾われやすさ（i49e_three.live_weight）、k=3     イ  重さ 0（聞いた順の初めの 3 個）、k=3
  ウ  聞くだけ（speak(aloud=False)。行は作るが、先生には見せず、拾われやすさにも入れない。重さはアと同じ）
作品が替わる所では、拾われやすさの直前の番を切る。記録（jsonl）はリポジトリの外。
手で置く物: 1 ページ 10 文・20 ページ（読む記録と同じ）、温度 0.7・seed 7、k=3、促しの間 10、KOTO と NUDGE の文、
  お手本の形の見分け（MODEL_RE）、器（Vessel(2048, bands=20, longest=1024)）。
python tools/past/i49e_three_run.py --src ... --meta ... --wave <波の板> --scorer <型.pkl> --pickup <拾われやすさの表.json>
    --warm <記録1.jsonl>,...（並びの 1 番目から順に） --books 6,7,8 --host http://127.0.0.1:<port> --model_a <A>
    --url_b <口の URL> --model_b <B> --key_env <鍵の環境変数名> --variants ア,イ,ウ --outdir <置き場>
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import copy
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

KOTO = ("もう 1 人、ことゆいという子がいます。まだ赤ちゃんくらいの話し方しかできません。言うのは 2〜3 語で、変な並びの事もあります。"
        "直したり点を付けたりはしません。分かった語を拾って、返してあげてください。")
NUDGE = "できそうなら、こういうふうに言うといいよ、と短いお手本を 1 つ見せてあげてね。ことゆいが言った語を使って、短く。無理ならしなくていいよ。"
EVERY = 10
K = 3
MODEL_RE = re.compile(r"(って|と)(言う|いう|言っ|いっ)(と|たら|てみ)|みたいに言")


def role():
    import i49e_read_together as RT
    return RT.ROLE + KOTO


def prompt(no, page, said, who, ask):
    hist = []
    for p, d in said:
        ko = f" ことゆい「{d['こ']}」" if d.get("こ") else ""
        hist.append(f"{p} ページ目: A「{d['A1']}」 B「{d['B1']}」{ko} A「{d['A2']}」")
    h = ("これまでに言ったこと:\n" + "\n".join(hist) + "\n\n") if hist else ""
    return [dict(role="system", content=role()),
            dict(role="user", content=f"{h}{no} ページ目:\n{page}\n\nあなたは {who} です。{ask}")]


class Felt:
    """組立ごとの行づくりの違和感（感じた物）を、席の番（聞く ／ 話す）で分けて持つ。"""

    def __init__(self):
        self.phase, self.rows = "聞く", []

    def after(self, th, rec):
        if rec["した事"] == "組立" and rec.get("感じた物"):
            self.rows.append((self.phase, dict(rec["感じた物"])))


def warm_live(nlp, warm_logs, src, meta, tb):
    import i49e_listen as L
    import i49e_listen_count as LC
    import i49e_read_together as RT
    live = L.LivePickup(prior=tb.value)
    for rk, log in enumerate(warm_logs, 1):
        _, S = RT.pick_book(src, meta, rank=rk)
        live.last = None
        for _, who, w in LC.turns_of(nlp, log, RT.pages_of(S)):
            live.hear(who, w)
    return live


def run_variant(a, var, books, live0, wv, scorer, nlp, out_path):
    import i49d_two_loops as TL
    import i49e_listen_count as LC
    import i49e_read_together as RT
    import i49e_three as T3
    import openai_chat as OC
    from i49a_inner_vessel import Vessel
    from i49c_think_loop import Words
    live = copy.deepcopy(live0)
    n0 = live.n
    pos_of = {}
    weight = (lambda n: 0.0) if var == "イ" else T3.live_weight(live, pos_of)
    th = TL.Thinker(Vessel(2048, bands=20, longest=1024), Words(), None, TL.Scored(scorer, pos_of, pick=TL.Pick(weight, K)))
    felt = th.attach(Felt())
    seat = T3.Seat(th, live, pos_of)
    aloud = var != "ウ"
    budget = OC.Budget(a.budget_b)
    meta = {}                                             # 拾われやすさの番の番号 → 話し手の組（直前 → 今）
    out = open(out_path, "w", encoding="utf-8")
    t0, spoke_n = time.time(), 0

    def words(text):
        return [(l, p, wv.of(l, p)) for l, p in LC.words_of(nlp, [text])[0]]

    def hear(kind, who, text, prev):
        meta[live.n] = (prev, kind)
        seat.listen(who, words(text))

    for bi, (info, P) in enumerate(books):
        live.last = None
        said = []
        prev = None
        for i in range(min(a.pages, len(P) - 1)):
            no, page, d, tm = i + 1, P[i], {}, {}
            hear("本", "本", page, prev)
            for key, who in (("A1", "A"), ("B1", "B")):
                ask = ("次に何が起きそうか、一言で言ってね。" if key == "A1" else
                       f"友だち（A）は「{d['A1']}」と言ったよ。きみは、次に何が起きると思う？ 同じでもちがってもいいよ。一言で。")
                t1 = time.time()
                msgs = prompt(no, page, said, who, ask)
                r = (RT.ask_ollama(a.host, a.model_a, msgs, a.temperature, a.seed, a.max_tokens) if who == "A" else
                     _api(a, msgs, budget, RT, OC))
                d[key] = RT.first_line(r["text"])
                tm[key] = dict(秒=round(time.time() - t1, 2), finish=r["finish"])
                hear(key, who, d[key], "本" if key == "A1" else "A1")
            felt.phase = "話す"
            meta[live.n] = ("B1", "こ")                    # 声に出した時だけ、この番号で拾われやすさの番になる
            n_mat = len(th.tray.vecs())
            line = seat.speak(aloud=aloud)
            felt.phase = "聞く"
            ko = "".join(line) if line else ""
            spoke = bool(line)
            nudge = False
            if aloud and spoke:
                spoke_n += 1
                nudge = spoke_n % EVERY == 0
                d["こ"] = ko
                ask = f"友だち（B）は「{d['B1']}」と言ったよ。ことゆいは「{ko}」と言ったよ。それを聞いて、一言。" + (NUDGE if nudge else "")
                prev_a2 = "こ"
            else:
                ask = f"友だち（B）は「{d['B1']}」と言ったよ。それを聞いて、一言。"
                prev_a2 = "B1"
            t1 = time.time()
            r = RT.ask_ollama(a.host, a.model_a, prompt(no, page, said, "A", ask), a.temperature, a.seed, a.max_tokens)
            d["A2"] = RT.first_line(r["text"])
            tm["A2"] = dict(秒=round(time.time() - t1, 2), finish=r["finish"])
            hear("A2", "A", d["A2"], prev_a2)
            prev = "A2"
            said.append((no, d))
            line_names = [w for w in (line or []) if pos_of.get(w)]
            out.write(json.dumps(dict(作品=info["並びの番号"], ページ=no, 発言={k: d.get(k, "") for k in ("A1", "B1", "A2")},
                                      こ=dict(行=line, 内容語=line_names, 声に出した=bool(aloud and spoke), 入れ物の材料=n_mat, 促し=nudge),
                                      呼び=tm), ensure_ascii=False) + "\n")
            out.flush()
        print(var, info["並びの番号"], round(time.time() - t0), flush=True)
    sec = round(time.time() - t0)
    tail = {"秒": sec, "B の呼んだ回数": budget.used, "歩": int(th.n), "内訳": th.counts(),
            "見張り": dict(張り付き=th.v.watch["張り付き"].tolist(), 死んだ刻み=int(th.v.watch["死んだ"]))}
    out.write(json.dumps(tail, ensure_ascii=False) + "\n")
    out.close()
    rec = [x for x in live.rec if x[0] >= n0]
    return dict(rec=rec, meta=meta, felt=felt.rows, tail=tail)


def _api(a, msgs, budget, RT, OC):
    try:
        return RT.ask_api(a.url_b, a.model_b, msgs, a.temperature, a.seed, a.max_tokens, a.key_env, budget)
    except OC.BudgetOver:
        return dict(text="", finish="budget")


# ---- 数える ----
def mean(xs):
    import numpy as np
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def rate(ws, target):
    ws = set(ws)
    return len(ws & target) / len(ws) if ws else None


def count_variant(nlp, path, res, books_pages):
    import numpy as np
    import i49e_listen as L
    import i49e_listen_count as LC
    import i49e_read_together as RT
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    pages = [r for r in rows if "ページ" in r]
    tail = rows[-1]
    T = {k: [r["発言"][k] for r in pages] for k in ("A1", "B1", "A2")}
    W = {k: [{l for l, _ in x} for x in LC.words_of(nlp, T[k])] for k in T}
    PW = [{l for l, _ in x} for x in LC.words_of(nlp, [books_pages[r["作品"]][r["ページ"] - 1] for r in pages])]
    ko = [r["こ"] for r in pages]
    spoke = [bool(x["行"]) for x in ko]
    lines = [set(x["内容語"]) for x in ko]
    o = dict(ページ=len(pages))
    o["話した割合"] = mean(spoke)
    o["声に出した割合"] = mean([x["声に出した"] for x in ko])
    o["行の内容語の数の中央値"] = float(np.median([len(x["内容語"]) for x in ko if x["行"]])) if any(spoke) else None
    o["行の語の数の中央値_詰め物込み"] = float(np.median([len(x["行"]) for x in ko if x["行"]])) if any(spoke) else None
    src = dict(本=0, A1=0, B1=0, 前のページまで=0)
    for i, ln in enumerate(lines):
        for w in ln:
            src["B1" if w in W["B1"][i] else "A1" if w in W["A1"][i] else "本" if w in PW[i] else "前のページまで"] += 1
    o["行の語の出どころ"] = src                                  # 同じページの中で、いちばん後の番（B1 → A1 → 本）
    idx = [i for i in range(len(pages)) if lines[i]]
    o["こ→A2_拾われた割合"] = mean([rate(lines[i], W["A2"][i]) for i in idx])
    o["こ→A2_A1とB1に無かった語だけ"] = mean([rate(lines[i] - W["A1"][i] - W["B1"][i], W["A2"][i]) for i in idx])
    o["B1→A2_拾われた割合"] = mean([rate(W["B1"][i], W["A2"][i]) for i in range(len(pages))])
    # 拾われやすさの言い当て（話し手の組ごと）
    meta = res["meta"]
    pairs = {}
    for n, p0, p1, y in res["rec"]:
        pr = meta.get(n)
        if pr is None:
            continue
        pairs.setdefault(f"{pr[0]}→{pr[1]}", []).append((p0, p1, y))
    o["言い当て"] = {k: dict(語の数=len(v), 拾われた割合=mean([x[2] for x in v]),
                           初めの見込みだけ=_r(L.auc([x[0] for x in v], [x[2] for x in v])),
                           聞いて育てた見込み=_r(L.auc([x[1] for x in v], [x[2] for x in v]))) for k, v in sorted(pairs.items())}
    # 促しとお手本
    nud = [i for i, x in enumerate(ko) if x["促し"]]
    shown = [i for i in nud if MODEL_RE.search(T["A2"][i])]
    nxt = lambda i: i + 1 < len(pages) and pages[i + 1]["作品"] == pages[i]["作品"]
    o["促し"] = dict(促しの番=len(nud), お手本の形が出た=len(shown),
                    お手本の語が次のページの行に出た割合=mean([rate(W["A2"][i], lines[i + 1]) for i in nud if nxt(i)]),
                    促しの無い番のA2の語が次のページの行に出た割合=mean([rate(W["A2"][i], lines[i + 1])
                                                         for i, x in enumerate(ko) if x["声に出した"] and not x["促し"] and nxt(i)]))
    o["先生の決まり"] = {k: mean([RT.one_sent(x) and len(x) <= RT.LIMIT and RT.ja(x) for x in T[k] if x]) for k in T}
    o["答えでない物"] = {k: sum(1 for x in T[k] if not x or not RT.ja(x)) for k in T}
    fr = {}
    for ph in ("聞く", "話す"):
        rs = [f for p, f in res["felt"] if p == ph]
        fr[ph] = {k: (dict(数=len(v), 中央値=round(float(np.median(v)), 4)) if v else None)
                  for k in ("迷い", "自信のなさ_選び", "組の珍しさ") for v in [[f[k] for f in rs if k in f]]}
    o["行づくりの違和感の割合"] = fr
    o["器と時間"] = {k: tail[k] for k in ("秒", "歩", "内訳", "見張り", "B の呼んだ回数")}
    return o, idx, lines, W


def _r(v):
    return None if v is None else round(v, 4)


def main():
    import spacy
    import i49c_read_words as R
    import i49d_odd as OD
    import i49d_pickup as PU
    import i49e_read_together as RT
    ap = argparse.ArgumentParser()
    for k in ("src", "meta", "wave", "scorer", "pickup", "warm", "books", "host", "model_a", "url_b", "model_b", "key_env", "outdir"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--variants", default="ア,イ,ウ")
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--budget_b", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_tokens", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    tb = PU.PickupTable.load(a.pickup)
    live0 = warm_live(nlp, a.warm.split(","), a.src, a.meta, tb)
    wv = R.WordVecs.load(a.wave)
    scorer = OD.OddScorer.load(a.scorer)
    books, bp = [], {}
    for rk in map(int, a.books.split(",")):
        info, S = RT.pick_book(a.src, a.meta, rank=rk)
        books.append((info, RT.pages_of(S)))
        bp[rk] = RT.pages_of(S)
    print("用意", round(time.time() - t0), "聞かせた番", live0.n, flush=True)
    stats = dict(用意=dict(聞かせた番=live0.n, 秒=round(time.time() - t0)), 作品=[i for i, _ in books], 組={})
    keep = {}
    for var in a.variants.split(","):
        path = os.path.join(a.outdir, f"three_{var}.jsonl")
        res = run_variant(a, var, books, live0, wv, scorer, nlp, path)
        o, idx, lines, W = count_variant(nlp, path, res, bp)
        stats["組"][var] = o
        keep[var] = (idx, lines, W)
        json.dump(stats, open(os.path.join(a.outdir, "three_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if "ウ" in keep:                                      # 土台: 見せていない「言ったはずの行」の語が A2 に出た割合
        idx, lines, W = keep["ウ"]
        stats["組"]["ウ"]["言ったはずの行の語がA2に出た割合"] = mean([rate(lines[i], W["A2"][i]) for i in idx])
    stats["秒"] = round(time.time() - t0)
    json.dump(stats, open(os.path.join(a.outdir, "three_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
