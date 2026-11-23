"""I49e: 聞き返しのあるやり取り（intervention/i49e_askback.Exchange、変えない）を、3 人で読む試しと同じ読み方で回す（持ち主の案、未確定）。

読み方は tools/i49e_three_run.py と同じ: 初めに今ある読む記録（--warm）を拾われやすさ（LivePickup、prior = 会話の表）に聞かせ、
作品（--three_log の記録にある作品、例 6〜8）を 1 ページずつ。1 ページ: 本 → A1 → B1（--three_log の記録の物を聞かせ直す。先生は呼び直さない）
→ ことゆいの番を Exchange(席, 先生, rounds).run() に替える。A2 は無い（やり取りの先生の返事が代わり）。
先生: OpenAI 互換の口（手元の ollama の gemma4:31b など、考えを止める reasoning_effort）。聞き返し役の比べ（tools/i49e_askback_compare.py）と同じ役の文・
  答えの形（分かった・意味の候補・返事、JSON）で、今のページの本文・B1・このやり取りのそれまでの番を見せる。答えの文は内容語 [(原形, 品詞, ベクトル)] に直して渡す。
  読めない答えは「分からない・返事も意味も無し」として渡し、数に残す。
比べ: いつもの選び方のまま言い直す形（Exchange を継いで、_say に intent を渡さない。骨組みは変えない）。
数える物: 終わり方ごとの数、1 番目の行が「分かった」の割合、言い直しの番の数と「分からない → 分かった」の数、
  言い直しで選んだ語の近さの中央値、言い直しの語のうち先生の「意味」の語から来た割合、1 やり取りの番の数の分布、1 ページの秒数、読めなかった答えの数。
記録（jsonl）と数えはリポジトリの外。
手で置く物: 番の数 rounds 4、k=3、温度 0.7、seed、答えの上限、器（Vessel(2048, bands=20, longest=1024)）、先生の役の文（聞き返しの比べと同じ）。
python tools/i49e_askback_run.py --src ... --meta ... --wave <波の板> --scorer <型.pkl> --pickup <拾われやすさの表.json> --warm <記録1.jsonl>,...
    --three_log <three_ア.jsonl> --url <口の URL .../v1> --key_env <鍵の環境変数名> --model <先生のモデル> --mode 近さ|いつも --out <置き場の頭>
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

K = 3


def make_plain_exchange():
    import i49e_askback as AB

    class PlainResay(AB.Exchange):
        """比べ: 言い直しでも、いつもの選び方のまま（言いたかった向きへの近さを使わない）。"""

        def _say(self, intent=None):
            return super()._say(None)

    return PlainResay


class Teacher:
    """外の先生を包む。今のページの本文・B1・このやり取りのそれまでの番を見せ、答えを語の形に直す。"""

    def __init__(self, a, nlp, wv, OC, budget):
        self.a, self.nlp, self.wv, self.OC, self.budget = a, nlp, wv, OC, budget
        self.page, self.b1, self.hist, self.log, self.bad = "", "", [], [], 0

    def start(self, page, b1):
        self.page, self.b1, self.hist, self.log = page, b1, [], []

    def words(self, text):
        import i49e_listen_count as LC
        return [(l, p, self.wv.of(l, p)) for l, p in LC.words_of(self.nlp, [text])[0]] if text else []

    def __call__(self, line):
        import i49e_askback_compare as AC
        ko = "".join(line)
        user = AC.ask_text(self.page, self.b1, ko)
        if self.hist:
            user = "このやり取りのそれまで:\n" + "\n".join(self.hist) + "\n\n" + user
        msgs = [dict(role="system", content=AC.ROLE), dict(role="user", content=user)]
        o = self.OC.chat(self.a.url, self.a.model, msgs, self.a.key_env, self.budget, temperature=self.a.temperature,
                         max_tokens=self.a.max_tokens, json_mode=False, reasoning_effort=self.a.reasoning_effort, extra=dict(seed=self.a.seed))
        p = AC.parse(o["text"])
        if p is None:
            self.bad += 1
            p = dict(分かった=False, 意味の候補="", 返事="")
        self.hist.append(f"ことゆい「{ko}」 先生「{p['返事']}」")
        rep, mean = self.words(p["返事"]), self.words(p["意味の候補"])
        self.log.append(dict(分かった=p["分かった"], 返事の語=[l for l, _, _ in rep], 意味の語=[l for l, _, _ in mean], 読めた=o["text"] != "" and AC.parse(o["text"]) is not None))
        return dict(分かった=p["分かった"], 返事=rep, 意味=mean)


def run(a):
    import spacy
    import i49c_read_words as R
    import i49d_odd as OD
    import i49d_pickup as PU
    import i49d_two_loops as TL
    import i49e_askback as AB
    import i49e_listen_count as LC
    import i49e_read_together as RT
    import i49e_three as T3
    import i49e_three_run as TR
    import openai_chat as OC
    from i49a_inner_vessel import Vessel
    from i49c_think_loop import Words
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    live = TR.warm_live(nlp, a.warm.split(","), a.src, a.meta, PU.PickupTable.load(a.pickup))
    wv = R.WordVecs.load(a.wave)
    scorer = OD.OddScorer.load(a.scorer)
    rows = [json.loads(l) for l in open(a.three_log, encoding="utf-8")]
    rows = [r for r in rows if "ページ" in r][: a.limit]
    pages = {rk: RT.pages_of(RT.pick_book(a.src, a.meta, rank=rk)[1]) for rk in sorted({r["作品"] for r in rows})}
    pos_of = {}
    th = TL.Thinker(Vessel(2048, bands=20, longest=1024), Words(), None, TL.Scored(scorer, pos_of, pick=TL.Pick(T3.live_weight(live, pos_of), K)))
    seat = T3.Seat(th, live, pos_of)
    budget = OC.Budget(a.limit * a.rounds * 2)
    teacher = Teacher(a, nlp, wv, OC, budget)
    Ex = AB.Exchange if a.mode == "近さ" else make_plain_exchange()
    words = lambda text: [(l, p, wv.of(l, p)) for l, p in LC.words_of(nlp, [text])[0]]
    out = open(a.out + ".jsonl", "w", encoding="utf-8")
    print("用意", round(time.time() - t0), flush=True)
    last_book = None
    for r in rows:
        if r["作品"] != last_book:
            live.last = None
            last_book = r["作品"]
        t1 = time.time()
        page = pages[r["作品"]][r["ページ"] - 1]
        seat.listen("本", words(page))
        seat.listen("A", words(r["発言"]["A1"]))
        seat.listen("B", words(r["発言"]["B1"]))
        teacher.start(page, r["発言"]["B1"])
        res = Ex(seat, teacher, who="A", rounds=a.rounds).run()
        out.write(json.dumps(dict(作品=r["作品"], ページ=r["ページ"], 秒=round(time.time() - t1, 2), 終わり=res["終わり"], 番=res["番"],
                                  言い直しで分かった=res["言い直しで分かった"], 先生=teacher.log), ensure_ascii=False, default=float) + "\n")
        out.flush()
    out.write(json.dumps({"秒": round(time.time() - t0), "呼んだ回数": budget.used, "読めなかった答え": teacher.bad, "歩": int(th.n),
                          "見張り": dict(張り付き=th.v.watch["張り付き"].tolist(), 死んだ刻み=int(th.v.watch["死んだ"]))}, ensure_ascii=False) + "\n")
    out.close()


def count(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ex, tail = [r for r in rows if "ページ" in r], rows[-1]
    o = dict(ページ=len(ex), 終わり方=dict(Counter(r["終わり"] for r in ex)))
    spoke = [r for r in ex if r["番"]]
    o["1番目の行が分かった"] = round(float(np.mean([r["番"][0]["分かった"] for r in spoke])), 4) if spoke else None
    resay = [(r, i) for r in ex for i, t in enumerate(r["番"]) if t["言い直し"]]
    o["言い直しの番"] = len(resay)
    o["分からない→分かった"] = sum(r["言い直しで分かった"] for r in ex)
    near = [v for r, i in resay for v in (r["番"][i]["近さ"] or {}).values() if v is not None]
    o["言い直しの語の近さの中央値"] = round(float(np.median(near)), 4) if near else None
    from_mean, n_words = 0, 0
    for r, i in resay:
        mean = set(r["先生"][i - 1]["意味の語"]) if i - 1 < len(r["先生"]) else set()
        ws = list((r["番"][i]["近さ"] or {}).keys())
        n_words += len(ws)
        from_mean += sum(1 for w in ws if w in mean)
    o["言い直しの語のうち先生の意味の語から"] = round(from_mean / n_words, 4) if n_words else None
    o["やり取りの番の数"] = dict(sorted(Counter(len(r["番"]) for r in ex).items()))
    o["1ページの秒の中央値"] = round(float(np.median([r["秒"] for r in ex])), 2)
    o["全体"] = tail
    return o


def main():
    ap = argparse.ArgumentParser()
    for k in ("src", "meta", "wave", "scorer", "pickup", "warm", "three_log", "url", "key_env", "model", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--mode", choices=("近さ", "いつも"), default="近さ")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--reasoning_effort", default="none")
    a = ap.parse_args()
    run(a)
    res = count(a.out + ".jsonl")
    json.dump(res, open(a.out + "_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
