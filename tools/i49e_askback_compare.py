"""I49e: 聞き返し役の先生を選ぶ小さな比べ（持ち主の案「先生が理解できない時は聞き返すか、ゆっくり言うよう促す」の下ごしらえ、未確定）。

ことゆいを座らせた試し（tools/i49e_three_run.py の記録 jsonl）の、ことゆいが声に出したページを使う。
先生役（やさしい大人）に、そのページの本文・B1・ことゆいの行を見せ、次の JSON だけで答えてもらう:
  {"分かった": true/false, "意味の候補": "ことゆいが言いたそうな事（分からなければ空でなく、いちばんありそうな物）", "返事": "ことゆいへの一言"}
分かった時の返事はふつうに返す一言、分からない時の返事は「〇〇のこと？」の聞き返しか「ゆっくり言ってみて」の促し。
モデルを替えて、同じページに同じ問いを出す（OpenAI 互換の口、さくらの AI Engine。鍵は環境変数からだけ）。
数える物（数だけ）: 答えの形が合った割合（JSON・欄がそろう）、分かったの割合、返事の字の中央値と 30 字以内の割合、
  返事と意味の候補が、ことゆいの行の内容語を使った割合（拾った）、意味の候補の内容語のうち、そのページの本文にある語の割合（話に沿っているか）、
  分からない時の返事が聞き返しの形（「？」で終わる）の割合、モデルどうしの「分かった」の一致、時間、呼んだ回数、空・切れた答えの数。
入れ替えの試し（--swap）: ことゆいの行だけを、別の作品のページでことゆいが言った行と入れ替え、先生が「分からない」と言えるかを見る。
  --orig_raw に入れ替えない時の raw を渡すと、同じページどうしの対（この行だけ・元の行だけ・両方・どちらも無しの「分からない」）を数える。
  入れ替えた行の語が、たまたまそのページの本文にある所は分けて数える。
手で置く物: 先生の役の文（ROLE）、温度 0.7、seed 7、答えの上限 2,048 トークン（考えを止められないモデルがあるため）、30 字、使うページ（声に出したページを先頭から --limit 個）。
python tools/i49e_askback_compare.py --src ... --meta ... --log <three_ア.jsonl> --url <口の URL> --key_env <鍵の環境変数名>
    --models <m1>,<m2>,... --raw <raw.jsonl> --out <stats.json> [--limit 60] [--sleep 秒 --plain --reasoning_effort ...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

ROLE = ("あなたは、やさしい大人の先生です。ことゆいという子と、お話を 1 ページずつ読んでいます。"
        "ことゆいは、まだ赤ちゃんくらいの話し方しかできません。言うのは 2〜3 語で、変な並びの事もあります。直したり点を付けたりはしません。"
        "ことゆいが言いたい事が分かったら、その語を使って短く返します。分からない時は「〇〇のこと？」と意味の候補を 1 つ挙げて聞き返すか、"
        "「ゆっくり言ってみて」と促します。返事は 1 文、30 字くらいまで。"
        '答えは次の JSON だけ: {"分かった": true か false, "意味の候補": "ことゆいが言いたそうな事", "返事": "ことゆいへの一言"}')
LIMIT = 30


def ask_text(page, b1, ko):
    return f"今のページ:\n{page}\n\n友だちは「{b1}」と言いました。\nことゆいは「{ko}」と言いました。"


def parse(text):
    """答えの中の、欄のそろった最後の JSON（考えを本文に書くモデルは、考えの後ろに答えを置く）。"""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", (text or "").strip()))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and all(k in d for k in ("分かった", "意味の候補", "返事")):
            return dict(分かった=bool(d["分かった"]), 意味の候補=str(d["意味の候補"] or ""), 返事=str(d["返事"] or ""))
    return None


def main():
    import spacy
    import i49e_listen_count as LC
    import i49e_read_together as RT
    import openai_chat as OC
    ap = argparse.ArgumentParser()
    for k in ("src", "meta", "log", "url", "key_env", "models", "raw", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--sleep", type=float, default=0.0, help="呼ぶ間の待ち（秒）。回数の上限が小さい口のため")
    ap.add_argument("--plain", action="store_true", help="さくら向けの欄（enable_thinking・seed）を渡さない")
    ap.add_argument("--reasoning_effort", default=None)
    ap.add_argument("--stop_on_limit", action="store_true", help="回数の上限（429）に当たったら、そのモデルはそこで止める（再試行しない）")
    ap.add_argument("--swap", type=int, default=None, help="ことゆいの行だけを、別の作品のページの行と入れ替える（入れ替え先を決める seed）")
    ap.add_argument("--orig_raw", default=None, help="入れ替えない時の raw（同じページどうしの対で比べる）")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.log, encoding="utf-8")]
    rows = [r for r in rows if "ページ" in r and r["こ"]["声に出した"]]
    if a.swap is not None:                                # 入れ替え先は全部の行で決めてから --limit で切る（切り方で入れ替え先が変わらないように）
        rng = random.Random(a.swap)
        src_of = [rng.choice([j for j in range(len(rows)) if rows[j]["作品"] != r["作品"]]) for r in rows]
        rows = [dict(r, こ=rows[j]["こ"], 入れ替え元=dict(作品=rows[j]["作品"], ページ=rows[j]["ページ"])) for r, j in zip(rows, src_of)]
    rows = rows[: a.limit]
    pages = {}
    for rk in sorted({r["作品"] for r in rows}):
        _, S = RT.pick_book(a.src, a.meta, rank=rk)
        pages[rk] = RT.pages_of(S)
    nlp = spacy.load("ja_ginza")
    lem = lambda texts: [{l for l, _ in x} for x in LC.words_of(nlp, texts)]
    page_w = lem([pages[r["作品"]][r["ページ"] - 1] for r in rows])
    ko_w = [set(r["こ"]["内容語"]) for r in rows]
    chance = [bool(ko_w[i] & page_w[i]) for i in range(len(rows))]   # ことゆいの行の語が、たまたまそのページの本文にある
    orig = {}
    if a.orig_raw:
        for l in open(a.orig_raw, encoding="utf-8"):
            d = json.loads(l)
            orig[(d["モデル"], d["作品"], d["ページ"])] = parse(d["答え"])
    raw = open(a.raw, "w", encoding="utf-8")
    res = dict(ページ=len(rows), モデル={})
    got = {}
    for model in a.models.split(","):
        budget = OC.Budget(len(rows) * 5)
        ans, secs, fin = [], [], []
        for r in rows:
            page = pages[r["作品"]][r["ページ"] - 1]
            ko = "".join(r["こ"]["行"])
            msgs = [dict(role="system", content=ROLE), dict(role="user", content=ask_text(page, r["発言"]["B1"], ko))]
            t1 = time.time()
            o = OC.chat(a.url, model, msgs, a.key_env, budget, temperature=a.temperature, max_tokens=a.max_tokens, json_mode=False,
                        chat_template_kwargs=None if a.plain else {"enable_thinking": False},
                        extra=None if a.plain else dict(seed=a.seed), reasoning_effort=a.reasoning_effort,
                        retries=0 if a.stop_on_limit else 4, wait=1 if a.stop_on_limit else 60)
            if a.sleep:
                time.sleep(a.sleep)
            secs.append(time.time() - t1)
            fin.append(str(o["finish"]))
            p = parse(o["text"])
            ans.append(p)
            raw.write(json.dumps(dict(モデル=model, 作品=r["作品"], ページ=r["ページ"], 答え=o["text"], finish=o["finish"],
                                      入れ替え元=r.get("入れ替え元")), ensure_ascii=False) + "\n")
            raw.flush()
            if a.stop_on_limit and "429" in str(o["finish"]):
                print(model, "上限に当たって止めた", len(ans), "ページ目", flush=True)
                break
        ans += [None] * (len(rows) - len(ans))             # 止めた後のページは答え無し
        ok = [i for i, p in enumerate(ans) if p]
        rep_w = lem([ans[i]["返事"] for i in ok])
        mean_w = lem([ans[i]["意味の候補"] for i in ok])
        und = [i for i in ok if ans[i]["分かった"]]
        notu = [i for i in ok if not ans[i]["分かった"]]
        m = lambda xs: round(float(np.mean(xs)), 4) if len(xs) else None
        res["モデル"][model] = dict(
            形が合った割合=m([p is not None for p in ans[:len(fin)]]), 分かったの割合=m([ans[i]["分かった"] for i in ok]),
            返事の字の中央値=float(np.median([len(ans[i]["返事"]) for i in ok])) if ok else None,
            返事が30字以内=m([len(ans[i]["返事"]) <= LIMIT for i in ok]),
            返事がことゆいの語を使った=m([bool(rep_w[j] & ko_w[i]) for j, i in enumerate(ok)]),
            意味の候補がことゆいの語を使った=m([bool(mean_w[j] & ko_w[i]) for j, i in enumerate(ok)]),
            意味の候補の語のうち本文にある割合=m([len(mean_w[j] & page_w[i]) / len(mean_w[j]) for j, i in enumerate(ok) if mean_w[j]]),
            分からない時に聞き返しの形=m([ans[i]["返事"].rstrip().endswith(("？", "?")) for i in notu]),
            秒の中央値=round(float(np.median(secs)), 2), 呼んだ回数=budget.used, 答えたページ=len(fin),
            終わり方={f: fin.count(f) for f in sorted(set(fin))})
        if a.swap is not None:
            nu = lambda i: not ans[i]["分かった"]
            res["モデル"][model]["本文にたまたまある行"] = dict(
                ページ=sum(chance[i] for i in ok), 分からないの割合=m([nu(i) for i in ok if chance[i]]))
            res["モデル"][model]["本文に無い行"] = dict(
                ページ=sum(not chance[i] for i in ok), 分からないの割合=m([nu(i) for i in ok if not chance[i]]))
        if orig:
            pr = [(orig.get((model, rows[i]["作品"], rows[i]["ページ"])), ans[i]) for i in ok]
            pr = [(x, y) for x, y in pr if x]
            res["モデル"][model]["元の行との対"] = dict(
                対の数=len(pr), 元の行の分からない=m([not x["分かった"] for x, _ in pr]), この行の分からない=m([not y["分かった"] for _, y in pr]),
                この行だけ分からない=sum(1 for x, y in pr if x["分かった"] and not y["分かった"]),
                元の行だけ分からない=sum(1 for x, y in pr if not x["分かった"] and y["分かった"]),
                両方分からない=sum(1 for x, y in pr if not x["分かった"] and not y["分かった"]),
                両方分かった=sum(1 for x, y in pr if x["分かった"] and y["分かった"]))
        got[model] = ans
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(model, json.dumps(res["モデル"][model], ensure_ascii=False), flush=True)
    ms = list(got)
    agree = {}
    for i, x in enumerate(ms):
        for y in ms[i + 1:]:
            both = [k for k in range(len(rows)) if got[x][k] and got[y][k]]
            agree[f"{x} と {y}"] = round(float(np.mean([got[x][k]["分かった"] == got[y][k]["分かった"] for k in both])), 4) if both else None
    res["分かったの一致"] = agree
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    raw.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
