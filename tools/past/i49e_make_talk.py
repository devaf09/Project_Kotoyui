"""I49e: 語を 1 つ渡して、子どもと大人の短い日常会話を作ってもらう（持ち主の案「Gemini に要点を混ぜて日常会話を作ってもらう」、未確定）。

渡すのは語 1 つだけ（関係は作り手に出してもらう。3 つ組をこちらで決めて混ぜない）。1 回の呼び出しで 1 語ぶん。
作り手は OpenAI 互換の口（さくらの AI Engine・Gemini など）。鍵は環境変数からだけ読む。送るのは語と頼み方だけ。
入力: 語の jsonl（1 行 = {"原形": ..., "品詞": ..., "選び方": ...}）。出力: 作った会話の jsonl（語・選び方・本文・終わり方・秒）。リポジトリの外に置く。
手で置く物: 頼み方の文（ASK）、10 往復、温度 0.7、答えの上限 2,048 トークン。
python tools/past/i49e_make_talk.py --words <語.jsonl> --url <口の URL .../v1> --key_env <鍵の環境変数名> --model <モデル> --out <会話.jsonl>
    [--limit N] [--sleep 秒 --plain --stop_on_limit]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))

ASK = ("「{w}」という語を使った、子どもと大人の短い日常会話を 10 往復作ってください。"
       "1 文は短くしてください。だれが・何を・どうしたを省かずに言ってください。"
       "1 行に 1 つの発言で、行の初めに「子:」か「大人:」を付けてください。会話のほかは書かないでください。")


def lines_of(text):
    """作った会話の本文 → 発言の文の並び（「子:」「大人:」を外す）。"""
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip().lstrip("-*・ ").strip()
        for head in ("子:", "子：", "大人:", "大人："):
            if ln.startswith(head):
                ln = ln[len(head):].strip()
                break
        else:
            continue
        if ln:
            out.append(ln)
    return out


def main():
    import openai_chat as OC
    ap = argparse.ArgumentParser()
    for k in ("words", "url", "key_env", "model", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--plain", action="store_true", help="さくら向けの欄（enable_thinking・seed）を渡さない")
    ap.add_argument("--stop_on_limit", action="store_true")
    a = ap.parse_args()
    words = [json.loads(l) for l in open(a.words, encoding="utf-8")][: a.limit]
    budget = OC.Budget(len(words) * 5)
    out = open(a.out, "w", encoding="utf-8")
    for w in words:
        t1 = time.time()
        o = OC.chat(a.url, a.model, [dict(role="user", content=ASK.format(w=w["原形"]))], a.key_env, budget,
                    temperature=a.temperature, max_tokens=a.max_tokens, json_mode=False,
                    chat_template_kwargs=None if a.plain else {"enable_thinking": False},
                    extra=None if a.plain else dict(seed=a.seed),
                    retries=0 if a.stop_on_limit else 4, wait=1 if a.stop_on_limit else 60)
        out.write(json.dumps(dict(語=w, 本文=o["text"], 発言=lines_of(o["text"]), 終わり方=o["finish"], 秒=round(time.time() - t1, 2),
                                  モデル=a.model), ensure_ascii=False) + "\n")
        out.flush()
        if a.sleep:
            time.sleep(a.sleep)
        if a.stop_on_limit and "429" in str(o["finish"]):
            print("上限に当たって止めた", flush=True)
            break
    out.close()
    print(json.dumps({"呼んだ回数": budget.used}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
