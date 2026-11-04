"""ローカル AI の評点で、ことゆいの設定を比べる（持ち主の判断 2026-11-03「ある程度まではローカル AI の評価で十分」）

試しの会話の発言（会話コンソールの評点だけの画面と同じ問の並び）に、設定ごとのことゆいで返事を付け、ローカル AI（Ollama）に
「会話の続きとして自然か」を 1〜5 で付けさせる。同じ問・同じ乱数の種で比べる。
    python tools/eval_ai.py --ai=<モデル名> --n 150 --empathy 1.0,0.2,0.0     # 気持ちの出ていない発言に「〜んですね」を付ける割合を比べる
    python tools/eval_ai.py --agreement                                     # 会話コンソールの持ち主の評点（👍・👎）と、AI の評点がどれくらい合うか
記録: 非公開の置き場の results/eval_ai.jsonl（返事の文は材料の発言を含むので、リポジトリには入れない）。数字の要約だけを NOTES に書く。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kotoyui  # noqa: E402

EMPATHY = re.compile(r"^[^。！？]{1,30}(んですね|ですね)[。！]")


def agreement():
    """持ち主の 👍・👎 と AI の評点（1〜5）の合い方。"""
    path = os.path.join(kotoyui.PRIVATE, "ratings", "ratings.jsonl")
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    rs = [r for r in rs if r.get("ai_score") and r.get("rating") in ("good", "bad")]
    good = [r["ai_score"] for r in rs if r["rating"] == "good"]
    bad = [r["ai_score"] for r in rs if r["rating"] == "bad"]
    pairs = [(g, b) for g in good for b in bad]
    auc = sum(1.0 if g > b else 0.5 if g == b else 0.0 for g, b in pairs) / max(1, len(pairs))
    hit = sum((r["ai_score"] >= 3) == (r["rating"] == "good") for r in rs) / max(1, len(rs))
    print(f"評点 {len(rs)}（👍 {len(good)}・👎 {len(bad)}）")
    print(f"AI の平均: 👍 の返事 {statistics.mean(good):.2f}・👎 の返事 {statistics.mean(bad):.2f}")
    print(f"見分ける力（AUC、0.5 = でたらめ）: {auc:.3f}、3 以上 = 👍 とみたときの一致: {100 * hit:.1f}%")
    for emp in (True, False):
        sub = [r for r in rs if bool(EMPATHY.match(r["reply"])) == emp]
        if sub:
            print(f"「〜んですね」{'あり' if emp else 'なし'}: {len(sub)} 個、👍 {100 * sum(r['rating'] == 'good' for r in sub) / len(sub):.0f}%、AI の平均 {statistics.mean(r['ai_score'] for r in sub):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--empathy", default="1.0,0.2,0.0", help="気持ちの出ていない発言に「〜んですね」を付ける割合（コンマで区切って比べる）")
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--assemble", default="full", choices=["full", "plain"], help="full = 共感の型 + 選んだ返事 + 質問、plain = 選んだ返事だけ")
    ap.add_argument("--agreement", action="store_true")
    a = ap.parse_args()
    if a.agreement:
        return agreement()
    from console import prompts
    from ollama_client import Ollama
    ai = Ollama(a.ai, timeout=300)
    qs = prompts()[:a.n]
    out = os.path.join(kotoyui.PRIVATE, "results", "eval_ai.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {}                                                                     # 途中で止まっても続きから（同じ版・設定・問は付け直さない）
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r.get("version") == kotoyui.__version__ and r.get("light") == a.light and r.get("ai") == a.ai and r.get("assemble", "full") == a.assemble:
                done[(r["empathy_plain"], r["prompt"])] = r

    def rate(u, reply):
        for k in range(3):                                                       # ローカル AI が混んで時間切れになったら、やり直す
            try:
                return ai.rate([u], reply)
            except Exception as e:
                why = f"（評点できず: {e}）"
                time.sleep(5)
        return None, why
    summary = {}
    for v in [float(x) for x in a.empathy.split(",")]:
        bot = kotoyui.Bot(seed=0, light=a.light, empathy_plain=v, assemble=a.assemble, log=lambda *_: None)
        scores, emp, t0 = [], 0, time.perf_counter()
        with open(out, "a", encoding="utf-8") as f:
            for u in qs:
                reply, conf = bot.reply([u])
                if (v, u) in done:
                    r = done[(v, u)]
                    emp += bool(EMPATHY.match(r["reply"]))
                    if r.get("ai_score"):
                        scores.append(r["ai_score"])
                    continue
                sc, why = rate(u, reply)
                emp += bool(EMPATHY.match(reply))
                if sc:
                    scores.append(sc)
                f.write(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), version=kotoyui.__version__, empathy_plain=v, light=a.light, assemble=a.assemble,
                                        ai=a.ai, prompt=u, reply=reply, confidence=conf, ai_score=sc, ai_reason=why), ensure_ascii=False) + "\n")
        summary[v] = dict(mean=statistics.mean(scores), good=sum(s >= 4 for s in scores) / len(scores), empathy=emp / len(qs), n=len(scores))
        print(f"共感_ふだん {v}: AI の平均 {summary[v]['mean']:.2f}・4 以上 {100 * summary[v]['good']:.0f}%・「〜んですね」付き {100 * summary[v]['empathy']:.0f}%"
              f"（{len(scores)} 問、{time.perf_counter() - t0:.0f} 秒）", flush=True)


if __name__ == "__main__":
    main()
