"""実験22（非公開の材料）: 主語を隠した人称当て（実験19〜21）の天井を、ローカル AI に解かせて測る（2026-11-04）

問: 実験21 の dump と同じ問（../data/persona/subject_items_s0.jsonl の試し 3,315 問）から、人称ごとに同じ数（--n_per）を取る。
    ローカル AI（Ollama）に、主語を消した発言を見せ、「省かれた主語は 話し手・相手・第三者 のどれか」を 1 語で答えさせる。
読み方: AI も 55% 前後なら問そのものの限界（発言 1 つでは決まらない）。ずっと高ければ、ことゆいの手がかりの側が足りない。
結果: ../results/exp22.txt、答え ../docs/exp22_answers.json（非公開）
実行（Project_Kotoyui の直下で）: python experiments/exp22_subject_ceiling.py --ai=<モデル名>
"""
import argparse
import json
import os
import random
import re
import sys
import time

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
sys.path.insert(0, os.path.join(KOTO_ROOT, "tools"))
from ollama_client import Ollama  # noqa: E402

PERSONS = ["話し手", "相手", "第三者"]
Q = ("会話の発言から主語（だれが）が省かれています。省かれた主語がだれかを、次の 3 つから選んでください。\n"
     "話し手 = この発言を言った本人（「私は」が省かれている）\n相手 = この発言を聞いている人（「あなたは」が省かれている）\n"
     "第三者 = その場にいない別の人（「彼は」「彼女は」が省かれている）\n\n"
     "例: 「昨日カレーを作りました」→ 話し手\n例: 「明日は来られますか？」→ 相手\n例: 「最近忙しいらしいよ」→ 第三者\n\n"
     "発言: 「{t}」→")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", required=True)
    ap.add_argument("--n_per", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(os.path.join(PRIV, "data", "persona", "subject_items_s0.jsonl"), encoding="utf-8")]
    te = [x for x in items if x["split"] == "test"]
    rng = random.Random(a.seed)
    qs = []
    for k in range(3):
        sub = [x for x in te if x["label"] == k]
        qs += rng.sample(sub, a.n_per)
    rng.shuffle(qs)
    ai = Ollama(a.ai, timeout=300)
    t0 = time.perf_counter()
    ans, ok, per = [], 0, {p: [0, 0] for p in PERSONS}
    for i, x in enumerate(qs):
        try:
            out = ai._post([dict(role="user", content=Q.format(t=x["text"]))], temperature=0.0, max_tokens=48)
        except Exception as e:
            out = f"（失敗: {e}）"
        m = re.search("話し手|相手|第三者", out)
        pred = m.group() if m else None
        gold = PERSONS[x["label"]]
        ok += pred == gold
        per[gold][0] += pred == gold
        per[gold][1] += 1
        ans.append(dict(text=x["text"], gold=gold, pred=pred, raw=out[:60]))
        if (i + 1) % 50 == 0:
            print(i + 1, f"{100 * ok / (i + 1):.1f}%", f"{time.perf_counter() - t0:.0f} 秒", flush=True)
    rec = dict(exp="22", tag=a.tag, ai=a.ai, n=len(qs), acc=100 * ok / len(qs), per_class={k: 100 * v[0] / v[1] for k, v in per.items()},
               no_answer=sum(x["pred"] is None for x in ans) / len(ans), total_s=time.perf_counter() - t0)
    print(rec, flush=True)
    with open(os.path.join(PRIV, "results", "exp22.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(PRIV, "docs", "exp22_answers.json"), "w", encoding="utf-8") as f:
        json.dump(ans, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
