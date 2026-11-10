"""介入の研究 実験I35y: 話し相手を差し替えられる会話の試し（ローカル AI を話し相手にして話数を稼ぐ）

持ち主（2026-11-10）:「話数が欲しいから、専用のローカルAIに頼んでもいいかもしれない」
chat_bot3 の Bot3 と、話し相手（Ollama の /api/chat）が、交互に 1 行ずつ話す。話し相手に頼むのは 2 つだけ:
  1. ボットの返事を受けて、話を続ける（問いには、はい・いいえから答える）。
  2. ボットの行に変な所があれば、その所をそのまま「〜って？」と聞き返す。
ボットは会話をまたいで同じもの（経験の器・自覚の板・自分の板を持ち越す）。会話ごとの数を出す:
  指摘（ボットの手がかりの見分けで指摘とみた行）・受け（受けの板で前の返事を受けた行）・中身なしの返事（所の無い返事）。
本の行は非公開（Web 小説なら _wn）。記録は --log・--stats に書く（非公開の置き場へ）。
GPU を使う時は D:/AI_git/TmpFiles/gpu/queue.txt で予約する。ボット自身は CPU で回す（CUDA_VISIBLE_DEVICES=）。
実行（PC）:
  python intervention/chat_partner.py --cache <解析済み JSON> --route <指摘の板> --pointing <指摘の JSON> --uptake <受けの板> \
      --attrs <content_boards.json> --model qwen3:30b-a3b-instruct-2507-q4_K_M --openers <最初の行の txt> \
      --convs 20 --turns 12 --log <md> --stats <jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PARTNER = (
    "あなたは日本語で気軽に雑談する相手です。相手（ボット）は日本語を学んでいる途中で、ときどき変な文を言います。\n"
    "決まり:\n"
    "1. 返事は短い 1〜2 文。くだけた話し言葉で。\n"
    "2. 相手の返事を受けて、話を続ける。相手が問いを言ったら、まず「うん」か「ううん」で答えてから続ける。\n"
    "3. 相手の文に、意味が分からない所や変な言い方があれば、その所をそのまま抜き出して「〇〇って？」とだけ聞く。\n"
    "4. 説明や添削はしない。正しい言い方を教えない。ふつうの会話の相手としてふるまう。\n"
    "5. 話題は身の回りのこと（天気・食べ物・散歩・動物・家族・仕事など）。"
)


def ollama_chat(host, model, messages, temperature=0.7):
    body = json.dumps(dict(model=model, messages=messages, stream=False,
                           options=dict(temperature=temperature, num_predict=80))).encode("utf-8")
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read().decode("utf-8"))
    text = out["message"]["content"].strip()
    if "</think>" in text:                                   # 考えの部分を出すモデルは、その後ろだけ
        text = text.split("</think>", 1)[1].strip()
    return text.splitlines()[0].strip() if text else "うん。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--pointing")
    ap.add_argument("--uptake")
    ap.add_argument("--attrs")
    ap.add_argument("--dialogue")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--openers", required=True, help="会話の最初の行（1 行 1 つ）。話し相手が先に言う")
    ap.add_argument("--convs", type=int, default=10)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log")
    ap.add_argument("--stats")
    ap.add_argument("--no_one_clause", action="store_true")
    a = ap.parse_args()
    import spacy
    import chat_bot3 as B3
    import i35_route_grain as R
    nlp = spacy.load("ja_ginza")
    works = json.load(open(a.cache, encoding="utf-8"))
    bot = B3.Bot3(nlp, works, pointing=a.pointing, route=a.route, dialogue=a.dialogue, smooth="事後", attrs=a.attrs, use_at=bool(a.attrs))
    bot.trial_mode, bot.use_you = "unseen", True
    if a.uptake:
        bot.uptake = json.load(open(a.uptake, encoding="utf-8"))
    if a.attrs:
        bot.attr_field, bot.use_book, bot.conv_share = True, False, 0.2
    bot.use_chain = a.no_one_clause
    openers = [x.strip() for x in open(a.openers, encoding="utf-8") if x.strip()]
    rng = random.Random(a.seed)
    log = [f"# 話し相手との会話（非公開）", "", f"- 話し相手: {a.model}、会話 {a.convs} × {a.turns} 往復", ""]
    for ci in range(a.convs):
        bot.conv["lines"] = []                              # 会話の行は会話ごとに空にする（経験・自覚・自分の板は持ち越す）
        bot.asked, bot.trial, bot._asked_attr = None, [], None
        human = rng.choice(openers)
        msgs = [dict(role="system", content=PARTNER), dict(role="assistant", content=human)]
        c = dict(会話=ci + 1, 往復=0, 指摘=0, 受け=0, 中身なし=0)
        log += [f"## 会話 {ci + 1}", ""]
        prev_bot = None
        for ti in range(a.turns):
            if prev_bot is not None:                        # 話し相手の行が、前の返事への指摘か・受けか（ボットの見分けと同じ物差し）
                ta = bot.conv["lines"][-1]["toks"]
                A_ = dict(text="「" + prev_bot + "」", toks=ta)
                B_ = dict(text="「" + human + "」", toks=B3.CB.to_toks(nlp(human)))
                cs = R.cues(A_, B_, bot.negx)
                if bot._is_q(ta):                           # 問いへの「ううん」は答え（chat_bot3 と同じ）
                    cs = [x for x in cs if x[0] not in ("否定の始め", "否定の繰り返し")]
                if cs or "って？" in human:
                    c["指摘"] += 1
                elif bot.uptake is not None and R.uptake_score(bot.uptake, A_, B_) > 0:
                    c["受け"] += 1
            t0 = time.time()
            r = bot.reply(human)
            line = r["line"]
            c["往復"] += 1
            if not bot.pb.pieces(bot.conv["lines"][-1]["toks"]):
                c["中身なし"] += 1
            log += [f"- 相手: {human}", f"- ボット: **{line}** （{r['why']}、{time.time() - t0:.1f} 秒）"]
            msgs.append(dict(role="user", content=line))
            human = ollama_chat(a.host, a.model, msgs) if a.model != "echo" else "うん、そうなんだ。"   # echo: 試し用
            msgs.append(dict(role="assistant", content=human))
            prev_bot = line
        log += ["", f"- 数: {json.dumps(c, ensure_ascii=False)}", ""]
        print(c, flush=True)
        if a.stats:
            with open(a.stats, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(exp="I35y", model=a.model, **c), ensure_ascii=False) + "\n")
        if a.log:
            os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
            with open(a.log, "w", encoding="utf-8") as f:
                f.write("\n".join(log))


if __name__ == "__main__":
    main()
