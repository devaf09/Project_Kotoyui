"""介入の研究 実験I35y: 話し相手を差し替えられる会話の試し（ローカル AI を話し相手にして話数を稼ぐ）

持ち主（2026-11-10）:「話数が欲しいから、専用のローカルAIに頼んでもいいかもしれない」
chat_bot3 の Bot3 と、話し相手（Ollama の /api/chat）が、交互に 1 行ずつ話す。話し相手に頼むのは 2 つだけ:
  1. ボットの返事を受けて、話を続ける（問いには、はい・いいえから答える）。
  2. ボットの行に変な所があれば、その所をそのまま「〜って？」と聞き返す。
ボットは会話をまたいで同じもの（経験の器・自覚の板・自分の板を持ち越す）。会話ごとの数を出す:
  指摘（ボットの手がかりの見分けで指摘とみた行）・受け（受けの板で前の返事を受けた行）・中身なしの返事（所の無い返事）。
本の行は非公開（Web 小説なら _wn）。記録は --log・--stats に書く（非公開の置き場へ）。
GPU を使う時は D:/AI_git/TmpFiles/gpu/queue.txt で予約する。ボット自身は CPU で回す（CUDA_VISIBLE_DEVICES=）。
I35z（--teacher）: 先生役（持ち主の案 2026-11-10）が毎回の返事を見て、不自然なら おかしい所・お手本の返事・優しいことば を JSON で返す。
  ボットは、おかしい所を指摘の経験に、褒められた返事の所をそのままの経験に、お手本の所をそのままの経験に足す（Bot3.taught）。
  先生の段: 1 文法と言い方 → 続けて 3 回褒められたら 2（前の行とのつながり）→ 5 回で 3（会話全体の自然さ）。間違えたら数え直し。
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
import urllib.error
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


TEACHER = (
    "あなたは、日本語の会話を練習しているボットに付き添う、優しくて面倒見の良い先生です。\n"
    "「会話の流れ」と「ボットの返事」を読んで、ボットの返事に不自然な所があるかを見ます。\n"
    "決まり:\n"
    "1. 見る所: 文法の誤り、言い方の不自然さ、前の相手の行とのつながり。\n"
    "2. 指摘は「〜だよ」「〜のほうが自然かな？」のように、優しい話し言葉で。難しい言葉は使わない。\n"
    "3. 間違いがあれば、おかしい所をボットの返事からそのまま抜き出し（odd_part）、自然な返事の例（suggested_reply）を 1 つ出す。\n"
    "4. 間違いが無ければ「うんうん、自然だね！」のように褒める。\n"
    "出力は次の JSON だけ:\n"
    '{"has_error": true または false, "odd_part": "ボットの返事の中のおかしい所（無ければ空）", '
    '"suggested_reply": "自然な返事の例（無ければ空）", "intervention_message": "ボットへの優しいことば"}\n'
)
STAGES = {
    1: "今の段: 1（はじめ）。文法と言い方だけを見る。話のつながりは少しずれていても間違いにしない。",
    2: "今の段: 2。文法・言い方に加えて、前の相手の行とつながっているかも見る。",
    3: "今の段: 3。会話全体の流れの中で、人がふつう言う返事かどうかまで見る。",
}
UP = {1: 3, 2: 5}                                            # 続けて褒められた回数で段を上げる（持ち主の案）


def teacher_eval(host, model, history, reply, stage):
    hist = "\n".join(history[-6:])
    msgs = [dict(role="system", content=TEACHER + STAGES[stage]),
            dict(role="user", content=f"会話の流れ:\n{hist}\n\nボットの返事:\n{reply}")]
    text = ollama_chat(host, model, msgs, temperature=0.2, first_line=False, num_predict=512, fmt="json")
    j = text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""
    try:
        r = json.loads(j)
    except Exception:
        return None, "読めない"
    r["has_error"] = r.get("has_error") in (True, "true", "True")
    odd, sug = str(r.get("odd_part") or "").strip("「」。 "), str(r.get("suggested_reply") or "").strip()
    r["odd_part"], r["suggested_reply"] = odd, sug
    # 見立てが自分と食い違うものは経験に入れない（決まりの見本をそのまま返す・変と言った所をお手本でも言う・返事に無い所を変と言う）
    if r.get("intervention_message", "").strip() in ("ボットへの優しいことば", ""):
        return None, "見本の写し"
    if r["has_error"]:
        if not odd or odd not in reply:
            return None, "返事に無い所"
        if sug.strip("。") == reply.strip("。"):
            return None, "お手本が同じ"
    return r, ""


def ollama_chat(host, model, messages, temperature=0.7, first_line=True, num_predict=80, fmt=None, tries=3):
    """考えてから答える型のモデル（Qwen3・R1 系）は、考えに num_predict を使い切って本文が空になる。
    考えを止め（think: false）、それでも本文が空なら問い直す。最後まで空なら "" を返す（代わりの返事を作らない。数に残す）。"""
    for k in range(tries):
        opts = dict(model=model, messages=messages, stream=False, think=False,
                    options=dict(temperature=temperature, num_predict=num_predict * (k + 1)))
        if fmt:
            opts["format"] = fmt
        try:
            out = _post(host, opts)
        except urllib.error.HTTPError:                     # 考えを止められないモデルは think を外して問い直す
            opts.pop("think")
            out = _post(host, opts)
        text = out["message"]["content"].strip()
        if "</think>" in text:                               # 考えの部分を本文に出すモデルは、その後ろだけ
            text = text.split("</think>", 1)[1].strip()
        elif "<think>" in text:                              # 考えの途中で切れた
            text = ""
        if text:
            return text if not first_line else text.splitlines()[0].strip()
    return ""


def _post(host, body):
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


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
    ap.add_argument("--teacher", help="I35z: 先生役のモデル（Ollama の名前）。毎回の返事を見て、指摘をボットの経験に足す")
    ap.add_argument("--reader", default="ginza", choices=("ginza", "own"), help="I36: own = GiNZA を使わない読み手")
    ap.add_argument("--no_model_reply", action="store_true", help="先生のお手本の返事を経験に足さない")
    a = ap.parse_args()
    import chat_bot3 as B3
    import i35_route_grain as R
    works = json.load(open(a.cache, encoding="utf-8"))
    if a.reader == "own":                                   # I36: GiNZA を使わない読み手
        import i36_own_reader as OR
        nlp = OR.OwnNLP(works, json.load(open(a.attrs, encoding="utf-8")) if a.attrs else None)
    else:
        import spacy
        nlp = spacy.load("ja_ginza")
    B3.Bot3.use_variants = a.reader == "own"
    bot = B3.Bot3(nlp, works, pointing=a.pointing, route=a.route, dialogue=a.dialogue, smooth="事後", attrs=a.attrs, use_at=bool(a.attrs))
    bot.trial_mode, bot.use_you = "unseen", True
    if a.uptake:
        bot.uptake = json.load(open(a.uptake, encoding="utf-8"))
    if a.attrs:
        bot.attr_field, bot.use_book, bot.conv_share = True, False, 0.2
    bot.use_chain = a.no_one_clause
    openers = [x.strip() for x in open(a.openers, encoding="utf-8") if x.strip()]
    rng = random.Random(a.seed)
    log = [f"# 話し相手との会話（非公開）", "", f"- 話し相手: {a.model}、先生: {a.teacher or 'なし'}、会話 {a.convs} × {a.turns} 往復", ""]
    stage, streak = 1, 0
    for ci in range(a.convs):
        bot.conv["lines"] = []                              # 会話の行は会話ごとに空にする（経験・自覚・自分の板は持ち越す）
        bot.asked, bot.trial, bot._asked_attr = None, [], None
        human = rng.choice(openers)
        msgs = [dict(role="system", content=PARTNER), dict(role="assistant", content=human)]
        c = dict(会話=ci + 1, 往復=0, 指摘=0, 受け=0, 中身なし=0, 先生の指摘=0, 先生の段=stage, 先生の見立てを捨てた=0, 相手の空=0)
        hist = [f"相手: {human}"]
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
            if a.teacher:                                   # 先生の見立て（話し相手とは別）
                ev, bad = teacher_eval(a.host, a.teacher, hist, line, stage)
                if ev is not None and ev["has_error"] and ev["suggested_reply"]:   # 変と言った所を、お手本でも言っている（同じ所に両方の経験が入る）
                    key = lambda t: {(p["ns"], p["c"], p["ps"]) for p in bot.pb.pieces(B3.CB.to_toks(nlp(t)))}
                    if key(ev["odd_part"]) & key(ev["suggested_reply"]):
                        ev, bad = None, "お手本が同じ"
                if bad:
                    c["先生の見立てを捨てた"] += 1
                    log.append(f"  - 先生（段 {stage}）: 見立てを捨てた（{bad}）")
                if ev is not None:
                    got = bot.taught(ev["has_error"], ev.get("odd_part", ""), "" if a.no_model_reply else ev.get("suggested_reply", ""))
                    if ev["has_error"]:
                        c["先生の指摘"] += 1
                        streak = 0
                    else:
                        streak += 1
                        if stage in UP and streak >= UP[stage]:
                            stage, streak = stage + 1, 0
                    log.append(f"  - 先生（段 {stage}）: {ev.get('intervention_message', '')} ／ 経験: {'・'.join(k + ':' + w for k, w in got) or 'なし'}")
            hist.append(f"ボット: {line}")
            msgs.append(dict(role="user", content=line))
            human = ollama_chat(a.host, a.model, msgs) if a.model != "echo" else "うん、そうなんだ。"   # echo: 試し用
            if not human:                                   # 話し相手の本文が空（数に残し、会話はここで切る）
                c["相手の空"] += 1
                break
            msgs.append(dict(role="assistant", content=human))
            hist.append(f"相手: {human}")
            prev_bot = line
        log += ["", f"- 数: {json.dumps(c, ensure_ascii=False)}", ""]
        print(c, flush=True)
        if a.stats:
            with open(a.stats, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(exp="I35z" if a.teacher else "I35y", model=a.model, teacher=a.teacher, **c), ensure_ascii=False) + "\n")
        if a.log:
            os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
            with open(a.log, "w", encoding="utf-8") as f:
                f.write("\n".join(log))


if __name__ == "__main__":
    main()
