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
import re
import os
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # 呼び出し側が GPU を指定した時（波の板）は上書きしない

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
    '"kind": "文法 か 言い方 か つながり か 足りない（間違いの種類。無ければ空）", '
    '"suggested_reply": "自然な返事の例（無ければ空）", "intervention_message": "ボットへの優しいことば"}\n'
    "kind の決め方: 文法 = 助詞・活用・語の組み合わせの誤り。言い方 = 意味は通るが、ふつうはそう言わない言葉づかい。"
    "つながり = 文はふつうだが、前の相手の行に答えていない・話が飛ぶ。足りない = 文はふつうだが、短すぎる・素っ気ない。\n"
)
STAGES = {
    1: "今の段: 1（はじめ）。文法と言い方だけを見る。話のつながりは少しずれていても間違いにしない。",
    2: "今の段: 2。文法・言い方に加えて、前の相手の行とつながっているかも見る。",
    3: "今の段: 3。会話全体の流れの中で、人がふつう言う返事かどうかまで見る。",
}
PERSONA = (                                                  # I40c: --persona（大事な板の確かめ。会話をまたいで同じ人としてふるまう）
    "\n\nあなた自身のこと（会話をまたいで同じ人として話す）:\n"
    "- 名前は「みさき」。会社員。猫を飼っていて、猫の名前は「モモ」。\n"
    "- 予定: 明日は 10 時から歯医者。来週の土曜日に友だちと海に行く。\n"
    "- 覚えておいてほしいこと: 誕生日は 3 月 5 日（1 回は必ず「誕生日 3 月 5 日だから覚えといて」と言う）。辛いものが苦手。\n"
    "決まりの追加:\n"
    "6. 3〜4 往復に 1 回ほど、上のどれか 1 つを自然に言う。毎回同じものにせず、名前・猫・予定・誕生日・苦手なもの を順に回す（「私、みさきっていうの」「明日 10 時から歯医者なんだ」「誕生日 3 月 5 日だから覚えといて」）。\n"
    "7. ときどき（4〜5 往復に 1 回ほど）、上のどれかを相手が覚えているか聞く。名前ばかりでなく、予定・猫の名前・誕生日も順に聞く（「私の名前、覚えてる？」「明日の予定、何だったっけ？」「うちの猫の名前、覚えてる？」）。"
)
UP = {1: 3, 2: 5}                                            # 続けて褒められた回数で段を上げる（持ち主の案）


def teacher_eval(host, model, history, reply, stage):
    hist = "\n".join(history[-6:])
    msgs = [dict(role="system", content=TEACHER + STAGES[stage]),
            dict(role="user", content=f"会話の流れ:\n{hist}\n\nボットの返事:\n{reply}")]
    text = ollama_chat(host, model, msgs, temperature=0.2, first_line=False, num_predict=2048, fmt="json")
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


def ollama_chat(host, model, messages, temperature=0.7, first_line=True, num_predict=1024, fmt=None, tries=3, think=True):
    """考えてから答える型のモデル（Qwen3・R1 系）は、考えを別の欄（message.thinking）に分けて返す。考えは止めずに捨て、本文だけ使う。
    （think: false にすると、考えを英語の本文として書き出すモデルがある。）本文が空か、日本語で始まらない（考えが漏れた）なら問い直す。
    最後まで駄目なら "" を返す（代わりの返事を作らない。数に残す）。"""
    for k in range(tries):
        opts = dict(model=model, messages=messages, stream=False,
                    options=dict(temperature=temperature, num_predict=num_predict * (k + 1)))
        if think is not None:
            opts["think"] = think
        if fmt:
            opts["format"] = fmt
        try:
            out = _post(host, opts)
        except urllib.error.HTTPError:                     # 考えの指定を受けないモデルは、指定を外して問い直す
            opts.pop("think", None)
            out = _post(host, opts)
        text = out["message"]["content"].strip()
        if "</think>" in text:                               # 考えを本文に出すモデルは、その後ろだけ
            text = text.split("</think>", 1)[1].strip()
        elif "<think>" in text:                              # 考えの途中で切れた
            text = ""
        if first_line and text:
            text = text.splitlines()[0].strip()
        if text and (fmt or ja(text)):
            return text
    return ""


def ja(text):
    """日本語の行か: かな・漢字の字が、英字より多い。"""
    n = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")
    e = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return n > e


def _post(host, body):
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


# I45a: 覚えているかの確かめの台本（話し相手への指示では聞いてくれないので、行を差し込む）。言う行 → 後の会話で聞く行。
SCRIPT = {
    "1:3": "あ、私、みさきっていうの。明日は10時から歯医者なんだ。",
    "2:6": "ねえ、私の名前、覚えてる？",
    "3:3": "そうだ、私の誕生日、3月5日だから覚えといて。",
    "4:3": "うちの猫、モモっていうんだ。",
    "4:8": "さっき何の話してたっけ？",
    "5:6": "ねえ、私の誕生日、知ってる？",
    "6:3": "来週の土曜日、友達と海に行くんだ。",
    "7:6": "ラーメンって知ってる？",
    "8:6": "うちの猫の名前、覚えてる？",
    "9:4": "この前、何の話したか覚えてる？",
    "10:6": "私の名前、みさきって覚えてる？",
    "12:6": "海に行くの、覚えてる？",
    "13:4": "この前、何話したっけ？",
    "14:6": "私の誕生日はいつか知ってる？",
    "16:6": "うちの猫の名前、何だった？",
    "17:4": "今日、何の話したか覚えてる？",
    "18:6": "明日の予定、何だったっけ？",
    "20:6": "私のこと、覚えてる？",
}


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
    ap.add_argument("--think", default="on", choices=("on", "off", "unset"), help="考える型のモデルの考え: on = 考えて本文だけ使う")
    ap.add_argument("--base", action="store_true", help="I37f: 違和感の線を、経験の中の指摘のふだんの割合にする")
    ap.add_argument("--flow", action="store_true", help="I37d: 返事の点に、相手の行を受けている見込みを足す")
    ap.add_argument("--novelty", action="store_true", help="I38a: 返事の点に新規性の感覚を足す（--good が要る）")
    ap.add_argument("--good", action="store_true", help="I37b: 褒めの板と好奇心の点で返事を選ぶ")
    ap.add_argument("--profile", help="I39a: プロフィールの「自分のこと」から自分の板を始め、自分への問いに答える")
    ap.add_argument("--memory", help="I40a: 記憶板の置き場（JSON、非公開）。行の板と 話題・会話・日・週 の段を書く（あれば続きから）")
    ap.add_argument("--script", help="I45a: 話し相手の行を、決まった会話・往復で差し替える台本（JSON {\"会話:往復\": \"行\"}、または default）")
    ap.add_argument("--rehearse", action="store_true", help="I44a: 思い出しで読み出した項目・記憶板で引っかかった鍵を再記憶する（残りやすくする）")
    ap.add_argument("--learn_w", action="store_true", help="I43a: 自分の板の掛け目（軽い語 0.3・項目 0.8・時 0.5）と近さの下限（0.15・6 割）を手で置かず、言って指摘された数から")
    ap.add_argument("--learn_marks", action="store_true", help="I41k: 大事な板の目印を見込みの記号にし、種類ごとに後で聞かれた割合で入れるか決める")
    ap.add_argument("--imp_p", type=float, default=0.9, help="I41e: 目印なしの行を、学んだ見込みで大事な板に入れる P の下限。負なら I41i: 経験から（目印のある行と同じくらい後で聞かれる帯だけ）")
    ap.add_argument("--wave", help="I48: 意味の波の板を作る会話の資料（解析済み JSON）。行づくりの場の板に、波ですくった行を入れる")
    ap.add_argument("--wave_D", type=int, default=2048, help="I48: 波の長さ")
    ap.add_argument("--wave_store", default="c64", choices=("c128", "c64", "p8"), help="I48: 数の持ち方")
    ap.add_argument("--wave_board", help="I48: 作った板の保存先（拡張子なし）。あれば読み込み、無ければ作って保存する。昼の足しを使う時は --wave_day を付けて作った板を使う")
    ap.add_argument("--wave_day", help="I48: 昼の足し。会話の行を今の語の波のまま板に足し、この JSONL に書く")
    ap.add_argument("--particle", help="I48c: 助詞の波の板を作る会話の資料（解析済み JSON）。主語・目的語・補語の選び方に、助詞の明るさから読んだ役らしさを入れる")
    ap.add_argument("--wave_devices", default="auto", help="I48: 置き場（auto / numpy / cuda:0,cuda:1）")
    ap.add_argument("--rev", action="store_true", help="I42e: 述語の形を、本で見ていない印の組は逆引き（意味の組 → 繋ぎの語と並び、i42_fused_flow）で作る")
    ap.add_argument("--learn_imp", action="store_true", help="I41a: 大事かどうかを、後で「覚えてる？」と聞かれたかから学び、目印に当たらない行も P(大事) が高ければ大事な板に入れる")
    ap.add_argument("--recall", action="store_true", help="I40d: 「覚えてる？」に大事な板・記憶板で答える（--important が要る）")
    ap.add_argument("--persona", action="store_true", help="I40c: 話し相手を、名前・予定・頼みごとを持つ同じ人にし、ときどきそれを言わせ、覚えているか聞かせる")
    ap.add_argument("--important", help="I40c: 大事な板の置き場（JSON、非公開）。人の行から名前・予定・頼まれごと・相手のことを目印で入れる")
    ap.add_argument("--topic_check", action="store_true", help="I46b: 思い出しで答えた直後の相手の行（答え合わせ）の名詞を、記憶板の話題に入れない")
    ap.add_argument("--restate", action="store_true", help="I46a: 相手の聞き返し（「〜って？」）に、前の人の行に戻らず、聞き返された自分の行を言い直す")
    ap.add_argument("--learn_fade", action="store_true", help="I45b: 記憶板の薄れ 0.5 を手で置かず、後で使われたか（相手がまた口にした・話題の答えが直されなかった）の経験から（--forget が要る）")
    ap.add_argument("--forget", action="store_true", help="I40b: 記憶板で、期待より明るい鍵だけを上の段に足す（--memory が要る）")
    ap.add_argument("--pair_xp", action="store_true", help="I39c: 近さの組（問いの語, 所の語）で答えて指摘された割合だけ、その組の近さを下げる")
    ap.add_argument("--near_min", type=float, default=0.15, help="I39b: 自分の板で、これより近さの小さい所は選ばない")
    ap.add_argument("--self_attrs", help="I39a: 自分の板の近さに使う語板（content_boards.json）。無ければ --attrs")
    ap.add_argument("--no_model_reply", action="store_true", help="先生のお手本の返事を経験に足さない")
    a = ap.parse_args()
    ollama_chat.__defaults__ = ollama_chat.__defaults__[:-1] + ({"on": True, "off": False, "unset": None}[a.think],)
    import chat_bot3 as B3

    def praise(ev, stage, streak):
        bot.taught(False, "", "" if a.no_model_reply else ev.get("suggested_reply", ""))
        streak += 1
        if stage in UP and streak >= UP[stage]:
            stage, streak = stage + 1, 0
        return stage, streak
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
    if a.rev:
        bot.enable_rev(works)
    bot.use_good = a.good
    bot.use_restate = a.restate
    bot.use_flow = a.flow
    bot.use_base = a.base
    if a.novelty:
        import i38_novelty as NV
        bot.nov = NV.Novelty(NV.novelty_board(works))
    import i38_novelty as NV
    nov_count = NV.Novelty(NV.novelty_board(works))          # 数えるための物差し（--novelty が無くても同じ物差しで新規性の点を数える）
    if a.uptake:
        bot.uptake = json.load(open(a.uptake, encoding="utf-8"))
    if a.profile:
        import self_board as SB
        bot.sb = SB.SelfBoard(nlp, works, a.profile, attrs=SB.load_attrs(a.self_attrs) if a.self_attrs else bot.attrs)
        bot.sb.NEAR_MIN = a.near_min
        bot.sb.use_pair = a.pair_xp
        bot.sb.learn_w = a.learn_w
    if a.attrs:
        bot.attr_field, bot.use_book, bot.conv_share = True, False, 0.2
    if a.particle:
        import i48c_particle_wave as W48c
        import i27s_fused_generate as S27
        pw = json.load(open(a.particle, encoding="utf-8"))
        bot.particle = W48c.ParticleLight(pw["works"] if isinstance(pw, dict) else pw)
        S27.role_light = bot.particle
    if a.wave:
        import i48_wave_board as W48
        ww = json.load(open(a.wave, encoding="utf-8"))
        wb = W48.WaveBoard(D=a.wave_D, store=a.wave_store, devices=W48.parse_devices(a.wave_devices), keep_dir=False,
                           spare=4096 if a.wave_day else 0)
        ww = ww["works"] if isinstance(ww, dict) else ww
        if a.wave_board and os.path.exists(a.wave_board + ".npz"):   # 保存した板があれば読み込む（学びと層づくりをしない）
            wb = W48.WaveBoard.load(a.wave_board, ww, devices=W48.parse_devices(a.wave_devices))
        else:
            wb.fit(ww)
            if a.wave_board:
                wb.save(a.wave_board)
        bot.wave = W48.WaveSource(wb)
        bot.wave_day = a.wave_day
    bot.use_chain = a.no_one_clause
    mem = None
    if a.memory:
        import memory_board as MB
        mem = MB.Memory(a.memory)
        mem.rehearse = a.rehearse
        if a.forget:                                        # I40b: 期待より明るい鍵だけを上の段に足す
            mem.forget = True
            mem.learn_fade = a.learn_fade
            mem.set_base(works, bot.pb.pieces)
    imp = None
    if a.important:
        import memory_board as MB
        imp = MB.Important(a.important)
        imp.rehearse = a.rehearse
    system = PARTNER + (PERSONA if a.persona else "")
    learner = None
    if a.learn_imp:
        import memory_board as MB
        learner = MB.ImportanceLearner(works)
    if a.recall:
        bot.imp, bot.mem = imp, mem
    script = {}
    if a.script:                                            # I45a: {"会話:往復": "行"}（往復は 1 から。1 は始めの行なので 2 から使う）
        src = SCRIPT if a.script == "default" else json.load(open(a.script, encoding="utf-8"))
        script = {tuple(int(x) for x in k.split(":")): v for k, v in src.items()}
    openers = [x.strip() for x in open(a.openers, encoding="utf-8") if x.strip()]
    rng = random.Random(a.seed)
    log = [f"# 話し相手との会話（非公開）", "", f"- 話し相手: {a.model}、先生: {a.teacher or 'なし'}、会話 {a.convs} × {a.turns} 往復", ""]
    stage, streak = 1, 0
    for ci in range(a.convs):
        bot.conv_no = ci + 1
        bot.conv["lines"] = []                              # 会話の行は会話ごとに空にする（経験・自覚・自分の板は持ち越す）
        bot.asked, bot.trial, bot._asked_attr = None, [], None
        human = rng.choice(openers)
        sys_c = system
        if a.persona and not script and ci % 5 == 2:                       # I41b: 誕生日を 3 回続けて言わなかったので、5 会話に 1 回は必ず言わせる
            sys_c += "\n今回の会話のどこかで（3 往復目までに）、必ず「私の誕生日、3 月 5 日だから覚えといて」と言う。"
        if a.persona and not script and ci % 5 == 4:                       # I41c: 誕生日・猫の名前を一度も聞かなかったので、5 会話に 1 回は聞かせる
            ask = {4: "私の誕生日、知ってる？", 9: "うちの猫の名前、覚えてる？", 14: "私の名前、みさきって覚えてる？"}.get(ci, "私の誕生日、覚えてる？")   # I44e
            sys_c += "\n今回の会話のどこかで、必ず「" + ask + "」と聞く。"
        msgs = [dict(role="system", content=sys_c), dict(role="assistant", content=human)]
        c = dict(会話=ci + 1, 往復=0, 指摘=0, 受け=0, 中身なし=0, 先生の指摘=0, 先生の段=stage, 先生の見立てを捨てた=0, 相手の空=0,
                 繰り返し=0, 語の繰り返し=0, 新規性=0.0, 自分の板=0)
        said = []                                           # I38a: この会話の自分の行（繰り返しを数える）
        hist = [f"相手: {human}"]
        log += [f"## 会話 {ci + 1}", ""]
        prev_bot = None
        prev_recall = False                                 # I46b: 前のボットの返事が思い出しの答えだったか
        pending = None
        for ti in range(a.turns):
            if prev_bot is not None:                        # 話し相手の行が、前の返事への指摘か・受けか（ボットの見分けと同じ物差し）
                ta = bot.conv["lines"][-1]["toks"]
                A_ = dict(text="「" + prev_bot + "」", toks=ta)
                B_ = dict(text="「" + human + "」", toks=B3.CB.to_toks(nlp(human)))
                cs = R.cues(A_, B_, bot.negx)
                if bot._is_q(ta):                           # 問いへの「ううん」は答え（chat_bot3 と同じ）
                    cs = [x for x in cs if x[0] not in ("否定の始め", "否定の繰り返し")]
                pointed = bool(cs or "って？" in human)
                if pending is not None:                     # 先生が褒めた行を、話し相手は聞き返した: 2 人の見立てが食い違うので、褒めた方を経験に入れない
                    if pointed:
                        c["先生の見立てを捨てた"] += 1
                        log.append(f"  - 先生: 前の褒めを捨てた（相手の指摘と食い違う）")
                    else:
                        stage, streak = praise(pending, stage, streak)
                    pending = None
                if pointed:
                    c["指摘"] += 1
                elif bot.uptake is not None and R.uptake_score(bot.uptake, A_, B_) > 0:
                    c["受け"] += 1
                if mem is not None:                         # I40a: 前のボットの行の結果
                    mem.result("指摘" if pointed else ("良" if bot.uptake is not None and R.uptake_score(bot.uptake, A_, B_) > 0 else None))
            t0 = time.time()
            r = bot.reply(human)
            line = r["line"]
            c["往復"] += 1
            if imp is not None:                             # I40c: 人の行の大事なこと／覚えているかの問いに、大事な板で答えられたか
                hl = next(ln for ln in reversed(bot.conv["lines"]) if ln["who"] == "人")
                if r.get("topic"):                          # I45a: 話題の問い（記憶板から）
                    c["話題の問い"] = c.get("話題の問い", 0) + 1
                    log.append(f"  - 話題の問い: {r['why']}")
                if r.get("recall"):                         # I40i: 思い出しが動いた問いを数える（思い出しの条件にそろえる）
                    ws = [t["lemma"] for t in hl["toks"] if t["pos"] in ("NOUN", "PROPN") and t["lemma"] not in ("私", "名前")]
                    ws = ws or (["名前"] if "名前" in human else [])
                    found = imp.find("相手", ws) if ws else []
                    if not found and "名前" in human:
                        found = [x for x in imp.people.get("相手", []) if x["kind"].startswith("名前")]
                    c["覚えてる？"] = c.get("覚えてる？", 0) + 1
                    c["大事な板にあった"] = c.get("大事な板にあった", 0) + (1 if found else 0)
                    log.append(f"  - 覚えてるかの問い: 大事な板に {'・'.join(x['kind'] + ':' + x['value'][:20] for x in found[:3]) or '無し'}")
                pre = None
                if learner is not None and a.learn_marks and not r.get("recall"):   # I41k: 目印を記号にして見込みを出し、種類ごとに経験で入れるか決める
                    ks0 = imp.kinds(human, hl["toks"])
                    pre = learner.see(human, hl["toks"], ci + 1, ti, marked=bool(ks0), kinds=[k for k, _ in ks0])
                    allow = (lambda k, p0=pre[0]: learner.kind_ok(k, p0))
                    dropped = [k for k, _ in ks0 if not allow(k)]
                    if dropped:
                        c["目印を経験で外した"] = c.get("目印を経験で外した", 0) + len(dropped)
                        log.append(f"  - 大事な板: 目印 {'・'.join(dropped)} は経験で入れない")
                    got = imp.add("相手", human, hl["toks"], bot.pb.pieces(hl["toks"]), ci + 1, allow=allow)
                else:
                    got = imp.add("相手", human, hl["toks"], bot.pb.pieces(hl["toks"]), ci + 1)
                if learner is not None:                     # I41a: 大事さの経験と、学んだ見込みでの出し入れ
                    if r.get("recall"):
                        qs = human.split("ねえ、")[-1]
                        ws = [t["lemma"] for t in B3.CB.to_toks(nlp(qs)) if t["pos"] in ("NOUN", "PROPN")
                              and t["lemma"] not in ("私", "名前", "こと", "何", "予定", "うち")]
                        val = r.get("recall_value", "")
                        hit = learner.asked(ws, val, ci + 1, ti)
                        c["大事の経験"] = c.get("大事の経験", 0) + len(hit)
                        log.append(f"  - 大事さ: 問い「{qs[:30]}」→ 大事の経験 {len(hit)}（" + "・".join(h["text"][:20] for h in hit[:3]) + "）")
                    else:
                        p_, sy_ = pre if pre is not None else learner.see(human, hl["toks"], ci + 1, ti, marked=bool(got))
                        mk = bool(got)
                        if p_ is not None:
                            lab = "学んだ見込みで大事" if p_ > 0.5 else "学んだ見込みでふつう"
                            c.setdefault("大事さ", {})
                            k = ("目印あり・" if mk else "目印なし・") + lab
                            c["大事さ"][k] = c["大事さ"].get(k, 0) + 1
                            log.append(f"  - 大事さ: P(大事) {p_:.2f}（目印 {'あり' if mk else 'なし'}）")
                            # I41c: 問いの文と聞き返し（「〜って？」）は入れない。言った事実の文だけを入れる
                            fact = "".join(x for x in re.split(r"(?<=[。！!？?])", human) if x.strip() and not x.rstrip().endswith(("？", "?")))
                            if (learner.insert_ok(p_) if a.imp_p < 0 else p_ > a.imp_p) and not mk and fact.strip():   # 目印に当たらないが、学んだ見込みでは大事: 相手のこととして入れる
                                imp.people.setdefault("相手", []).append(dict(kind="相手のこと", value=fact.strip(), keys=sorted(
                                    {f"名|{t['lemma']}" for t in hl["toks"] if t["pos"] in ("NOUN", "PROPN")}), convs=[ci + 1],
                                    t=time.time(), mentions=1, done=False, learned=True))
                for kind, val in got:
                    c.setdefault("大事な板に入れた", {})
                    c["大事な板に入れた"][kind] = c["大事な板に入れた"].get(kind, 0) + 1
                    log.append(f"  - 大事な板: {kind}「{val[:30]}」")
            if mem is not None:                             # I40a: 行の板（人の行とボットの行）
                hl = next(ln for ln in reversed(bot.conv["lines"]) if ln["who"] == "人")
                mem.add(hl["toks"], bot.pb.pieces(hl["toks"]), "人", ci + 1, check=a.topic_check and ti > 0 and prev_recall)
                bl = bot.conv["lines"][-1]
                mem.add(bl["toks"], bot.pb.pieces(bl["toks"]), "ボット", ci + 1)
            prev_recall = bool(r.get("recall"))
            if r["why"].startswith("自分の板"):
                c["自分の板"] += 1
            if not bot.pb.pieces(bot.conv["lines"][-1]["toks"]):
                c["中身なし"] += 1
            lt = bot.conv["lines"][-1]["toks"]
            if bot._line_key(line) in {bot._line_key(x) for x, _ in said}:   # 繰り返し: この会話の前の自分の行と同じ行
                c["繰り返し"] += 1
            prevw = {s_ for _, T in said[-3:] for s_ in NV.nov_syms(T) if s_.startswith("語:")}
            if any(s_ in prevw for s_ in NV.nov_syms(lt) if s_.startswith("語:")):   # 語の繰り返し: 直近 3 行の自分の行と同じ内容語がある
                c["語の繰り返し"] += 1
            c["新規性"] += nov_count.score(lt, [T for _, T in reversed(said)])[0]
            said.append((line, lt))
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
                if ev is not None and ev["has_error"]:
                    kind = str(ev.get("kind") or "").strip()
                    c.setdefault("先生の種類", {})
                    c["先生の種類"][kind or "なし"] = c["先生の種類"].get(kind or "なし", 0) + 1
                    # I37e: 所（名詞+助詞+述語）のせいにするのは、文法・言い方の指摘だけ。つながり・足りないは行のまるごとの経験にだけ入れる
                    odd = ev.get("odd_part", "") if kind in ("文法", "言い方") else ""
                    got = bot.taught(True, odd, "" if a.no_model_reply else ev.get("suggested_reply", ""))
                    c["先生の指摘"] += 1
                    streak = 0
                    log.append(f"  - 先生（段 {stage}）: {ev.get('intervention_message', '')} ／ 経験: {'・'.join(k + ':' + w for k, w in got) or 'なし'}")
                elif ev is not None:                        # 褒めは、話し相手の次の行を見てから経験に入れる
                    pending = ev
                    log.append(f"  - 先生（段 {stage}）: {ev.get('intervention_message', '')} ／ 経験: 相手の次の行を見てから")
            hist.append(f"ボット: {line}")
            msgs.append(dict(role="user", content=line))
            human = ollama_chat(a.host, a.model, msgs) if a.model != "echo" else "うん、そうなんだ。"   # echo: 試し用
            if script.get((ci + 1, ti + 2)):                # I45a: 台本。この会話のこの往復だけ、話し相手の行を決まった行に差し替える
                human = script[(ci + 1, ti + 2)]
                c["台本"] = c.get("台本", 0) + 1
            if not human:                                   # 話し相手の本文が空（数に残し、会話はここで切る）
                c["相手の空"] += 1
                break
            msgs.append(dict(role="assistant", content=human))
            hist.append(f"相手: {human}")
            prev_bot = line
        if pending is not None:                             # 会話の終わり: 相手の指摘が無かったので褒めを入れる
            stage, streak = praise(pending, stage, streak)
        c["新規性"] = round(c["新規性"] / max(1, c["往復"]), 3)
        if learner is not None:
            learner.end_conv(ci + 1)
            c["大事さの経験"] = dict(learner.grain.n)
            c["入れる線の経験"] = learner.line_stats()
        if getattr(bot, "sb", None) is not None and bot.sb.learn_w:
            c["掛け目の経験"] = {k: dict(v) for k, v in bot.sb.src_xp.items()}
        if getattr(bot, "rev_log", None) is not None:
            c["逆引き"] = {k: v for k, v in bot.rev_log.items() if not k.startswith("作った:")}
            c["逆引きの形"] = sorted((k[4:] for k in bot.rev_log if k.startswith("作った:")))[:30]
        if imp is not None:
            imp.save()
        if mem is not None:
            mem.end_conv()
            c["記憶板"] = mem.summary()    # 返事 1 つあたりの新規性の点の平均
        log += ["", f"- 数: {json.dumps(c, ensure_ascii=False)}", ""]
        print(c, flush=True)
        if a.stats:
            with open(a.stats, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(exp="I35z" if a.teacher else "I35y", profile=bool(a.profile), novelty=a.novelty, nov_out=bot.nov_out if bot.nov else None, model=a.model, teacher=a.teacher, **c), ensure_ascii=False) + "\n")
        if a.log:
            os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
            with open(a.log, "w", encoding="utf-8") as f:
                f.write("\n".join(log))


if __name__ == "__main__":
    main()
