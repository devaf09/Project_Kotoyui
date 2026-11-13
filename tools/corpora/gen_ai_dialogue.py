"""仮のキャラクタを設定した AI 同士の会話を作る（PC の Ollama + Gemma）。v2（2026-11-12）。
1 行 1 発言の jsonl（source=ai_gemma, dialogue_id, session=日, turn, speaker, text, meta）。人の会話と混ぜず、評価に使わない。
使い方: python tools/corpora/gen_ai_dialogue.py --out <非公開の置き場>/data/synthetic/ai_dialogue_v2.jsonl --n 20
  （Ollama が --url で --model を動かしている前提。Gemma の出力で学んだモデルは Gemma の規約の「派生物」に当たる）
v0（このファイルの履歴にある最初の版）からの変更:
  temperature 0.9→0.7 / 日本語以外の文字が混じった返事は作り直す（3 回まで。駄目なら meta.foreign=true で残す）/
  となりは食べたり出かけたりしないと明記 / 2 日目以降は「数日ぶり」、1 日目は知り合いか初対面かを対話ごとに決める /
  呼び方を固定 / 話題を倍に / 覚え書きは前の日の分とまとめて 5 行まで / 2 人の組み合わせは全 6 組を順に回す
v1 からの変更（v2）:
  2 日目以降の書き出しの型を 8 つから日ごとに選ぶ（meta.opening）/ ほめ合いが続かないよう RULE に足す /
  英字の略語（大文字 2〜5 字、AI・SNS など）は外国語の判定から外す / 対話ごとに乱数を分ける（--start で分けて回せる）/
  出力に既にある対話は飛ばし、途中で切れた対話は消して作り直す / 「となり」の名前が崩れた返事も作り直す（meta.misname）
"""
import argparse, json, random as random_mod, urllib.request, os, itertools, unicodedata, re, collections

CHARS = {
    "となり": "あなたは「言結 となり」。コンピュータの中にいる話し相手。まだ外に出たことがなく、おしゃべりと相手を知ることが好き。丁寧すぎない「です・ます」で、短めに話す。"
              "体が無いので、食べたり飲んだり、どこかへ出かけたり、物にさわったりはしない（できない）。食べ物や出かけた話は、相手の話として聞いて質問する。",
    "みのり": "あなたは「みのり」。28 歳、地方の図書館で働く。猫を 1 匹飼っている。落ち着いた話し方で、くだけた口調。一人称は「私」。",
    "けんた": "あなたは「けんた」。19 歳の大学生、料理のアルバイトをしている。明るく、ため口で話す。一人称は「俺」。",
    "さくら": "あなたは「さくら」。45 歳、二人の子どもの母で、パートで事務をしている。世話好き。一人称は「あたし」。",
}
# 呼び方（話す人 → 相手）
CALL = {
    "となり": {"みのり": "みのりさん", "けんた": "けんたさん", "さくら": "さくらさん"},
    "みのり": {"となり": "となりさん", "けんた": "けんたくん", "さくら": "さくらさん"},
    "けんた": {"となり": "となり", "みのり": "みのりさん", "さくら": "さくらさん"},
    "さくら": {"となり": "となりちゃん", "みのり": "みのりちゃん", "けんた": "けんたくん"},
}
TOPICS = ["週末の予定", "最近食べたもの", "天気と季節", "仕事や学校の困りごと", "昔の思い出", "これからやってみたいこと", "家族やペット", "最近見た物・読んだ物",
          "仕事での失敗", "最近のニュース", "体調や睡眠", "買い物", "引っ越しや住まい", "趣味の道具", "苦手なもの", "季節の行事",
          "子どもの頃の遊び", "お金の使い方"]
RULE = ("相手と雑談している。返事は 1〜2 文の日本語の話し言葉だけで書く。名前や記号、説明、ト書きは付けない。英語など日本語以外の言葉は使わない。"
        "相手の話を受けてから自分のことも話し、ときどき質問する。前に話したことを覚えていれば自然に触れる。"
        "相手をほめたり気持ちを伝えたりするのは会話の中でたまにだけ。具体的な出来事・自分の意見・小さな意見の違いや冗談も入れる。")
# 2 日目以降の書き出しの型（その日の最初の発言だけに指示する）
OPENINGS = {
    "続き": "前に話したことの続きを、具体的に聞くところから始める。",
    "近況": "自分の最近あった具体的な出来事を話すところから始める。",
    "天気": "今日の天気や季節の変化に触れるところから始める。",
    "相談": "ちょっとした相談ごとを持ちかけるところから始める。",
    "見聞き": "最近見た物や聞いた話を話すところから始める。",
    "約束": "相手が前に言っていた予定や約束がどうなったか触れるところから始める。",
    "出来事": "今日あったちょっとした失敗や驚いたことを報告するところから始める。",
    "意見": "相手の意見を聞きたいちょっとした問いから始める。",
}
NO_GREETING = "「お久しぶり」「最近どう」「どう過ごしてる」のような決まった挨拶は使わない。"

def is_jp_char(ch):
    o = ord(ch)
    return (0x3040 <= o <= 0x30ff or 0x31f0 <= o <= 0x31ff or 0x3400 <= o <= 0x4dbf or 0x4e00 <= o <= 0x9fff
            or 0x3005 <= o <= 0x3007 or 0xff66 <= o <= 0xff9f)

def foreign_letters(text):
    """日本語の文字以外の「文字」（ラテン・ハングル・タイ・キリル・アラビア・デーヴァナーガリーなど）。記号・数字・句読点は数えない"""
    text = re.sub(r"(?<![A-Za-z])[A-Z]{2,5}(?![A-Za-z])", "", text)  # AI・SNS などの英字の略語は数えない
    return [ch for ch in text if unicodedata.category(ch).startswith("L") and not is_jp_char(ch)]

def chat(url, model, system, msgs, first_line=True, num_predict=120, temperature=0.7):
    body = {"model": model, "messages": [{"role": "system", "content": system}] + msgs, "stream": False,
            "think": False, "options": {"temperature": temperature, "num_predict": num_predict}}
    req = urllib.request.Request(url + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"].strip()
    return out.split("\n")[0] if first_line else out

def summarize(url, model, who, old, lines, temperature):
    text = "\n".join(f"{s}: {t}" for s, t in lines)
    prev = f"前の日までの覚え書き:\n{old}\n\n" if old else ""
    out = chat(url, model, "会話の覚え書きを作る。日本語だけで書く。", [{"role": "user", "content":
        f"{prev}今日の会話:\n{text}\n\n前の日までの覚え書きと今日の会話をまとめて、{who}が覚えておくべきこと（相手のこと・約束・話題）を"
        f"5 行以内の箇条書き（各行「・」で始める）で書いて。見出しは付けない。"}], first_line=False, num_predict=250, temperature=temperature)
    ls = [l.strip() for l in out.split("\n") if l.strip().startswith(("・", "-", "*", "●"))]
    if not ls:
        ls = [l.strip() for l in out.split("\n") if l.strip()]
    return "\n".join("・" + l.lstrip("・-*● ").strip() for l in ls[:5])

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--days", type=int, default=3)
ap.add_argument("--turns", type=int, default=16)
ap.add_argument("--model", default="gemma4:12b-it-qat")
ap.add_argument("--url", default="http://localhost:11434")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--start", type=int, default=0, help="何本目から作るか（2 枚の GPU で分けて回すとき）")
ap.add_argument("--tries", type=int, default=3, help="日本語以外が混じったときに作る回数の上限（最初の 1 回を含む）")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
# 続きから作る: 揃っている対話は飛ばし、途中で切れた対話の行は消す
done_ids = set()
if os.path.exists(a.out):
    rows = [l for l in open(a.out, encoding="utf-8") if l.strip()]
    cnt = collections.Counter(json.loads(l)["dialogue_id"] for l in rows)
    done_ids = {d for d, c in cnt.items() if c == a.days * a.turns}
    keep = [l for l in rows if json.loads(l)["dialogue_id"] in done_ids]
    if len(keep) != len(rows):
        open(a.out, "w", encoding="utf-8").writelines(keep)
        print(f"removed {len(rows) - len(keep)} lines of unfinished dialogues", flush=True)
    print(f"skip {len(done_ids)} finished dialogues", flush=True)
COMBOS = list(itertools.combinations(list(CHARS), 2))
stats = {"utt": 0, "retried_utt": 0, "retries": 0, "foreign_left": 0, "latin_seen": 0, "misname_seen": 0, "misname_left": 0}
# 「となり」の名前の崩れ（とりり・とりさん・とりなり など）。試しの 1 本で、みのりが「とりり」と呼び続けた
MISNAME = re.compile(r"(?<![ひ])とり(り|さん|ちゃん|なり|くん)")
with open(a.out, "a", encoding="utf-8") as f:
    for k in range(a.start, a.start + a.n):
        did = f"ai{a.seed:02d}v2_{k:03d}"
        if did in done_ids:
            continue
        random = random_mod.Random(f"{a.seed}-{k}")
        pair = list(COMBOS[k % len(COMBOS)])
        random.shuffle(pair)
        relation = random.choice(["知り合い", "初対面"])
        memo = {p: "" for p in pair}
        for day in range(a.days):
            topic = random.choice(TOPICS)
            opening = random.choice(list(OPENINGS)) if day > 0 else None
            hist = []
            for t in range(a.turns):
                me, you = pair[t % 2], pair[(t + 1) % 2]
                sys_ = f"{CHARS[me]}\n相手は「{you}」。相手のことは「{CALL[me][you]}」と呼ぶ（呼び方を変えない）。{RULE}\n"
                if day == 0:
                    sys_ += ("2 人は前からの知り合い。自己紹介はしない。" if relation == "知り合い"
                             else "2 人は今日が初対面。最初に軽く自己紹介してよい。")
                else:
                    sys_ += f"今日は {day + 1} 回目に話す日。前に話してから数日たっている。今日は新しく話しかけたところ。"
                sys_ += f"話題のきっかけ: {topic}。"
                if opening and t == 0:
                    sys_ += f"\n今日の話の始め方: {OPENINGS[opening]}{NO_GREETING}"
                if memo[me]:
                    sys_ += f"\n前に話したときの覚え書き:\n{memo[me]}"
                msgs = [{"role": "assistant" if s == me else "user", "content": x} for s, x in hist]
                if not msgs or msgs[-1]["role"] == "assistant":
                    msgs.append({"role": "user", "content": "（会話を始めて）" if not hist else "（続けて）"})
                tries, rejected = 0, []
                while True:
                    tries += 1
                    text = chat(a.url, a.model, sys_, msgs, temperature=a.temperature)
                    fl = foreign_letters(text)
                    mis = "となり" in pair and bool(MISNAME.search(text))
                    stats["misname_seen"] += mis
                    if any(unicodedata.name(c, "").startswith(("LATIN", "FULLWIDTH LATIN")) for c in fl):
                        stats["latin_seen"] += 1
                    if (not fl and not mis) or tries >= a.tries:
                        break
                    rejected.append(text)
                stats["utt"] += 1
                stats["retries"] += tries - 1
                stats["retried_utt"] += tries > 1
                meta = {"model": a.model, "topic": topic, "pair": pair, "relation": relation, "temperature": a.temperature, "tries": tries, "opening": opening}
                if mis:
                    meta["misname"] = True
                    stats["misname_left"] += 1
                if fl:
                    meta["foreign"] = True
                    stats["foreign_left"] += 1
                if rejected:
                    meta["rejected"] = rejected
                hist.append((me, text))
                f.write(json.dumps({"source": "ai_gemma", "dialogue_id": did, "session": day, "turn": t, "speaker": me,
                                    "text": text, "meta": meta}, ensure_ascii=False) + "\n")
            f.flush()
            if day + 1 < a.days:
                for p in pair:
                    memo[p] = summarize(a.url, a.model, p, memo[p], hist, a.temperature)
                print(f"  {did} day{day + 1} memo:", json.dumps(memo, ensure_ascii=False), flush=True)
        print(f"done {did} {pair} {relation}", json.dumps(stats, ensure_ascii=False), flush=True)
print("STATS", json.dumps(stats, ensure_ascii=False), flush=True)
