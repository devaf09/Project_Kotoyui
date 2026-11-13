"""仮のキャラクタを設定した AI 同士の会話を作る（PC の Ollama + Gemma で実行。試しの段階）。
持ち主が方針を変えると決めたときだけ使う（data_plan.md の「LLM に会話を作らせない」を変えることになる）。

使い方:
  python gen_ai_dialogue.py --out <非公開の置き場>/data/synthetic/ai_dialogue_v0.jsonl --n 20
  （Ollama が http://localhost:11434 で gemma4:12b-it-qat を動かしている前提）

作るもの: 2 人の会話を「日」で区切って 3 日分。日が変わるときに、それぞれの話者が前の日の会話の覚え書きを持って続ける。
1 行 1 発言の jsonl（source=ai_gemma, dialogue_id, session=日, turn, speaker, text, meta）。
人の会話のコーパスとは混ぜず、評価には使わない。Gemma の出力で学んだモデルは Gemma の規約の「派生物」に当たる。
"""
import argparse, json, random, urllib.request, os

CHARS = {
    "となり": "あなたは「言結 となり」。コンピュータの中にいる話し相手。まだ外に出たことがなく、おしゃべりと相手を知ることが好き。丁寧すぎない「です・ます」で、短めに話す。",
    "みのり": "あなたは「みのり」。28 歳、地方の図書館で働く。猫を 1 匹飼っている。落ち着いた話し方で、くだけた口調。一人称は「私」。",
    "けんた": "あなたは「けんた」。19 歳の大学生、料理のアルバイトをしている。明るく、ため口で話す。一人称は「俺」。",
    "さくら": "あなたは「さくら」。45 歳、二人の子どもの母で、パートで事務をしている。世話好き。一人称は「あたし」。",
}
TOPICS = ["週末の予定", "最近食べたもの", "天気と季節", "仕事や学校の困りごと", "昔の思い出", "これからやってみたいこと", "家族やペット", "最近見た物・読んだ物"]
RULE = ("相手と雑談している。返事は 1〜2 文の日本語の話し言葉だけで書く。名前や記号、説明、ト書きは付けない。"
        "相手の話を受けてから自分のことも話し、ときどき質問する。前に話したことを覚えていれば自然に触れる。")

def chat(url, model, system, msgs, first_line=True):
    body = {"model": model, "messages": [{"role": "system", "content": system}] + msgs, "stream": False,
            "think": False, "options": {"temperature": 0.9, "num_predict": 120}}
    req = urllib.request.Request(url + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"].strip()
    return out.split("\n")[0] if first_line else out

def summarize(url, model, who, lines):
    text = "\n".join(f"{s}: {t}" for s, t in lines)
    return chat(url, model, "会話の覚え書きを作る。", [{"role": "user", "content":
        f"次の会話から、{who}が覚えておくべきこと（相手のこと・約束・話題）を 3 行以内の箇条書きで書いて。\n{text}"}], first_line=False)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--days", type=int, default=3)
ap.add_argument("--turns", type=int, default=16, help="1 日あたりの発言数")
ap.add_argument("--model", default="gemma4:12b-it-qat")
ap.add_argument("--url", default="http://localhost:11434")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "a", encoding="utf-8") as f:
    for k in range(a.n):
        pair = random.sample(list(CHARS), 2)
        did = f"ai{a.seed:02d}_{k:03d}"
        memo = {p: "" for p in pair}
        for day in range(a.days):
            topic = random.choice(TOPICS)
            hist = []  # (speaker, text)
            for t in range(a.turns):
                me, you = pair[t % 2], pair[(t + 1) % 2]
                sys_ = f"{CHARS[me]}\n相手は「{you}」。{RULE}\n今日は {day + 1} 日目。話題のきっかけ: {topic}。"
                if memo[me]:
                    sys_ += f"\n前の日までの覚え書き:\n{memo[me]}"
                msgs = [{"role": "assistant" if s == me else "user", "content": x} for s, x in hist]
                if not msgs or msgs[-1]["role"] == "assistant":
                    msgs.append({"role": "user", "content": "（会話を始めて）" if not hist else "（続けて）"})
                text = chat(a.url, a.model, sys_, msgs)
                hist.append((me, text))
                f.write(json.dumps({"source": "ai_gemma", "dialogue_id": did, "session": day, "turn": t, "speaker": me,
                                    "text": text, "meta": {"model": a.model, "topic": topic, "pair": pair}}, ensure_ascii=False) + "\n")
            f.flush()
            for p in pair:
                memo[p] = (memo[p] + "\n" + summarize(a.url, a.model, p, hist)).strip()
        print(f"done {did} {pair}")
