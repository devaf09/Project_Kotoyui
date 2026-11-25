"""I48v: 試しの場の話し相手（LLM）に、会話の初めの気分の設定と、往復ごとの自分の気分の書き出しを足す（chat_partner の --partner_mood）。
気分の欄は bot には渡さず、ログと stats にだけ書く（感情グレインの答え合わせ・学びの材料。試しの場の中だけの答え）。

気分の欄: 4 つの対の両側の強さ（0〜1）、合わせて 8 つ。対の片側だけでなく両側を別々に持つ（「うれしいけれど寂しい」のように両方ある状態を残す）。
置いた物: 気分の設定 5 通り（MOODS）と、8 つの名前（SIDES）。設定は会話ごとに、seed で並べ替えた順に当てる（どの設定も同じ回数になるように回す）。
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
import json
import random

SIDES = ("喜び", "悲しみ", "信頼", "嫌悪", "恐れ", "怒り", "驚き", "期待")
PAIRS = (("喜び", "悲しみ"), ("信頼", "嫌悪"), ("恐れ", "怒り"), ("驚き", "期待"))
MOODS = {
    "落ち着いている": "今日のあなたは、落ち着いていて、ふだんどおりの気分です。",
    "落ち込んでいる": "今日のあなたは、少し落ち込んでいます（仕事でうまくいかないことがあった）。話しているうちに気分が変わってもよい。",
    "怒っている": "今日のあなたは、少しいらいらしています（朝から嫌なことが続いた）。話しているうちに気分が変わってもよい。",
    "不安": "今日のあなたは、少し不安です（明日の予定がうまくいくか心配）。話しているうちに気分が変わってもよい。",
    "うれしい": "今日のあなたは、うれしい気分です（いいことがあった）。話しているうちに気分が変わってもよい。",
}
RATE = ("（会話の外で答える）今のあなたの気分を、次の 8 つそれぞれの強さ 0〜1 の数で答える。両方が同時にあってもよい（うれしいけれど寂しい、など）。"
        "出力は次の JSON だけ: {" + ", ".join(f'"{s}": 0〜1' for s in SIDES) + "}")


def order(n, seed):
    """会話 n 本に当てる気分の設定の並び（どの設定も同じ回数に近く、seed で並べ替える）。"""
    ks = [list(MOODS)[i % len(MOODS)] for i in range(n)]
    random.Random(f"mood:{seed}").shuffle(ks)
    return ks


def parse(text):
    j = text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""
    try:
        r = json.loads(j)
    except Exception:
        return None
    out = {}
    for s in SIDES:
        try:
            out[s] = min(1.0, max(0.0, float(r.get(s))))
        except (TypeError, ValueError):
            return None
    return out


def rate(chat, host, model, msgs):
    """話し相手の今の気分（8 つの強さ）。会話の続きとは別に聞き、msgs には足さない。chat = chat_partner.ollama_chat。"""
    q = list(msgs) + [dict(role="user", content=RATE)]
    return parse(chat(host, model, q, temperature=0.2, first_line=False, num_predict=512, fmt="json"))
